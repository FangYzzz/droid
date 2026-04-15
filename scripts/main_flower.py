import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from typing import Optional

import copy
import numpy as np
import pandas as pd
from PIL import Image
from moviepy.editor import ImageSequenceClip
from droid.robot_env import RobotEnv
import tqdm
import tyro
import cv2

from openpi_client import image_tools

# ==== 新增：用于连接 FlowerVLA 推理 server ====
import requests
import json_numpy
from json_numpy import loads
json_numpy.patch()
# ===========================================
from pathlib import Path

faulthandler.enable()

# 控制频率（和原 DROID 一样）
DROID_CONTROL_FREQUENCY = 6  # Hz


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "24285872"
    right_camera_id: str = None #"21664821"
    wrist_camera_id: str = "11022812"

    # 使用哪路外部相机作为 primary_image / # Policy parameters
    external_camera: Optional[str] = "left"

    # Rollout parameters
    max_timesteps: int = 180  # 360  # 3000

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # 改成跑 server.py 那台机器的 IP
    remote_port: int = 8008       # server.py 里用的端口（你现在是 8003）


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt   

def save_np_image(img: np.ndarray, path: str):
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3
    Image.fromarray(img).save(path)

def prepare_image_256(img, size=(256, 256)):
    """中心裁剪成正方形，再缩放到指定大小 (默认 256x256)，输出 RGB uint8。"""
    if img is None:
        return None
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = img[y0:y0 + side, x0:x0 + side]
    out = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return out.astype(np.uint8, copy=False)

def main(args: Args):
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment
    env = RobotEnv(action_space="joint_position", gripper_action_space="position")
    # env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")

    # 记录结果的 DataFrame
    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    # 获取一次 robot_state，只是为了 sanity check
    robot_state, _ = env.get_state()
    print("Initial joint positions:", robot_state.get("joint_positions", "N/A"))
    
    # SERVER = "http://<server_ip>:8003"
    SERVER = "http://0.0.0.0:8008"
    
    while True:
        instruction = input("Enter instruction: ").strip()
        if len(instruction) == 0:
            print("Empty instruction, please input again.")
            continue
        
        # 在每个新指令开始前，重置 Flower server 的任务（/reset）
        requests.post(f"{SERVER}/reset", json={"text": instruction})
        first_round = True
        # Prepare to save video of rollout
        video_wrist, video_left = [], []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout with FlowerVLA... press Ctrl+C to stop early.")

        for t_step in bar:
            start_time = time.time()
            try:
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    save_to_disk=False,  # save_to_disk=t_step == 0,
                )

                # 记录视频帧
                video_wrist.append(curr_obs["wrist_image"])
                video_left.append(curr_obs[f"{args.external_camera}_image"])

                # 构造发给 Flower 的图片 payload
                primary_image = curr_obs[f"{args.external_camera}_image"]
                wrist_image = curr_obs["wrist_image"]
                # joint_state = curr_obs["joint_position"]

                # ================== save external camera frame ==================
                # frame_path = image_dir / f"{t_step:06d}.png"
                # Image.fromarray(wrist_image).save(frame_path)

                # now = time.time()
                # if now - last_save_time >= 3.0:
                #     Image.fromarray(primary_image).save(frame_path)  # 每次覆盖同一个文件
                #     last_save_time = now

                # 调用 Flower server 的 /query，得到关节位置 + gripper
                primary_image = prepare_image_256(primary_image)
                wrist_image = prepare_image_256(wrist_image)  # padding
                primary_resized = image_tools.resize_with_pad(primary_image, 224, 224)
                wrist_resized = image_tools.resize_with_pad(wrist_image, 224, 224)
                save_np_image(primary_resized, "inference_primary.png")
                save_np_image(wrist_resized, "inference_wrist.png")
                with prevent_keyboard_interrupt():
                    resp = requests.post(
                        f"{SERVER}/query",
                        json={
                            "primary_image":primary_resized,              #primary_image.astype(np.uint8),  # Flower 期望 uint8 [H, W, 3]
                            "wrist_image": wrist_resized,
                        },
                        # timeout=1.0,
                    )
                if resp.status_code != 200:  # 200: HTTP 返回码, 请求成功（server 正常返回 action）
                    print(f"[Flower] /query failed, status: {resp.status_code}, text: {resp.text}")
                    break

                action = np.array(loads(resp.json()))  # numpy array
                # print("action chunk horizon: ",action.shape)
                # 兼容 (1,8) 或 (8,) 形状
                if action.ndim == 2 and action.shape[0] == 1:
                    action = action[0]
                if action.ndim != 1:
                    print(f"[Flower] unexpected action shape: {action.shape}")
                    break

                if action.shape[0] < 8:
                    print(f"[Flower] action dim < 8, got {action.shape}")
                    break

                # [7 joints, 1 gripper]
                joint_targets = action[:7].astype(np.float32)
                gripper_target = action[7]
                # print("curr_obs[joint_position]: ", curr_obs["joint_position"])
                # print("joint_targets: ", action[:7])
                # print("gripper_target", gripper_target)
                if gripper_target > 0.8:  # 0.8
                    gripper_target=1.0
                else:
                    gripper_target = 0.0
                if first_round:
                    first_round = False
                    continue
                robot_action = np.concatenate([joint_targets, [gripper_target]], axis=-1)
                env.step(robot_action)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)

            except KeyboardInterrupt:
                print("Rollout interrupted by user.")
                break
            except Exception as e:
                print(f"[ERROR] during rollout step: {e}")
                break

        # ================== 后处理：保存视频 & 人工评价 ==================
        save_filename = "None"
        if input("Save videos? (enter y or n) ").lower() == "y":
            os.makedirs("videos", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

            video_wrist = np.stack(video_wrist)
            save_filename = os.path.join("videos", f"video_{timestamp}_wrist")
            ImageSequenceClip(list(video_wrist), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

            video_left = np.stack(video_left)
            save_filename = os.path.join("videos", f"video_{timestamp}_left")
            ImageSequenceClip(list(video_left), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        # success: float | None = None
        # while success is None:
        #     s_input = input(
        #         "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100: "
        #     )
        #     if s_input == "y":
        #         success = 1.0
        #     elif s_input == "n":
        #         success = 0.0
        #     else:
        #         try:
        #             val = float(s_input) / 100.0
        #             if 0.0 <= val <= 1.0:
        #                 success = val
        #             else:
        #                 print("Please enter a number between 0 and 100.")
        #         except Exception:
        #             print("Invalid input, try again.")

        # df = pd.concat([
        #     df,
        #     pd.DataFrame([{
        #         "success": success,
        #         "duration": t_step,
        #         "video_filename": save_filename,
        #     }])
        # ], ignore_index=True)

        # if input("Do one more eval? (enter y or n): ").lower() != "y":
        #     break

        env.reset()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None

    # for key in image_observations:
    #     # Note the "left" below refers to the left camera in the stereo pair.
    #     # The model is only trained on left stereo cams, so we only feed those.
    #     if args.left_camera_id in key and "left" in key:
    #         left_image = image_observations[key]
    #     # elif args.right_camera_id in key and "left" in key:
    #     #     right_image = image_observations[key]
    #     elif args.wrist_camera_id in key and "left" in key:
    #         wrist_image = image_observations[key]
    for key in image_observations:
        print(key)
        if "left_cam" == key:
            left_image = image_observations[key]
        elif "wrist_cam" == key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension
    left_image = left_image[..., :3]
    # right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # BGR -> RGB
    left_image = left_image[..., ::-1]
    # right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    if save_to_disk:
        # combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        combined_image = np.concatenate([left_image, wrist_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
