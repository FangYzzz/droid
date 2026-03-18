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
from pathlib import Path

# ==== 新增：用于连接 FlowerVLA 推理 server ====
import requests
import json_numpy
from json_numpy import loads
json_numpy.patch()


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

    # Flower server(server.py)
    flower_host: str = "127.0.0.1"  # 所机器的 IP
    flower_port: int = 8003  # 端口

    # Residual server(train_residual_td3_flower.py)
    residual_host: str = "127.0.0.1"
    residual_port: int = 8005


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

def process_policy_images(args, curr_obs):
    primary_image = curr_obs[f"{args.external_camera}_image"]
    wrist_image = curr_obs["wrist_image"]

    primary_image = prepare_image_256(primary_image)
    wrist_image = prepare_image_256(wrist_image)  # padding

    primary_resized = image_tools.resize_with_pad(primary_image, 224, 224)
    wrist_resized = image_tools.resize_with_pad(wrist_image, 224, 224)

    return primary_resized, wrist_resized

def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None

    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        # elif args.right_camera_id in key and "left" in key:
        #     right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    if left_image is None:
        raise RuntimeError(f"Left image not found for camera id {args.left_camera_id}")
    if wrist_image is None:
        raise RuntimeError(f"Wrist image not found for camera id {args.wrist_camera_id}")

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

def build_query_payload(curr_obs, curr_action_base):
    return {
        "obs": {
            "observation.images.agentview": curr_obs["left_image"].astype(np.uint8),
            "observation.images.robot0_eye_in_left_hand": curr_obs["wrist_image"].astype(np.uint8),
            "observation.state": np.concatenate(
                [
                    curr_obs["joint_position"].astype(np.float32),
                    curr_obs["gripper_position"].astype(np.float32),
                ],
                axis=-1,
            ),
            "observation.base_action": curr_action_base.astype(np.float32),
        }
    }

def build_transition_payload( # Todo: obs 是否需要经过 process_policy_images 或者 scale
    prev_obs,
    curr_obs,
    prev_action_base,
    curr_action_base,
    prev_action,
    reward: float = 0.0,
    done: bool = False,
    info: dict | None = None
):
    obs = {
        "observation.images.agentview": prev_obs["left_image"].astype(np.uint8),
        "observation.images.robot0_eye_in_left_hand": prev_obs["wrist_image"].astype(np.uint8),
        "observation.state": np.concatenate(
            [
                prev_obs["joint_position"].astype(np.float32),
                prev_obs["gripper_position"].astype(np.float32),
            ],
            axis=-1,
        ),
        "observation.base_action": prev_action_base.astype(np.float32),  # Todo
    }
    next_obs = {
        "observation.images.agentview": curr_obs["left_image"].astype(np.uint8),
        "observation.images.robot0_eye_in_left_hand": curr_obs["wrist_image"].astype(np.uint8),
        "observation.state": np.concatenate(
            [
                curr_obs["joint_position"].astype(np.float32),
                curr_obs["gripper_position"].astype(np.float32),
            ],
            axis=-1,
        ),
        "observation.base_action": curr_action_base.astype(np.float32),
    }
   
    return {
        "obs": obs, 
        "next_obs": next_obs,
        "action": prev_action.astype(np.float32),  # Todo: torch.Tensor 不能被 JSON 直接序列化
        "reward": float(reward),
        "done": bool(done), 
        "info": {} if info is None else info,
    }

def get_action_base(primary_resized, wrist_resized, flower_server: str):
    # 调用 Flower server 的 /query，得到关节位置 + gripper
    with prevent_keyboard_interrupt():
        resp = requests.post(
            f"{flower_server}/query",
            json={
                "primary_image":primary_resized,  #primary_image.astype(np.uint8),  # Flower 期望 uint8 [H, W, 3]
                "wrist_image": wrist_resized,
            },
            # timeout=1.0,
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"[Flower] /query failed, status: {resp.status_code}, text: {resp.text}"
        )

    action = np.array(loads(resp.json()))  # numpy array
    # print("action chunk horizon: ",action.shape)
    # 兼容 (1,8) 或 (8,) 形状
    if action.ndim == 2 and action.shape[0] == 1:
        action = action[0]
    if action.ndim != 1:
        print(f"[Flower] unexpected action shape: {action.shape}")
    if action.shape[0] < 8:
        print(f"[Flower] action dim < 8, got {action.shape}")

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

    action_base = np.concatenate([joint_targets, [gripper_target]], axis=-1)

    return action_base

def get_action_residual(payload, residual_server: str):  # Todo： obs 是否经过 process_policy_images
    with prevent_keyboard_interrupt():
        resp = requests.post(
            f"{residual_server}/query",
            json=payload,  # todo
        )

    if resp.status_code != 200:
        raise RuntimeError(
            f"[Residual] /query failed, status: {resp.status_code}, text: {resp.text}"
        )

    action_residual = np.array(loads(resp.json()), dtype=np.float32)

    if action_residual.ndim == 2 and action_residual.shape[0] == 1:
        action_residual = action_residual[0]

    if action_residual.ndim != 1:
        raise RuntimeError(f"[Residual] unexpected action shape: {action_residual.shape}")
    if action_residual.shape[0] < 8:
        raise RuntimeError(f"[Residual] action dim < 8, got {action_residual.shape}")

    return action_residual

def add_transition_to_residual_server(payload, residual_server: str):
    with prevent_keyboard_interrupt():
        resp = requests.post(
            f"{residual_server}/add_transition",
            json=payload,
        )

    if resp.status_code != 200:
        raise RuntimeError(
            f"[Residual] /add_transition failed, status: {resp.status_code}, text: {resp.text}"
        )

def main(args: Args):
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment
    env = RobotEnv(action_space="joint_position", gripper_action_space="position")
    # env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
    
    flower_server = f"http://{args.flower_host}:{args.flower_port}"
    residual_server = f"http://{args.residual_host}:{args.residual_port}"
   
    # 记录结果的 DataFrame
    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    # 获取一次 robot_state，只是为了 sanity check
    robot_state, _ = env.get_state()
    print("Initial joint positions:", robot_state.get("joint_positions", "N/A"))
    
    while True:
        instruction = input("Enter instruction: ").strip()
        if len(instruction) == 0:
            print("Empty instruction, please input again.")
            continue
        
        # 在每个新指令开始前，重置 Flower server 的任务（/reset）
        requests.post(f"{flower_server}/reset", json={"text": instruction})
        requests.post(f"{residual_server}/reset", json={"text": instruction})

        # first_round = True
        prev_obs, prev_action_base, prev_action = None, None, None
        video_wrist, video_left = [], []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout with FlowerVLA... press Ctrl+C to stop early.")

        for t_step in bar:
            start_time = time.time()
            # if first_round:
            #     first_round = False
            #     continue
            try:
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    save_to_disk=False,  # save_to_disk=t_step == 0,
                )
                primary_resized, wrist_resized = process_policy_images(args, curr_obs)

                # 记录图像和视频帧
                # save_np_image(primary_resized, "inference_primary.png")
                # save_np_image(wrist_resized, "inference_wrist.png")
                video_wrist.append(curr_obs["wrist_image"])
                video_left.append(curr_obs[f"{args.external_camera}_image"])

                # Flower
                curr_action_base = get_action_base(primary_resized, wrist_resized, flower_server)
                # Residual
                query_payload = build_query_payload(curr_obs, curr_action_base)
                curr_action_residual = get_action_residual(query_payload, residual_server)
                curr_action = curr_action_base + curr_action_residual
                # curr_action = curr_action.astype(np.float32)

                env.step(curr_action)

                if prev_obs is not None and prev_action_base is not None and prev_action is not None:  # Todo: reward, terminated, truncated, info
                    reward = int(reward_from_gpt_label)  # 0-5
                    terminated = (reward == 1)  # terminated：任务本身的终止条件满足 
                    truncated = (t_step == args.max_timesteps - 1)  # truncated：被外部强制截断（时间上限等）
                    done = terminated | truncated
                    info = {}

                    transition_payload = build_transition_payload(
                        prev_obs=prev_obs,
                        curr_obs=curr_obs,
                        prev_action_base=prev_action_base,
                        curr_action_base=curr_action_base,
                        prev_action=prev_action,
                        reward=reward,
                        done=done,
                        info=info,
                    )
                    add_transition_to_residual_server(transition_payload, residual_server)

                prev_obs = curr_obs
                prev_action_base = curr_action_base
                prev_action = curr_action

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

        env.reset()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
