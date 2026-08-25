import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from typing import Optional
import io
import copy
import numpy as np
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation as R

faulthandler.enable()

# 控制频率（和原 DROID 一样）
DROID_CONTROL_FREQUENCY =15 # Hz


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "24285872"
    right_camera_id: Optional[str] = None #"21664821"
    wrist_camera_id: str = "11022812"

    # 使用哪路外部相机作为 primary_image / # Policy parameters
    external_camera: Optional[str] = "left"

    # Rollout parameters
    max_timesteps: int = 300

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

def decode_png_bytes(png_bytes: bytes) -> np.ndarray:
    """Decode PNG bytes to uint8 RGB image (H, W, 3)."""
    img = Image.open(io.BytesIO(png_bytes))
    img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)
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
    #####path###
    # task1
    # path = "/home/yuan/self_vla/tele_op/lerobot/dataset_7050_7050_5050_7050/data/chunk-000/episode_000221.parquet"
    # path = "/home/yuan/self_vla/tele_op/lerobot/dataset_7050_7050_5050_7050/data/chunk-000/episode_000221.parquet"
    # task2
    # path = "/home/yuan/self_vla/tele_op/lerobot/dataset_7050_7050_5050_7050/data/chunk-000/episode_000135.parquet"  # 86-135
    # path = "/home/yuan/self_vla/tele_op/lerobot/dataset_7050_7050_5050_7050/data/chunk-000/episode_000156.parquet"  # 138-187
    # task3
    path = "/home/yuan/self_vla/tele_op/lerobot/dataset_7050_7050_5050_7050/data/chunk-000/episode_000388.parquet"  # 368-417
    # path = "/home/yuan/self_vla/tele_op/lerobot/dataset_7050_7050_5050_7050/data/chunk-000/episode_000279.parquet"  # 250-279
    # task4
    # path = "/home/yuan/self_vla/tele_op/lerobot/dataset_7050_7050_5050_7050/data/chunk-000/episode_000300.parquet"  # 268-317 418-437
    # path = "/home/yuan/self_vla/tele_op/lerobot/dataset_7050_7050_5050_7050/data/chunk-000/episode_000424.parquet"  # 330-359
    df = pd.read_parquet(path)

    # Replay the recorded end-effector targets.  Their layout is
    # [x, y, z, qx, qy, qz, qw, gripper].
    eef_actions = np.stack(
        [np.asarray(x, dtype=np.float32) for x in df["eef_actions"].tolist()],
        axis=0
    )
    if eef_actions.ndim != 2 or eef_actions.shape[1] != 8:
        raise ValueError(
            "Expected eef_actions with shape [frames, 8] containing "
            "[x, y, z, qx, qy, qz, qw, gripper], "
            f"but got {eef_actions.shape}"
        )
    if not np.all(np.isfinite(eef_actions)):
        raise ValueError("eef_actions contains NaN or Inf")
    # print(eef_actions.shape)
    # Video saving disabled: do not decode and retain the recorded wrist frames.
    
    # 初始化 Panda 环境
    env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")

    # 记录结果的 DataFrame
    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    # 获取一次 robot_state，只是为了 sanity check
    robot_state, _ = env.get_state()
    # Video saving disabled: do not retain live camera frames.
    
    for action in eef_actions:

        # Prepare to save video of rollout
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout with FlowerVLA... press Ctrl+C to stop early.")
        # Camera capture was only used for saving the replay video and is disabled.
        # ================== image dump setup ==================
        # IMAGE_ROOT = Path("/home/yuan/flower_vla_pret/videos/images")
        # IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # image_dir = IMAGE_ROOT / timestamp
        # image_dir.mkdir(parents=True, exist_ok=True)

        # IMAGE_ROOT = Path("/home/yuan/flower_vla_pret/videos/images")
        # IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
        # frame_path = IMAGE_ROOT / "latest.png"
        # last_save_time = 0.0

        # for t_step in bar:
        start_time = time.time()
        try:
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

            # Match main_delta_absolute_eef_2cams.py's Cartesian controller:
            # RobotEnv expects [x, y, z, roll, pitch, yaw, gripper].  Positions
            # in the recorded eef_actions are already absolute, so unlike the
            # policy's delta-position output they must not be added to the
            # current end-effector position.
            position_target = action[:3].astype(np.float32, copy=True)
            # position_target[2] = max(position_target[2], 0.225)

            quat_target = action[3:7].astype(np.float64, copy=True)
            quat_norm = np.linalg.norm(quat_target)
            if quat_norm < 1e-12:
                raise ValueError("Received a zero-norm end-effector quaternion")
            quat_target /= quat_norm
            if quat_target[3] < 0:
                quat_target *= -1.0
            rpy_target = R.from_quat(quat_target).as_euler("xyz", degrees=False)

            gripper_target = 1.0 if action[7] > 0.5 else 0.0
            robot_action = np.concatenate(
                [position_target, rpy_target, [gripper_target]], axis=-1
            )
            env.step(robot_action)

            # Sleep to match DROID data collection frequency
            elapsed_time = time.time() - start_time
            if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            # print("elapsed_time: ", elapsed_time)

        except KeyboardInterrupt:
            print("Rollout interrupted by user.")
            break
        except Exception as e:
            print(f"[ERROR] during rollout step: {e}")
            break
    
    # Video saving disabled.  In particular, do not invoke MoviePy/FFmpeg here.
    # os.makedirs("results", exist_ok=True)
    # timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    # csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    # df.to_csv(csv_filename)
    # print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None

    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if "right_cam" in key:
            right_image = image_observations[key]
        elif "wrist_cam" in key:
            wrist_image = image_observations[key]

    missing_cameras = []
    if right_image is None:
        missing_cameras.append("right_cam")
    if wrist_image is None:
        missing_cameras.append("wrist_cam")
    if missing_cameras:
        raise KeyError(
            f"Missing camera observations {missing_cameras}; "
            f"available keys: {list(image_observations.keys())}"
        )

    # Drop the alpha dimension
    # left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # BGR -> RGB
    # left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    if save_to_disk:
        # combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        combined_image = np.concatenate([wrist_image, right_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        # "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
