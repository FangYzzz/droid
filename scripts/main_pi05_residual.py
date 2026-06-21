import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from typing import Optional, Any, Dict
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import torch

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
from openpi_client import websocket_client_policy
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import requests
import json_numpy
from json_numpy import loads
json_numpy.patch()

import contextlib
import signal
import threading


faulthandler.enable()

# DROID_CONTROL_FREQUENCY = 15  # Hz


@dataclasses.dataclass
class Args:
    max_timesteps: int = 150  # 90  # 180   

    # GPT server(task_reward_generation_zedx.py)
    gpt_host: str = "127.0.0.1"  # 机器 IP
    gpt_port: int = 8007  # 端口

    # PI05 server(server.py)
    pi05_host: str = "127.0.0.1"  # 机器 IP
    pi05_port: int = 8008  # 端口

def json_response(obj):
    return JSONResponse(json_numpy.dumps(obj))

class ResidualServer:
    def __init__(self, env=None, args=None, gpt_server=None, pi05_client=None):  # flower_server
        self.env = env
        self.args = args
        self.gpt_server = gpt_server
        self.pi05_client = pi05_client
        # self.text = "pick up the tomato and place it into the bowl"
        self.text = "pick up the cube and place it into the bowl"
        self.max_timesteps = args.max_timesteps
        self.t_step = 0
        self.round = 0
        self.reward = 0.0
    
    def run(self, port = 8009, host = "127.0.0.1"):
        self.app = FastAPI()
        self.app.post("/query_offline_action_base")(self.get_offline_action_base)
        self.app.post("/query_transition")(self.get_transition)
        self.app.post("/reset")(self.reset)
        uvicorn.run(self.app, host=host, port=port)

    def reset(self):
        self.t_step = 0
        self.env.reset()  # 1.593s
        
        obs = _extract_observation(  # 0.000s
            self.args,
            self.env.get_observation(),
            save_to_disk=False,
        )
        obs_left = obs["left_image"]
        obs_right = obs["right_image"]
        obs_wrist = obs["wrist_image"]
        eef_pose = obs["cartesian_position"] # [x, y, z, roll, pitch, yaw]
        gripper_position = obs["gripper_position"]
        left_resized, right_resized, wrist_resized = process_policy_images(self.args, obs_left, obs_right, obs_wrist)
        
        if self.round != 0:
            self.reward =  float(self.reward_generation(obs_right))  # 0 or 1
            print("reward:", self.reward)
            time.sleep(5)

        print(f"---------------------------- trajectory {self.round} ----------------------------")
        self.text, self.round = self.task_generation(obs_right)  # 6.860s
        print("task:", self.text)
        action_base = self.get_online_action_base(left_resized, right_resized, wrist_resized, eef_pose, gripper_position)  # 0.093s

        #----------------------------rpy -> quat----------------------------#
        cartesian_position = np.asarray(eef_pose, dtype=np.float32)
        pos = cartesian_position[:3]
        rpy = cartesian_position[3:6]

        quat = R.from_euler('xyz', rpy, degrees=False).as_quat()  # [qx, qy, qz, qw]

        # 可选：统一四元数符号，避免跳变
        if quat[3] < 0:
            quat = -quat

        eef_position_quat = np.concatenate([pos, quat], axis=-1)
        #-------------------------------------------------------------------#

        return {
            "obs_left": left_resized.tolist(),
            "obs_right": right_resized.tolist(),
            "obs_wrist": wrist_resized.tolist(),
            "eef_position": eef_position_quat.tolist(),  # 7
            "gripper_position": obs["gripper_position"].tolist(),
            "action_base": action_base.tolist(),
            "text": self.text,
        }

    def task_generation(self, obs_right):  # 调用时除了第一次 obs=obs，其余 obs=None
        resp = requests.post(
            f"{self.gpt_server}/query_task",
            json={
                "img": obs_right,
            },
        )
        resp.raise_for_status()
        raw = resp.json()

        task = raw["task"]
        round = raw["round"]

        return task, round

    def reward_generation(self, obs_right):
        resp = requests.post(
            f"{self.gpt_server}/query_reward",
            json={
                "img": obs_right,
            },
            # timeout=1.0,
        )
        resp.raise_for_status()
        raw = resp.json()
        
        reward = raw["reward"]
        return reward
    
        # reward = np.array(loads(resp.json()))
        # return json_response(reward)

    def get_offline_action_base(self, payload: Dict[Any, Any]):
        obs_left = payload["exterior_image_1_left"]
        obs_right = payload["exterior_image_2_left"]
        obs_wrist = payload["wrist_image_left"]
        eef_pose = payload["eef_position"]  # [x, y, z, qx, qy, qz, qw]
        gripper_position = payload["gripper_position"]
        left_resized, right_resized, wrist_resized = process_policy_images(self.args, obs_left, obs_right, obs_wrist)
        if payload["query_action_base"]:
            if eef_pose.ndim == 2 :
                eef_pose = eef_pose.squeeze(0)
            
            request_data = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(left_resized, 224, 224),
                "observation/wrist_image_left": image_tools.resize_with_pad(right_resized, 224, 224),
                "observation/exterior_image_2_left": image_tools.resize_with_pad(wrist_resized, 224, 224),
                "observation/eef_position": eef_pose,
                "observation/gripper_position": gripper_position,
                "prompt": self.text,  # instruction
            }

            pred_action_chunk = self.pi05_client.infer(request_data)["actions"]
            assert pred_action_chunk.shape == (50, 8) # 10,8
            # action = pred_action_chunk[0]
            action = pred_action_chunk[:20]  # 20
            action = action[::2]

            action = np.asarray(action).copy()
            action[:, -1] = (action[:, -1] > 0.8).astype(action.dtype) # 0.6 0.5

            # 如果前3维是 delta position，就转成 absolute position
            action[:,:3] = action[:,:3] + eef_pose[:3]  # [20,8] todo!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            action = np.asarray(action, dtype=np.float32) 
        else:
            action = np.zeros((50,8))
        # # 归一化 quaternion，避免数值漂
        # q_action = action[3:7]
        # norm = np.linalg.norm(q_action, axis=-1, keepdims=True)
        # q_action = q_action / np.clip(norm, 1e-12, None)

        # # 可选：固定 qw 为正，减少四元数双解抖动
        # sign = -1.0 if q_action[3] < 0 else 1.0
        # q_action = q_action * sign
        # action[3:7] = q_action

        # action_base = np.concatenate([action[:6], gripper], axis=-1)   # shape = (8,)
        return json_response(action)

    def get_online_action_base(self, left_resized, right_resized, wrist_resized, eef_pose, gripper_position):
        eef_rpy = eef_pose[3:6]  # [x, y, z, roll, pitch, yaw]
        eef_quat = R.from_euler('xyz', eef_rpy, degrees=False).as_quat()
        eef_pose = np.concatenate([eef_pose[:3], eef_quat], axis=-1)
                
        request_data = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(left_resized, 224, 224),
            "observation/wrist_image_left": image_tools.resize_with_pad(right_resized, 224, 224),
            "observation/exterior_image_2_left": image_tools.resize_with_pad(wrist_resized, 224, 224),
            "observation/eef_position": eef_pose,
            "observation/gripper_position": gripper_position,
            "prompt": self.text,  # instruction
        }
        # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
        # Ctrl+C will be handled after the server call is complete
        pred_action_chunk = self.pi05_client.infer(request_data)["actions"]
        assert pred_action_chunk.shape == (50, 8) # 10,8
        action = pred_action_chunk[:20]  # 20
        action = action[::2]

        # gripper binary
        # if action[:,-1].item() > 0.5:
        #     action = np.concatenate([action[:-1], np.ones((1,))])
        #     gripper = np.ones((1,))
        # else:
        #     action = np.concatenate([action[:-1], np.zeros((1,))])
        #     gripper = np.zeros((1,))
        
        action = np.asarray(action).copy()
        action[:, -1] = (action[:, -1] > 0.8).astype(action.dtype) # 0.6 0.5

        # 如果前3维是 delta position，就转成 absolute position
        action[:,:3] = action[:,:3] + eef_pose[:3]  # [20,8] todo!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        action = np.asarray(action, dtype=np.float32) 

        return action

    def get_transition(self, payload: Dict[Any, Any]):
        combined_action = np.asarray(payload["combined_action"], dtype=np.float32)  # (8,)
        query_action_base = payload["query_action_base"]

        #----------------------------quat -> rpy----------------------------#
        pos = combined_action[:3]
        q_action = combined_action[3:7]
        gripper = combined_action[7:]

        # 触地保护
        z_min = 0.225
        pos[2] = max(pos[2], z_min)
        
        norm = np.linalg.norm(q_action, keepdims=True)
        q_action = q_action / np.clip(norm, 1e-12, None)
        sign = np.where(q_action[..., 3:4] < 0, -1.0, 1.0)
        q_action = q_action * sign

        # if q_action[3] < 0:
        #     q_action = -q_action

        rpy_cmd = R.from_quat(q_action).as_euler('xyz', degrees=False)
        combined_action = np.concatenate([pos, rpy_cmd, gripper], axis=-1)
        # print("combined_action: ", combined_action)
        #-------------------------------------------------------------------#

        self.env.step(combined_action)
        self.t_step += 1
        print("t_step:::::::::::::::::::::::", self.t_step)
        
        terminated = False  # terminated：任务本身的终止条件满足 TODO: 每隔一段时间请求一次 gpt 生成 reward
        truncated = (self.t_step >= self.args.max_timesteps - 1)  # truncated：被外部强制截断（时间上限等）
        done = terminated | truncated
        
        if done:
            reset_obs = self.reset()
            return {
                "obs_left": reset_obs["obs_left"],
                "obs_right": reset_obs["obs_right"],
                "obs_wrist": reset_obs["obs_wrist"],
                "eef_position": reset_obs["eef_position"],
                "gripper_position": reset_obs["gripper_position"],
                "next_action_base": reset_obs["action_base"],
                "reward": self.reward,
                "done": done,
            }
        else:
            next_obs = _extract_observation(
                self.args,
                self.env.get_observation(),
                save_to_disk=False,
            )
            obs_left = next_obs["left_image"]
            obs_right = next_obs["right_image"]
            obs_wrist = next_obs["wrist_image"]
            eef_pose = next_obs["cartesian_position"] # [x, y, z, roll, pitch, yaw]
            gripper_position = next_obs["gripper_position"]
            left_resized, right_resized, wrist_resized = process_policy_images(self.args, obs_left, obs_right, obs_wrist)

            if query_action_base:
                next_action_base = self.get_online_action_base(left_resized, right_resized, wrist_resized, eef_pose, gripper_position)
            else:
                next_action_base = None

            #----------------------------rpy -> quat----------------------------#
            cartesian_position = np.asarray(next_obs["cartesian_position"], dtype=np.float32)
            pos = cartesian_position[:3]
            rpy = cartesian_position[3:6]
            quat = R.from_euler('xyz', rpy, degrees=False).as_quat()  # [qx, qy, qz, qw]

            # 可选：统一四元数符号，避免跳变
            if quat[3] < 0:
                quat = -quat

            eef_position_quat = np.concatenate([pos, quat], axis=-1)
            #-------------------------------------------------------------------#

            return {
                "obs_left": left_resized.tolist(),
                "obs_right": right_resized.tolist(),
                "obs_wrist": wrist_resized.tolist(),
                "eef_position": eef_position_quat.tolist(),  # 7
                "gripper_position": gripper_position.tolist(),
                # "next_action_base": next_action_base.tolist(),
                "next_action_base": None if next_action_base is None else next_action_base.tolist(),
                "reward": 0.0,
                "done": done,
            }


# @contextlib.contextmanager
# def prevent_keyboard_interrupt():
#     """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
#     interrupted = False
#     original_handler = signal.getsignal(signal.SIGINT)

#     def handler(signum, frame):
#         nonlocal interrupted
#         interrupted = True

#     signal.signal(signal.SIGINT, handler)
#     try:
#         yield
#     finally:
#         signal.signal(signal.SIGINT, original_handler)
#         if interrupted:
#             raise KeyboardInterrupt   

@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Only install SIGINT handler in main thread."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

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

def prepare_image_256(img, size=(256, 256)):  # TODO: 一步到位
    """中心裁剪成正方形，再缩放到指定大小 (默认 256x256)，输出 RGB uint8。"""
    if img is None:
        print("❌ img is None")
        return None
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = img[y0:y0 + side, x0:x0 + side]
    out = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return out.astype(np.uint8, copy=False)

def to_hwc(img):
    if img is None:
        return None

    img = np.asarray(img)

    # (1, 3, H, W) -> (3, H, W)
    if img.ndim == 4:
        if img.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got shape={img.shape}")
        img = img[0]

    # (3, H, W) -> (H, W, 3)
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))

    if img.ndim != 3:
        raise ValueError(f"Invalid image ndim={img.ndim}, shape={img.shape}")

    return img

def process_policy_images(args: Args, obs_left, obs_right, obs_wrist):
    obs_left = prepare_image_256(to_hwc(obs_left))
    obs_right = prepare_image_256(to_hwc(obs_right))
    obs_wrist = prepare_image_256(to_hwc(obs_wrist))  # padding

    left_resized = image_tools.resize_with_pad(obs_left, 224, 224)
    right_resized = image_tools.resize_with_pad(obs_right, 224, 224)
    wrist_resized = image_tools.resize_with_pad(obs_wrist, 224, 224)

    return left_resized, right_resized, wrist_resized

def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if "left_cam" in key:
            left_image = image_observations[key]
        elif "right_cam" in key:
            right_image = image_observations[key]
        elif "wrist_cam" in key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension
    left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        Image.fromarray(combined_image).save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }

def main(args: Args):
    # Initialize the Panda environment
    env = RobotEnv(action_space="cartesian_position", gripper_action_space="position") # cartesian_position
    robot_state, _ = env.get_state()

    # GPT server - Rollout client
    gpt_server = f"http://{args.gpt_host}:{args.gpt_port}"
    # PI05 server - Rollout client
    # pi05_server = f"http://{args.pi05_host}:{args.pi05_port}"
    pi05_client = websocket_client_policy.WebsocketClientPolicy(args.pi05_host, args.pi05_port)
    # Rollout server - Residual training client
    training_server = ResidualServer(env, args, gpt_server, pi05_client)
    training_server.run(host="127.0.0.1", port=8009)
   
    # 记录结果的 DataFrame
    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    # 获取一次 robot_state，只是为了 sanity check
    robot_state, _ = env.get_state()
    print("Initial joint positions:", robot_state.get("joint_positions", "N/A"))


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)




    # get_action_base():
    #     调用 flower server 的 /query，得到关节位置 + gripper
    #     with prevent_keyboard_interrupt():
    #         resp = requests.post(
    #             f"{self.flower_server}/query_flower",
    #             json={
    #                 "left_image": left_resized,  #primary_image.astype(np.uint8),  # flower 期望 uint8 [H, W, 3]
    #                 "right_image": right_resized,
    #                 "wrist_image": wrist_resized,
    #             },
    #             # timeout=1.0,
    #         )
    #     if resp.status_code != 200:
    #         raise RuntimeError(
    #             f"[flower] /query failed, status: {resp.status_code}, text: {resp.text}"
    #         )

    #     action = np.array(loads(resp.json()))  # numpy array
    #     # print("action chunk horizon: ",action.shape)
    #     # 兼容 (1,8) 或 (8,) 形状
    #     if action.ndim == 2 and action.shape[0] == 1:
    #         action = action[0]
    #     if action.ndim != 1:
    #         print(f"[pi05] unexpected action shape: {action.shape}")
    #     if action.shape[0] < 8:
    #         print(f"[pi05] action dim < 8, got {action.shape}")

    #     # [7 joints, 1 gripper]
    #     joint_targets = action[:7].astype(np.float32)
    #     gripper_target = action[7]
    #     # print("curr_obs[joint_position]: ", curr_obs["joint_position"])
    #     # print("joint_targets: ", action[:7])
    #     # print("gripper_target", gripper_target)
    #     if gripper_target > 0.8:  # 0.8
    #         gripper_target=1.0
    #     else:
    #         gripper_target = 0.0

    #     action_base = np.concatenate([joint_targets, [gripper_target]], axis=-1)


    # def get_offline_action_base(self, payload: Dict[Any, Any]):
#         ### just for test rl
#         # action_base = np.zeros((1,8))
#         # return json_response(action_base)

#         obs_left = payload["exterior_image_1_left"]
#         obs_right = payload["exterior_image_2_left"]
#         obs_wrist = payload["wrist_image_left"]
#         left_resized, right_resized, wrist_resized = process_policy_images(self.args, obs_left, obs_right, obs_wrist)

#         # Rollout parameters
#         actions_from_chunk_completed = 0
#         pred_action_chunk = None
        
#         if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
#             # robot_state,_ = self.env.get_state()
#             # eef_state = copy.deepcopy(robot_state["cartesian_position"])
#             # eef_state = copy.deepcopy(payload["eef_position"])
#             eef_pose = payload["eef_position"]
#             # eef_rpy = eef_pose[3:6]
#             # eef_quat = R.from_euler('xyz', eef_rpy, degrees=False).as_quat()
#             # eef_pose = np.concatenate([eef_pose[:3], eef_quat], axis=-1)
            
#             actions_from_chunk_completed = 0
            
#             request_data = {
#                 "observation/exterior_image_1_left": image_tools.resize_with_pad(left_resized, 224, 224),
#                 "observation/wrist_image_left": image_tools.resize_with_pad(right_resized, 224, 224),
#                 "observation/exterior_image_2_left": image_tools.resize_with_pad(wrist_resized, 224, 224),
#                 "observation/eef_position": eef_pose,
#                 "observation/gripper_position": payload["gripper_position"],
#                 "prompt": self.text,  # instruction
#             }
#             # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
#             # Ctrl+C will be handled after the server call is complete
#             # with prevent_keyboard_interrupt():
#             #     # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
#             #     pred_action_chunk = self.pi05_client.infer(request_data)["actions"]
#             pred_action_chunk = self.pi05_client.infer(request_data)["actions"]
#             assert pred_action_chunk.shape == (50, 8) # 10,8

#         action = pred_action_chunk[actions_from_chunk_completed]
#         actions_from_chunk_completed += 1  # ???????????不需要

#         if action[-1].item() > 0.5:
#             action = np.concatenate([action[:-1], np.ones((1,))])
#             gripper = np.ones((1,))
#         else:
#             # action[-1] = 0.0
#             action = np.concatenate([action[:-1], np.zeros((1,))])
#             gripper = np.zeros((1,))
        
#         # 如果前3维是 delta position，就转成 absolute position
#         action[:3] = action[:3] + eef_state[:3]  # todo!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#         # # 归一化 quaternion，避免数值漂
#         # q_action = action[3:7]
#         # norm = np.linalg.norm(q_action, axis=-1, keepdims=True)
#         # q_action = q_action / np.clip(norm, 1e-12, None)

#         # # 可选：固定 qw 为正，减少四元数双解抖动
#         # sign = -1.0 if q_action[3] < 0 else 1.0
#         # q_action = q_action * sign
#         # action[3:7] = q_action

#         action_base = np.concatenate([action[:6], gripper], axis=-1)   # shape = (8,)
#         return json_response(action_base)
    

        
#         #----------------------------quat -> rpy----------------------------#
#         R_state = R.from_euler('xyz', eef_state[3:6]).as_matrix()
#         q_action = action[3:7] 
#         norm = np.linalg.norm(q_action, axis=-1, keepdims=True)
#         q_action = q_action / np.clip(norm, 1e-12, None)
#         sign = np.where(q_action[..., 3:4] < 0, -1.0, 1.0)
#         q_action = q_action * sign
#         # print(q_action)
#         R_delta = R.from_quat(q_action).as_matrix()
#         rpy_cmd = R.from_matrix(R_delta).as_euler('xyz', degrees=False)
#         #-------------------------------------------------------------------#
#         action[:3] = action[:3] + eef_state[:3]
#         # action[0] = action[0]+0.005
#         # action[1] = action[1]+0.005
#         # action[2] = action[2]+0.005
#         action[3:6] = rpy_cmd
#         action_base = np.concatenate([action[:3], rpy_cmd, gripper],axis=-1)

#         return json_response(action_base)