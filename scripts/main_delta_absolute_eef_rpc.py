# ruff: noqa


import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from scipy.spatial.transform import Rotation as R
import copy
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro
from typing import Optional
import os
import numpy as np
from franky import *
from scipy.spatial.transform import Rotation
import time

import zerorpc
import websockets
from typing import Dict, Any, Optional
import asyncio
import logging
import json
import cv2
import base64
import contextlib
import signal
import threading



faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15  #15

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

import asyncio
import functools

try:
    to_thread = asyncio.to_thread
except AttributeError:
    async def to_thread(func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))



@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "24285872" # 26658469
    right_camera_id: str = None # "<your_camera_id>"
    wrist_camera_id: str = "11022812"  # "11022812"

    # Policy parameters
    external_camera: Optional[str] = (
        # None  # which external camera should be fed to the policy, choose from ["left", "right"]
        "left"
    )

    # Rollout parameters
    max_timesteps: int = 1200  # 600
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 45  # 8 

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )

    ws_server: bool = True                   # True 时以服务端模式启动（供 A 调用）
    ws_host: str = "0.0.0.0"
    ws_port: int = 4242
    max_concurrent: int = 1

# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
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
    """
    在主线程中:临时屏蔽 Ctrl+C(SIGINT)，防止打断关键 RPC 调用。
    在子线程中:signal.signal 不能用，直接 no-op。
    """
    # 如果当前不是主线程，就直接当普通 context，用于兼容线程池场景（WS 模式）
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


def main(args: Args):
   
    # Make sure external camera is specified by user -- we only use one external camera for the policy
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    # env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])
    robot_state,_ = env.get_state()
    print(robot_state["cartesian_position"])
    eef_q = copy.deepcopy(robot_state["cartesian_position"][3:6])
    gripper = None
    while True:
        instruction = input("Enter instruction: ")

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video_wirst = []
        video_left = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        
        for t_step in bar:
            start_time = time.time()
            try:
                # Get the current observation
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )
                # print(curr_obs["cartesian_position"])

                video_wirst.append(curr_obs["wrist_image"]) ###
                video_left.append(curr_obs[f"{args.external_camera}_image"]) ###
                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    robot_state_,_ = env.get_state()
                    eef_state = copy.deepcopy(robot_state_["cartesian_position"])
                    eef_pose = curr_obs["cartesian_position"]
                    eef_rpy = eef_pose[3:6]
                    eef_quat = R.from_euler('xyz', eef_rpy, degrees=False).as_quat()
                    eef_pose = np.concatenate([eef_pose[:3], eef_quat], axis=-1)
                    #######################################
                    # if curr_obs["gripper_position"]<0.3:
                    #     curr_obs["gripper_position"]=0.0
                    # else:
                    #     curr_obs["gripper_position"]=1.0
                    # curr_obs["cartesian_position"][2] = curr_obs["cartesian_position"][2]-0.11
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs[f"{args.external_camera}_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/eef_position": eef_pose,
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                        # pred_action_chunk = np.pad(pred_action_chunk, ((0,0),(0,1)), mode='constant') ###
                    
                    # print("pred_action_chunk.shape: ", pred_action_chunk.shape)
                    assert pred_action_chunk.shape == (50, 8) # 10,8

                # Select current action to execute from chunk
                # if actions_from_chunk_completed>0:
                #     action = pred_action_chunk[actions_from_chunk_completed] - pred_action_chunk[actions_from_chunk_completed-1]
                    
                # else:
                #     # pass
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1
                if action[-1].item() > 0.5:
                # if False:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                    gripper = np.ones((1,))
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])
                    gripper = np.zeros((1,))

                #----------------------------quat -> rpy----------------------------#
                R_state = R.from_euler('xyz', eef_state[3:6]).as_matrix()
                q_action = action[3:7] 
                norm = np.linalg.norm(q_action, axis=-1, keepdims=True)
                q_action = q_action / np.clip(norm, 1e-12, None)
                sign = np.where(q_action[..., 3:4] < 0, -1.0, 1.0)
                q_action = q_action * sign
                # print(q_action)
                R_delta = R.from_quat(q_action).as_matrix()
                rpy_cmd = R.from_matrix(R_delta).as_euler('xyz', degrees=False)
                #-------------------------------------------------------------------#
                
                # action[:-2] = np.clip(action[:-2], -0.1, .1)
                # action = np.clip(action, -0.2, 0.2) ###
                # action[:-2] = action[:-2]*2
                action[:3] = action[:3] + eef_state[:3]
                action[0] = action[0]#+0.005
                action[1] = action[1]#+0.005
                action[2] = action[2] #+0.005
                action[3:6] = rpy_cmd
                action_ = np.concatenate([action[:3], rpy_cmd, gripper],axis=-1)
                
                # Prevent touching the table
                # if action_[2]<0.22:
                #     action_[2] = 0.22
                # print(action[3:6])
                env.step(action_)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break
        
        save_filename = "None"
        if input("Save videos? (enter y or n) ").lower() == "y":
            video_wirst = np.stack(video_wirst)
            save_filename = "video_" + timestamp + "_wrist"
            ImageSequenceClip(list(video_wirst), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")
            video_left = np.stack(video_left)
            save_filename = "video_" + timestamp + "_left"
            ImageSequenceClip(list(video_left), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec: "
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")

        df = pd.concat([
            df,
            pd.DataFrame([{
                "success": success,
                "duration": t_step,
                "video_filename": save_filename,
            }])
        ], ignore_index=True)


        if input("Do one more eval? (enter y or n): ").lower() != "y":
            break
        env.reset()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


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

    # Drop the alpha dimension
    left_image = left_image[..., :3]
    # right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1]
    # right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    # joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    if save_to_disk:
        # combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        combined_image = np.concatenate([left_image, wrist_image], axis=1) ###
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")


    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        # "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


def _jpg_b64_from_rgb(img_rgb): # RGB -> BGR -> JPG -> b64
    if img_rgb is None:
        return None
    img_bgr = img_rgb[..., ::-1]
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ================== 非交互单轮（给 A 调用） ==================
def run_one_rollout(
    args: Args,
    instruction: str,
    *,
    env: Optional[RobotEnv] = None,          # 可以复用外部传入的 env
    external_camera: Optional[str] = None,
    max_timesteps: Optional[int] = None,
    save_videos: bool = False,
    success_value: Optional[float] = None,   # 不传则默认 1.0
    stop_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:

    if external_camera is not None:
        args.external_camera = external_camera
    assert args.external_camera in ["left", "right"]

    if max_timesteps is not None:
        args.max_timesteps = max_timesteps

    # 如果外面没传 env，就自己建一个；如果传了就直接用同一个
    owns_env = False
    if env is None:
        env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
        owns_env = True

    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])
    robot_state,_ = env.get_state()
    gripper = None

    actions_from_chunk_completed = 0
    pred_action_chunk = None

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    video_wirst, video_left = [], []
    bar = tqdm.tqdm(range(args.max_timesteps))
    print("Running rollout (WS single pass)...",instruction)

    t_step = 0
    try:
        for t_step in bar:
            # 每一轮先检查是否要停止
            if stop_event is not None and stop_event.is_set():
                logger.info("Rollout aborted via stop_event")
                break

            start_time = time.time()
            curr_obs = _extract_observation(args, env.get_observation(), save_to_disk=(t_step == 0))
            video_wirst.append(curr_obs["wrist_image"])
            video_left.append(curr_obs[f"{args.external_camera}_image"])

            if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                robot_state_,_ = env.get_state()
                eef_state = copy.deepcopy(robot_state_["cartesian_position"])
                eef_pose = curr_obs["cartesian_position"]
                eef_rpy = eef_pose[3:6]
                eef_quat = R.from_euler('xyz', eef_rpy, degrees=False).as_quat()
                eef_pose = np.concatenate([eef_pose[:3], eef_quat], axis=-1)
                actions_from_chunk_completed = 0

                request_data = {
                    "observation/exterior_image_1_left": image_tools.resize_with_pad(
                        curr_obs[f"{args.external_camera}_image"], 224, 224
                    ),
                    "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                    "observation/eef_position": eef_pose,
                    "observation/gripper_position": curr_obs["gripper_position"],
                    "prompt": instruction,
                }
                with prevent_keyboard_interrupt():
                    pred_action_chunk = policy_client.infer(request_data)["actions"]
                assert pred_action_chunk.shape == (50, 8)

            action = pred_action_chunk[actions_from_chunk_completed]
            actions_from_chunk_completed += 1
            if action[-1].item() > 0.5:
            # if False:
                # action[-1] = 1.0
                action = np.concatenate([action[:-1], np.ones((1,))])
                gripper = np.ones((1,))
            else:
                # action[-1] = 0.0
                action = np.concatenate([action[:-1], np.zeros((1,))])
                gripper = np.zeros((1,))

            #----------------------------quat -> rpy----------------------------#
            R_state = R.from_euler('xyz', eef_state[3:6]).as_matrix()
            q_action = action[3:7] 
            norm = np.linalg.norm(q_action, axis=-1, keepdims=True)
            q_action = q_action / np.clip(norm, 1e-12, None)
            sign = np.where(q_action[..., 3:4] < 0, -1.0, 1.0)
            q_action = q_action * sign
            R_delta = R.from_quat(q_action).as_matrix()
            rpy_cmd = R.from_matrix(R_delta).as_euler('xyz', degrees=False)
            #-------------------------------------------------------------------#
            
            action[:3] = action[:3] + eef_state[:3]
            action[0] = action[0]+0.005
            action[1] = action[1]+0.005
            action[2] = action[2]+0.005
            action[3:6] = rpy_cmd
            action_ = np.concatenate([action[:3], rpy_cmd, gripper],axis=-1)
            
            # Prevent touching the table
            if action_[2]<0.22:
                action_[2] = 0.22
            # print(action[3:6])
            env.step(action_)

            # Sleep to match DROID data collection frequency
            elapsed_time = time.time() - start_time
            if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
    except KeyboardInterrupt:
        pass
    finally:
        # 每次 rollout 结束都把机器人复位
        try:
            if env is not None:
                logger.info("Resetting RobotEnv to initial pose after rollout...")
                env.reset()
        except Exception as e:
            logger.warning(f"env.reset() failed: {e}")

    save_filename = "None"
    if save_videos:
        video_wirst = np.stack(video_wirst)
        save_filename = "video_" + timestamp + "_wrist"
        ImageSequenceClip(list(video_wirst), fps=10).write_videofile(
            save_filename + ".mp4", codec="libx264"
        )
        video_left = np.stack(video_left)
        save_filename = "video_" + timestamp + f"_{args.external_camera}"
        ImageSequenceClip(list(video_left), fps=10).write_videofile(
            save_filename + ".mp4", codec="libx264"
        )

    success = 1.0 if success_value is None else max(0.0, min(1.0, float(success_value)))
    df = pd.concat(
        [df, pd.DataFrame([{"success": success, "duration": t_step, "video_filename": save_filename}])],
        ignore_index=True,
    )

    os.makedirs("results", exist_ok=True)
    ts2 = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_path = os.path.join("results", f"eval_{ts2}.csv")
    df.to_csv(csv_path)
    logger.info(f"Results saved to {csv_path}")

    return {
        "success": float(success),
        "duration": int(t_step),
        "video_filename": save_filename,
        "csv_filename": csv_path,
    }



# ================== WebSocket 服务器（供 A 调用） ==================
class RobotWSApp:
    """
    JSON-RPC over WebSocket:
    - A 发送：{"method":"run_eval","params":{"instruction":"...","external_camera":"left","max_timesteps":1500,"save_videos":false,"success_value":1.0}}
    - B 返回：{"ok":true,"result":{...}} or {"ok":false,"error":"..."}
    - run_eval: 运行一次评测
    - snapshot: 返回一帧相机图(left/wrist), jpg base64
    """
    def __init__(self, args: Args, env: RobotEnv):
        self.args = args
        self._env = env  # 已经初始化好的 env
        self._sem = asyncio.Semaphore(args.max_concurrent)  # 限流，避免并发占用机器人
        self._current_stop_event: Optional[threading.Event] = None  # 简单起见：只支持一个在跑
        

    async def handler(self, websocket):
        logger.info("connection open")
        try:
            async for message in websocket:
                try:
                    req = json.loads(message)
                    method = req.get("method")
                    params = req.get("params", {}) or {}

                    if method == "snapshot":
                        # 直接用已经连好的 env
                        def _do_snapshot():
                            obs = self._env.get_observation()
                            obs2 = _extract_observation(self.args, obs, save_to_disk=False)
                            return {
                                "left_image": _jpg_b64_from_rgb(obs2.get("left_image")),
                                "wrist_image": _jpg_b64_from_rgb(obs2.get("wrist_image")),
                            }

                        try:
                            # result = await asyncio.to_thread(_do_snapshot)  # 避免阻塞事件循环
                            result = await to_thread(_do_snapshot)
                            resp = {"ok": True, "result": result}
                        except Exception as e:
                            logger.exception("snapshot failed")
                            resp = {"ok": False, "error": f"snapshot failed: {e}"}

                    elif method == "run_eval":
                        # 限制并发：一个一个跑
                        async with self._sem:
                            stop_event = threading.Event()
                            self._current_stop_event = stop_event

                            # 在线程池里跑阻塞的机器人执行，避免阻塞事件循环
                            # result = await asyncio.to_thread(
                            result = await to_thread(                             
                                run_one_rollout,
                                self.args,
                                params.get("instruction", "do task"),
                                env=self._env,   # 复用同一个 env
                                external_camera=params.get("external_camera", None),
                                max_timesteps=params.get("max_timesteps", None),
                                save_videos=bool(params.get("save_videos", False)),
                                success_value=params.get("success_value", None),
                                stop_event=stop_event,
                            )
                        self._current_stop_event = None
                        resp = {"ok": True, "result": result}
                    
                    elif method == "cancel":
                        if self._current_stop_event is not None:
                            self._current_stop_event.set()
                            resp = {"ok": True, "result": "cancel signaled"}
                        else:
                            resp = {"ok": False, "error": "no active rollout"}

                    else:
                        resp = {"ok": False, "error": f"unknown method: {method}"}

                except Exception as e:
                    logger.exception("WS handler error")
                    resp = {"ok": False, "error": str(e)}

                try:
                    await websocket.send(json.dumps(resp))
                except Exception as e:
                    logger.warning(f"send failed: {e}")
                    break

        except websockets.exceptions.ConnectionClosedError as e:
            logger.info(f"connection closed by peer: code={e.code}, reason={e.reason}")
        except Exception as e:
            logger.exception(f"WS handler outer error: {e}")
        finally:
            logger.info("connection handler finished")

    async def serve(self, host: str, port: int):
        logger.info(f"[B] WebSocket Server listening on ws://{host}:{port}")
        async with websockets.serve(
            self.handler,
            host,
            port,
            ping_interval=60,   # 稍微放宽一点
            ping_timeout=180,
        ):
            await asyncio.Future()


def run_ws_server(args: Args):
    # 这里就创建 RobotEnv，直接连上机器人 & 打开相机
    logger.info("Initializing RobotEnv (connecting robot & cameras)...")
    env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
    robot_state, _ = env.get_state()
    logger.info(f"Robot initial cartesian_position: {robot_state['cartesian_position']}")

    app = RobotWSApp(args, env)
    try:
        asyncio.run(app.serve(args.ws_host, args.ws_port))
    except KeyboardInterrupt:
        logger.info("WS server stopped by user")


# ================== 入口 ==================
if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    if args.ws_server:
        # 以“服务端模式”启动（给 A 调），内部自动作为 C 的客户端调用 run_one_rollout()
        run_ws_server(args)
    else:
        # 保持你原来的交互 CLI 行为
        main(args)