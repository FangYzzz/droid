from copy import deepcopy

import gym
import numpy as np
from scipy.spatial.transform import Rotation
import argparse
import os
# from franky import Affine
# import franky
import numpy as np
# from franky import *
import time

# from droid.calibration.calibration_utils import load_calibration_info
from droid.camera_utils.info import camera_type_dict
from droid.camera_utils.wrappers.multi_camera_wrapper import MultiCameraWrapper
from droid.misc.parameters import hand_camera_id, nuc_ip
from droid.misc.server_interface import ServerInterface
from droid.misc.time import time_ms
from droid.misc.transformations import change_pose_frame
import requests


class FrankaRobotClient:
    def __init__(self, ip="127.0.0.1", port=8006):
        self.base_url = f"http://{ip}:{port}"

    # def update_joints(self, joints, velocity=False, blocking=True, cartesian_noise=None):
    #     payload = {
    #         "joints": np.asarray(joints).tolist(),
    #         "velocity": velocity,
    #         "blocking": blocking,
    #         "cartesian_noise": None if cartesian_noise is None else np.asarray(cartesian_noise).tolist(),
    #     }
    #     r = requests.post(f"{self.base_url}/update_joints", json=payload)
    #     r.raise_for_status()
    #     return r.json()["result"]

    def update_joints(self, joints, velocity=False, blocking=False, cartesian_noise=None):
        r = requests.post(
            f"{self.base_url}/update_joints",
            json={
                "command": list(joints),
                "velocity": velocity,
                "blocking": blocking,
                "cartesian_noise": cartesian_noise if cartesian_noise is not None else None,
            },
        )
        r.raise_for_status()
        return r.json()

    # def update_gripper(self, value, velocity=False, blocking=True):
    #     payload = {
    #         "value": float(value),
    #         "velocity": velocity,
    #         "blocking": blocking,
    #     }
    #     r = requests.post(f"{self.base_url}/update_gripper", json=payload)
    #     r.raise_for_status()
    #     return r.json()["result"]

    def update_gripper(self, gripper_action, velocity=False, blocking=False):
        r = requests.post(
            f"{self.base_url}/update_gripper",
            json={
                "command": gripper_action,
                "velocity": velocity,
                "blocking": blocking,
            },
        )
        r.raise_for_status()
        return r.json()

    def update_command(
        self,
        action,
        action_space="cartesian_velocity",
        gripper_action_space=None,
        blocking=False,
    ):
        payload = {
            "command": np.asarray(action).tolist(),
            "action_space": action_space,
            "gripper_action_space": gripper_action_space,
            "blocking": blocking,
        }
        r = requests.post(f"{self.base_url}/update_command", json=payload)
        r.raise_for_status()
        return r.json()["result"]

    def get_robot_state(self):
        r = requests.get(f"{self.base_url}/get_robot_state")
        r.raise_for_status()
        data = r.json()
        return data["state_dict"], data["timestamp_dict"]

    def create_action_dict(self, action):
        payload = {
            "action": np.asarray(action).tolist(),
            "action_space": "cartesian_velocity",
            "gripper_action_space": None,
            "blocking": False,
        }
        r = requests.post(f"{self.base_url}/create_action_dict", json=payload)
        r.raise_for_status()
        return r.json()["result"]


class RobotEnv(gym.Env):
    def __init__(self, action_space="cartesian_velocity", gripper_action_space=None, camera_kwargs={}, do_reset=True):
        # Initialize Gym Environment
        super().__init__()

        # Define Action Space #
        assert action_space in ["cartesian_position", "joint_position", "cartesian_velocity", "joint_velocity"]
        self.action_space = action_space
        self.gripper_action_space = gripper_action_space
        self.check_action_range = "velocity" in action_space

        # Robot Configuration
        # self.reset_joints = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])
        # self.reset_joints = np.array([-0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 1.9])
        # [-0.6, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 1.9]
        # self.reset_joints = np.array([-0.2, -1 / 5 * np.pi, 0, -3.7 / 5 * np.pi, 0, 2.7 / 5 * np.pi, 2.2])  #####
        self.reset_joints = np.array([-0.3, -1 / 5 * np.pi, 0, -3.7 / 5 * np.pi, 0, 2.7 / 5 * np.pi, 2.2])

        # self.reset_joints = np.array([-0.27517385, -0.619897 ,   0.2653431  ,-2.2437236  ,-0.07290433 , 1.62032083, 2.40461694])  #####
        
        self.randomize_low = np.array([-0.1, -0.2, -0.1, -0.3, -0.3, -0.3])
        self.randomize_high = np.array([0.1, 0.2, 0.1, 0.3, 0.3, 0.3])
        self.DoF = 7 if ("cartesian" in action_space) else 8
        self.control_hz = 15 # 15

        if nuc_ip is None:
            self._robot = FrankaRobotClient()


        # Create Cameras
        self.camera_reader = MultiCameraWrapper(camera_kwargs)
        # self.calibration_dict = load_calibration_info()
        self.camera_type_dict = camera_type_dict

        # Reset Robot
        if do_reset:
            self.reset()

    def step(self, action):
        # Check Action
        assert len(action) == self.DoF
        # if self.check_action_range:
        #     assert (action.max() <= 1) and (action.min() >= -1)

        # Update Robot
        action_info = self.update_robot(
            action,
            action_space=self.action_space,
            gripper_action_space=self.gripper_action_space,
        )

        # Return Action Info
        return action_info

    def reset(self, randomize=False):
        self._robot.update_gripper(0, velocity=False, blocking=True)

        if randomize:
            noise = np.random.uniform(low=self.randomize_low, high=self.randomize_high)
        else:
            noise = None

        self._robot.update_joints(self.reset_joints, velocity=False, blocking=True, cartesian_noise=noise)

    def update_robot(self, action, action_space="cartesian_velocity", gripper_action_space=None, blocking=False):
        action_info = self._robot.update_command(
            action,
            action_space=action_space,
            gripper_action_space=gripper_action_space,
            blocking=blocking
        )
        return action_info

    def create_action_dict(self, action):
        return self._robot.create_action_dict(action)

    def read_cameras(self):
        return self.camera_reader.read_cameras()

    def get_state(self):
        read_start = time_ms()
        state_dict, timestamp_dict = self._robot.get_robot_state()
        timestamp_dict["read_start"] = read_start
        timestamp_dict["read_end"] = time_ms()
        return state_dict, timestamp_dict

    # def get_camera_extrinsics(self, state_dict):
    #     # Adjust gripper camere by current pose
    #     extrinsics = deepcopy(self.calibration_dict)
    #     for cam_id in self.calibration_dict:
    #         if hand_camera_id not in cam_id:
    #             continue
    #         gripper_pose = state_dict["cartesian_position"]
    #         extrinsics[cam_id + "_gripper_offset"] = extrinsics[cam_id]
    #         extrinsics[cam_id] = change_pose_frame(extrinsics[cam_id], gripper_pose)
    #     return extrinsics

    def get_observation(self):
        obs_dict = {"timestamp": {}}

        # Robot State #
        state_dict, timestamp_dict = self.get_state()
        obs_dict["robot_state"] = state_dict
        obs_dict["timestamp"]["robot_state"] = timestamp_dict

        # Camera Readings #
        camera_obs, camera_timestamp = self.read_cameras()
        obs_dict.update(camera_obs)
        obs_dict["timestamp"]["cameras"] = camera_timestamp

        # Camera Info #
        obs_dict["camera_type"] = deepcopy(self.camera_type_dict)
        # extrinsics = self.get_camera_extrinsics(state_dict)
        # obs_dict["camera_extrinsics"] = extrinsics

        intrinsics = {}
        # for cam in self.camera_reader.camera_dict.values():
        #     cam_intr_info = cam.get_intrinsics()
        #     for (full_cam_id, info) in cam_intr_info.items():
        #         intrinsics[full_cam_id] = info["cameraMatrix"]
        # obs_dict["camera_intrinsics"] = intrinsics

        return obs_dict