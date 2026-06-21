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

from droid.calibration.calibration_utils import load_calibration_info
from droid.camera_utils.info import camera_type_dict
from droid.camera_utils.wrappers.multi_camera_wrapper import MultiCameraWrapper
from droid.misc.parameters import hand_camera_id, nuc_ip
from droid.misc.server_interface import ServerInterface
from droid.misc.time import time_ms
from droid.misc.transformations import change_pose_frame


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
            from droid.franka.robot import FrankaRobot
            self._robot = FrankaRobot()
        else:
            self._robot = ServerInterface(ip_address=nuc_ip)

        # Create Cameras
        self.camera_reader = MultiCameraWrapper(camera_kwargs)
        self.calibration_dict = load_calibration_info()
        self.camera_type_dict = camera_type_dict

        # Reset Robot
        if do_reset:
            self.reset()

    def step(self, action):
        # Check Action
        assert len(action) == self.DoF
        if self.check_action_range:
            assert (action.max() <= 1) and (action.min() >= -1)

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

    def get_camera_extrinsics(self, state_dict):
        # Adjust gripper camere by current pose
        extrinsics = deepcopy(self.calibration_dict)
        for cam_id in self.calibration_dict:
            if hand_camera_id not in cam_id:
                continue
            gripper_pose = state_dict["cartesian_position"]
            extrinsics[cam_id + "_gripper_offset"] = extrinsics[cam_id]
            extrinsics[cam_id] = change_pose_frame(extrinsics[cam_id], gripper_pose)
        return extrinsics

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
        extrinsics = self.get_camera_extrinsics(state_dict)
        obs_dict["camera_extrinsics"] = extrinsics

        intrinsics = {}
        # for cam in self.camera_reader.camera_dict.values():
        #     cam_intr_info = cam.get_intrinsics()
        #     for (full_cam_id, info) in cam_intr_info.items():
        #         intrinsics[full_cam_id] = info["cameraMatrix"]
        # obs_dict["camera_intrinsics"] = intrinsics

        return obs_dict
    

# class RobotEnv_Franky(gym.Env):
#     def __init__(self, action_space="cartesian_velocity", gripper_action_space=None, camera_kwargs={}, do_reset=True):
#         # Initialize Gym Environment
#         super().__init__()

#         # Define Action Space #
#         assert action_space in ["cartesian_position", "joint_position", "cartesian_velocity", "joint_velocity"]
#         self.action_space = action_space
#         self.gripper_action_space = gripper_action_space
#         self.check_action_range = "velocity" in action_space

#         # Robot Configuration
#         # self.reset_joints = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])
#         # self.reset_joints = np.array([-0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 1.9])
#         self.reset_joints = np.array([-0.6, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 1.9])
#         self.randomize_low = np.array([-0.1, -0.2, -0.1, -0.3, -0.3, -0.3])
#         self.randomize_high = np.array([0.1, 0.2, 0.1, 0.3, 0.3, 0.3])
#         self.DoF = 7 if ("cartesian" in action_space) else 8
#         self.control_hz = 15 # 15
#         self.robot = Robot("172.17.0.2") 
#         self.robot.relative_dynamics_factor = 0.06
#         # urdf_model = self.robot.model_urdf
#         # print(urdf_model)
#         # Create Cameras
#         self.camera_reader = MultiCameraWrapper(camera_kwargs)
#         self.calibration_dict = load_calibration_info()
#         self.camera_type_dict = camera_type_dict

#         # Reset Robot
#         if do_reset:
#             self.reset()

#     def step(self, action):
#         # Check Action
#         assert len(action) == self.DoF
#         if self.check_action_range:
#             assert (action.max() <= 1) and (action.min() >= -1)

#         # Update Robot
#         trans = action[:3]
#         rot = action[3:6]
#         # quat = action[3:7]
#         # quat = Rotation.from_rotvec(rot).as_quat()
#         current_eef_state_ = self.robot.current_cartesian_state.pose.end_effector_pose

#         eef_q = current_eef_state_.quaternion
#         ee_pose_new = Affine(trans, eef_q)
#         # ee_pose_new = Affine(trans)
#         motion = CartesianMotion(ee_pose_new, ReferenceType.Absolute)
#         self.robot.move(motion, asynchronous=True)
#         # Return Action Info
#         return True

#     def reset(self, randomize=False):
#         # self.robot.relative_dynamics_factor = 0.2
#         # q_start = np.array([0.0, -0.785398, 0.0, -2.35619, 0.0, 1.5708, 0.785398])
#         # q_start = np.array([-0.6, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 1.9])
#         motion = JointMotion(self.reset_joints, ReferenceType.Absolute)
#         self.robot.move(motion, asynchronous=True)

#     def update_robot(self, action, action_space="cartesian_velocity", gripper_action_space=None, blocking=False):
#         action_info = self._robot.update_command(
#             action,
#             action_space=action_space,
#             gripper_action_space=gripper_action_space,
#             blocking=blocking
#         )
#         return action_info

#     def create_action_dict(self, action):
#         return self._robot.create_action_dict(action)

#     def read_cameras(self):
#         return self.camera_reader.read_cameras()

#     def get_state(self):
#         read_start = time_ms()
#         state_dict={}
#         timestamp_dict = {}
#         current_eef_state_ = self.robot.current_cartesian_state.pose.end_effector_pose
#         # print(current_eef_state_)
#         eef_t = current_eef_state_.translation
#         eef_q = current_eef_state_.quaternion
#         r = Rotation.from_quat(eef_q)
#         rpy = r.as_euler('xyz', degrees=False)
#         current_eef_state = np.concatenate([eef_t, rpy])
#         state_dict["cartesian_position"] = current_eef_state
#         state_dict["gripper_position"] = 0
#         timestamp_dict["read_start"] = read_start
#         timestamp_dict["read_end"] = time_ms()
#         return state_dict, timestamp_dict

#     def get_camera_extrinsics(self, state_dict):
#         # Adjust gripper camere by current pose
#         extrinsics = deepcopy(self.calibration_dict)
#         for cam_id in self.calibration_dict:
#             if hand_camera_id not in cam_id:
#                 continue
#             gripper_pose = state_dict["cartesian_position"]
#             extrinsics[cam_id + "_gripper_offset"] = extrinsics[cam_id]
#             extrinsics[cam_id] = change_pose_frame(extrinsics[cam_id], gripper_pose)
#         return extrinsics

#     def get_observation(self):
#         obs_dict = {"timestamp": {}}

#         # Robot State #
#         # state_dict, timestamp_dict = self.get_state()
#         state_dict={}
#         current_eef_state_ = self.robot.current_cartesian_state.pose.end_effector_pose
#         # print(current_eef_state_)
#         eef_t = current_eef_state_.translation
#         eef_q = current_eef_state_.quaternion
#         r = Rotation.from_quat(eef_q)
#         rpy = r.as_euler('xyz', degrees=False)
#         current_eef_state = np.concatenate([eef_t, rpy])
#         state_dict["cartesian_position"] = current_eef_state
#         state_dict["gripper_position"] = 0
#         obs_dict["robot_state"] = state_dict
#         # obs_dict["timestamp"]["robot_state"] = timestamp_dict

#         # Camera Readings #
#         camera_obs, camera_timestamp = self.read_cameras()
#         obs_dict.update(camera_obs)
#         obs_dict["timestamp"]["cameras"] = camera_timestamp

#         # Camera Info #
#         obs_dict["camera_type"] = deepcopy(self.camera_type_dict)
#         extrinsics = self.get_camera_extrinsics(state_dict)
#         obs_dict["camera_extrinsics"] = extrinsics

#         intrinsics = {}
#         for cam in self.camera_reader.camera_dict.values():
#             cam_intr_info = cam.get_intrinsics()
#             for (full_cam_id, info) in cam_intr_info.items():
#                 intrinsics[full_cam_id] = info["cameraMatrix"]
#         obs_dict["camera_intrinsics"] = intrinsics

#         return obs_dict
    

