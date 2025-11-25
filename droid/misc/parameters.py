# import os
# from cv2 import aruco

# # Robot Params #
# nuc_ip = ""
# robot_ip = "192.168.1.1"
# laptop_ip = ""
# sudo_password = ""
# robot_type = ""  # 'panda' or 'fr3'
# robot_serial_number = ""

# # Camera ID's #
# hand_camera_id = ""
# varied_camera_1_id = ""
# varied_camera_2_id = ""

# # Charuco Board Params #
# CHARUCOBOARD_ROWCOUNT = 9
# CHARUCOBOARD_COLCOUNT = 14
# CHARUCOBOARD_CHECKER_SIZE = 0.020
# CHARUCOBOARD_MARKER_SIZE = 0.016
# ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)

# # Ubuntu Pro Token (RT PATCH) #
# ubuntu_pro_token = ""

# # Code Version [DONT CHANGE] #
# droid_version = "1.3"
import os
from cv2 import aruco

# Robot Params #
nuc_ip =None #"0.0.0.0" # ""
robot_ip = "192.168.1.1"
laptop_ip = "127.0.1.1"
sudo_password = "F990123y"
robot_type = "fr3"  # 'panda' or 'fr3'
robot_serial_number = ""

# Camera ID's #
hand_camera_id = "24285872"
varied_camera_1_id = "11022812"
varied_camera_2_id = None  # "29931811"

# Charuco Board Params #
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 14
CHARUCOBOARD_CHECKER_SIZE = 0.020
CHARUCOBOARD_MARKER_SIZE = 0.016
# ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_100) ###

# Ubuntu Pro Token (RT PATCH) #
ubuntu_pro_token = ""

# Code Version [DONT CHANGE] #
droid_version = "1.3"


