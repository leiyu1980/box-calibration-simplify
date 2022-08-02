import os
import cv2
import json
import numpy as np
from cv2 import aruco
from Calibration.charuco_json_formatter_edge import formatter
from Calibration.calibrater import Calibrater
from Calibration.calibrater_utility import CalibraterUtility


# Define Calibration Aruco Board Information
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
num_rows = 8
num_cols = 6
m_chess = .142875
m_aruco = .10795

board = aruco.CharucoBoard_create(num_rows, num_cols, m_chess, m_aruco, aruco_dict)
board_dict = {	'board': board, 
                'num_rows': num_rows, 
                'num_cols': num_cols, 
                'm_chess': m_chess, 
                'm_aruco': m_aruco,
                'aruco_dict': aruco_dict}


#Read in Images for each video for intrinsic and extrinsic calibration, in "int" and "ext" folders
extrinsic_img_dict, num_cams, video_properties = CalibraterUtility.import_extrinsics_data('ext')
cam_img_dict = CalibraterUtility.import_intrinsics_data('int',num_cams)

reference_img = cv2.imread(os.path.join('./', 'pattern.png'))

#Define dictionary to store calibration data to. Extra commented out functions for test (legacy)
cam_dict = {}
for i in range(num_cams):
    # corner_path = os.path.join(corner_base_dir, 'pose_'+str(i+1)+'.json')
    # cal = Calibrater(cam_img_dict[i], extrinsic_img_dict[i][0], board_dict, i, corner_path)
    cal = Calibrater(cam_img_dict[i], extrinsic_img_dict[i][0], board_dict, i, None, reference_img)
    cal.calibrate_intrinsics()
    cal.calibrate_extrinsics()
    #cal.extrinsic_error()
    #cal.frontal_img_view(extrinsic_img_dict[i][0])
    cam_dict[i] = cal
    

# Compute final calibration information
cam_model_dict = CalibraterUtility.setup_cameras(cam_dict)

# Define JSON formatted output for saving .json file of ext/int for each camera
json_list = formatter(num_cams)
for i in range(num_cams):
    var = 'Camera' + str(i+1)
    json_list[var]['CameraMatrix'] = np.array(cam_model_dict[i].extrinsic).flatten().tolist()
    json_list[var]['CameraIntrinsic'] = np.array(cam_model_dict[i].intrinsic).flatten().tolist()
    json_list[var]['CameraNumber'] = i+1
    json_list[var]['DistCoeff'] = np.array(cam_model_dict[i].distortion).flatten().tolist()
    json_list[var]['FrameHeight'] = video_properties[1]
    json_list[var]['FrameWidth'] = video_properties[0]
    json_list[var]['fps'] = video_properties[2]

# Save JSON file with calibration parameters
with open("camera_model.json", "w") as outfile: 
    json.dump(json_list, outfile)


