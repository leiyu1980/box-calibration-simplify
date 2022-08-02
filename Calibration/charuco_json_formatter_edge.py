import numpy as np 
# define four camera format for JSON export of camera calibration parameters with dummy values
# TODO make this formatter modular for 'n' cameras


def formatter(num_cams):
	my_dict = {}
	for i in range(num_cams):
		var = 'Camera' + str(i+1)
		props = {"CameraIntrinsic": [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8],"CameraMatrix": [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.10,1.11,1.12,1.13,1.14,1.15,1.16],"CameraNumber": 1,"DistCoeff": [1.1,1.2,1.3,1.4,1.5],"FrameHeight": 720,
				"FrameWidth": 1280,"fps": 60}
		my_dict[var] = props
		
	
	return my_dict
