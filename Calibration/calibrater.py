import numpy as np
import cv2
import os
from cv2 import aruco
from Calibration.calibrater_utility import CalibraterUtility

#sys.path.insert(1, './camera_interaction_toolkit')
from Calibration.camera_model import CameraModel



class Calibrater:

	def __init__(self, intrin_imgs, extrin_img, board_dict, cam_id, corner_path, reference_img):
		
		# lst of images
		self.intrin_imgs = intrin_imgs
		# lst of images
		self.extrin_img = extrin_img
		# {board, num_rows, num_cols, m_chess, m_aruco, aruco_dict}
		self.board_dict = board_dict
		# camera_id
		self.cam_id = cam_id
		self.corner_path = corner_path

		self.cam_mat = None
		self.no_dist = np.array([0.,0.,0.,0.,0.])
		self.dist = None
		self.cm = None
		self.ext_mat = None
		self.rvec = None
		self.tvec = None
		self.ref_img = reference_img
		self.imsize = None

	def calibrate_intrinsics(self, debug=True):
		# 标定相机内参，计算多帧图像标定内参的误差，选取符合误差阈值的内参
		all_corners = []
		all_ids = []
		decimator = 0
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.00001)

		for image in self.intrin_imgs:

			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			#print(gray.shape)
			
			corners, ids, rejected_imgPoints = cv2.aruco.detectMarkers(gray, self.board_dict['aruco_dict'])
			
			corners, ids = cv2.aruco.refineDetectedMarkers(gray, self.board_dict['board'], corners, ids, rejected_imgPoints)[:2]
			
			if len(corners) > 0:
				for corner in corners:
					cv2.cornerSubPix(gray, corner,
									 winSize=(3, 3),
									 zeroZone=(-1, -1),
									 criteria=criteria)
				res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board_dict['board'])
				
				if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 1 == 0:
					# print('good')
					#print(res2[1].shape)
					#fsdfsd
					all_corners.append(res2[1])
					all_ids.append(res2[2])
				
			decimator += 1
		aruco.drawDetectedMarkers(gray,corners)
		
		# plt.figure()
		# plt.imshow(gray)
		# plt.show()
		imsize = gray.shape
		self.imsize = imsize
		flags = (cv2.CALIB_USE_INTRINSIC_GUESS) #+ cv2.CALIB_RATIONAL_MODEL)
		
		#print(all_corners)

		
		# 棋盘格的3D坐标点，规定为z轴为0，xy坐标与棋盘格的实际大小相同的坐标。
		# 移动棋盘格的位置，拍摄了多帧图像，计算这些图像棋盘格特征点和3D特征点之间的内外参矩阵，同一个相机的内参矩阵一致
		(ret, camera_matrix, distortion_coefficients,
		 rotation_vectors, translation_vectors,
		 std_dev_intrinsics, std_dev_extrinsics,
		 per_view_erro) = cv2.aruco.calibrateCameraCharucoExtended(charucoCorners=all_corners,
																   charucoIds=all_ids,
																   board=self.board_dict['board'],
																   imageSize=imsize,
																   cameraMatrix=np.array([[1000, 0, imsize[1]/2],[0, 1000, imsize[0]/2],[0,0,1]]),
																   distCoeffs=self.no_dist,
																   flags=flags,
																   criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT
																			 , 10000, 1e-9))

		print('DIST', distortion_coefficients)
		print('Finished Calibration of Camera:', self.cam_id)
		print('-------------------------------')
		if debug:
			print('Average Pixel Error Per Image:', np.array(per_view_erro).flatten().mean())

		
		
		# new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients.flatten(), imsize, 1, imsize)
		#img = self.intrin_imgs[0]#cv2.cvtColor(self.intrin_imgs[0], cv2.COLOR_BGR2GRAY)
		#CalibraterUtility.plot_intrinsics(new_cam_mat, distortion_coefficients.flatten(), self.board_dict['aruco_dict'], img)
		self.cam_mat = camera_matrix 
		print(camera_matrix)
		self.dist = distortion_coefficients.flatten()
		#print(self.dist)

		num_bad_imgs = 0
		for i, image in enumerate(self.intrin_imgs):

			#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			mismatch = CalibraterUtility.extrinsic_error(	self.cam_mat, 
											self.dist, 
											rotation_vectors[i], 
											translation_vectors[i], 
											image, 
											self.board_dict, 
											self.ref_img, 
											False)
			if mismatch > 2.5:

				num_bad_imgs +=1
		print('Number of bad images:', num_bad_imgs, 'out of', len(self.intrin_imgs))
		if num_bad_imgs > int(.1 * len(self.intrin_imgs)):

			print('Please redo intrinsic calibraiton.')
		print()
		print()


	def calibrate_extrinsics(self, debug=False):
		#print(self.dist)
		# undis_img = CalibraterUtility.undistort_img(self.cam_mat, self.dist, self.extrin_img)
		gray = cv2.cvtColor(self.extrin_img, cv2.COLOR_BGR2GRAY)
		# gray = cv2.cvtColor(undis_img, cv2.COLOR_BGR2GRAY)
		if len(gray.shape)==2:
			print('DEBUG: EXTRINSIC IMAGE READ')
		
		num_charuco_u, charuco_corners, charuco_ids = CalibraterUtility.detect_corner_char(
								gray, 
								self.board_dict['aruco_dict'], 
								self.board_dict['board']
								)


		print('DEBUG: NUMBER OF CHARUCO POINTS DETECTED:',charuco_corners.shape[0])
		ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
								charuco_corners, 
								charuco_ids, 
								self.board_dict['board'], 
								self.cam_mat, 
								self.dist, None, None, 
								useExtrinsicGuess=False)

		
		rot_mat, _ = cv2.Rodrigues(rvec.copy())
		
		temp = np.zeros((4,4))
		temp[0:3,0:3] = rot_mat
		temp[:3,3] = tvec.flatten()
		temp[3,3] = 1

		print(temp)
		self.ext_mat = temp
		self.rvec = rvec
		self.tvec = tvec

		self.cm = CameraModel(self.cam_mat, self.ext_mat, self.no_dist)
		# this guy is used for triangulating the points to 3D
		# if i pass in UNdistorted image points, then no dist is good!

		# take any imagr from camera I
		# undistort that image using dist coeff
		# apply cam mat 


	def frontal_img_view(self, img):
		# 功能：可视化相机拍摄的图像和其变换到归一化世界坐标系的图像，根据归一化世界坐标系图像和参考图像的相似程度，来肉眼大致判断内外参估计是否可靠

		undis_img = CalibraterUtility.undistort_img(self.cam_mat, self.dist, img)
		gray = cv2.cvtColor(undis_img, cv2.COLOR_BGR2GRAY)
		num_charuco_u, charuco_corners, charuco_ids = CalibraterUtility.detect_corner_char(
								gray, 
								self.board_dict['aruco_dict'], 
								self.board_dict['board'])
		ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
								charuco_corners, 
								charuco_ids, 
								self.board_dict['board'], 
								self.cam_mat, 
								self.no_dist, None, None, 
								useExtrinsicGuess=False)

		CalibraterUtility.frontal_view(	self.cam_mat, 
										self.no_dist, 
										rvec, tvec, 
										img, 
										self.board_dict, True)

	def extrinsic_error(self):
	#计算相机内外参估计的误差
		CalibraterUtility.extrinsic_error(	self.cam_mat, 
											self.dist, 
											self.rvec, 
											self.tvec, 
											self.extrin_img, 
											self.board_dict)

	def get_ground_plane_corners(self):
		# 获得世界坐标系下棋盘格的3D点，经过内外参矩阵变换后，图像坐标系下的坐标
		img_pts = CalibraterUtility.get_camera_grnd_plane_pts(	self.cam_mat, 
																self.no_dist, 
																self.rvec, 
																self.tvec, 
																self.board_dict)
		return img_pts



def main():

	num_cams = 4

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
	# cam_img_dict = CalibraterUtility.import_intrinsics_data(
	# 					'/home/otg/Desktop/muhammad/memes/intrinsics', 4)
	# extrinsic_img_dict = CalibraterUtility.import_extrinsics_data(
	# 					'/home/otg/Desktop/muhammad/memes/extrinsics')

	extrinsic_img_dict, num_cams, video_properties = CalibraterUtility.import_extrinsics_data('C:/mycode/camera_calib/box-calibration-master/ext')
	cam_img_dict = CalibraterUtility.import_intrinsics_data('C:/mycode/camera_calib/box-calibration-master/int', num_cams)


	cam_dict = {}
	corner_base_dir = './biocore_livetest/'

	reference_img = cv2.imread(os.path.join('C:/mycode/camera_calib/box-calibration-master/', 'pattern.png'))
	grnd_corners = []
	for i in range(num_cams):
		
		corner_path = os.path.join(corner_base_dir, 'pose_'+str(i+1)+'.json')
		# cal = Calibrater(cam_img_dict[i], extrinsic_img_dict[i][0], board_dict, i, corner_path)
		cal = Calibrater(cam_img_dict[i], extrinsic_img_dict[i][0], board_dict, i, None, reference_img)
		cal.calibrate_intrinsics()
		cal.calibrate_extrinsics()
		#cal.frontal_img_view(extrinsic_img_dict[i][0])
		#print(cal.cam_mat)
		#cal.extrinsic_error()
		grnd_corners.append(cal.get_ground_plane_corners())
		cam_dict[i] = cal


	cam_model_dict = CalibraterUtility.setup_cameras(cam_dict)

	



if __name__ == '__main__':
	main()
