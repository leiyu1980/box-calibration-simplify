import cv2
from cv2 import aruco
import numpy as np 
from glob import glob
from os.path import join
import matplotlib.pyplot as plt
import sys
from Calibration.camera_model import CameraModel

import json

acceptable_suffixes = {'MOV', 'mov', 'avi', 'mp4'}

class CalibraterUtility:

	@staticmethod
	def undistort_points(cam_mat, dist, pts, imsize):
		new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(cam_mat, dist, (imsize[1], imsize[0]), 1, (imsize[1], imsize[0]))
		temp = cv2.undistortPoints(pts, cam_mat, dist, None, new_cam_mat)

		return temp

	@staticmethod 
	def undistort_img(cam_mat, dist, img):
		h, w = img.shape[0:2]
		#print(w,h)

		new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(cam_mat, dist, (w,h),1,(w,h))
		image = cv2.undistort(img.copy(), cam_mat, dist, None, new_cam_mat)

		return image

	@staticmethod
	def detect_corner_char(gray, aruco_dict, board):
		parameters = aruco.DetectorParameters_create()
		
		corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
		corners, ids = cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)[:2]
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
		aruco.drawDetectedMarkers(gray,corners)

		# plt.figure()
		# plt.imshow(gray)
		# plt.show()
		
		for corner in corners:
			cv2.cornerSubPix(gray, corner, winSize = (3,3), zeroZone = (-1,-1), criteria = criteria)
		num_charuco, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)

		return num_charuco, charuco_corners, charuco_ids

	@staticmethod
	def detect_corner_char_click(gray, aruco_dict, board, pnt_path, cam_mat, dist):

		with open(pnt_path) as f:
			data = json.load(f)

		img_pts = np.array([data['tl'],data['tr'], data['br'], data['bl']])
		#img_pts = img_pts.reshape((4,1,2))
		#print(img_pts)
		#print(dist)
		#new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(cam_mat, dist, (720, 1280), 1, (720, 1280))
		#n_img_pts = cv2.undistortPoints(img_pts, cam_mat, dist, None, new_cam_mat)
		#n_img_pts = n_img_pts.reshape((4,2))
		#img_pts = n_img_pts
		#img_pts = np.array([[847, 551],[649, 613], [489, 571], [692, 523]])

		plt.imshow(gray)
		plt.scatter(img_pts[:,0], img_pts[:,1])
		#plt.show()
		number = 500
		pts_dst = np.array([[number, 0],[number, number],[0,number],[0, 0]])
		h, status = cv2.findHomography(img_pts, pts_dst)
		size = ( number, number)
		im_dst = cv2.warpPerspective(gray, h, size)

		# plt.imshow(im_dst)
		#cv2.imwrite('pose_4.png', gray)
		#plt.show()

		parameters = aruco.DetectorParameters_create()
		parameters.adaptiveThreshWinSizeStep = 1
		#parameters.polygonalApproxAccuracyRate=0.9
		corners, ids, rejectedImgPoints = aruco.detectMarkers(im_dst, aruco_dict, parameters = parameters)
		corners, ids = cv2.aruco.refineDetectedMarkers(im_dst, board, corners, ids, rejectedImgPoints)[:2]
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
		for corner in corners:
			cv2.cornerSubPix(im_dst, corner, winSize = (3,3), zeroZone = (-1,-1), criteria = criteria)
		num_charuco, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, im_dst, board)
		new_crners = np.zeros(charuco_corners.shape, dtype=np.float32)
		#print(charuco_corners)
		
		h_inv = np.linalg.inv(h)
		# plt.imshow(gray)
		for i in range(num_charuco):
			c = charuco_corners[i,:,:].flatten()
			tmp = h_inv.dot([c[0], c[1],1])
			tmp[0:2] = tmp[0:2] / tmp[2]	
			new_crners[i,:,0] = tmp[0]
			new_crners[i,:,1] = tmp[1]	
			plt.scatter(tmp[0], tmp[1])
		# plt.show()

		new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(cam_mat, dist, (720, 1280), 1, (720, 1280))
		new_crners = cv2.undistortPoints(new_crners, cam_mat, dist, None, new_cam_mat)
		return num_charuco, new_crners, charuco_ids

	@staticmethod
	def apply_hom_to_pts(h, pts):
		pts = np.array(pts)
		tmp = np.ones((pts.shape[0], pts.shape[1]+1))
		tmp[:,0:2] = pts
		res = h.dot(tmp.T)
		res[0,:] = res[0,:] / res[2,:]
		res[1,:] = res[1,:] / res[2,:]

		return res[0:2,:].T

	@staticmethod
	def import_extrinsics_data(extrin_pth):

		print()
		print('##### Initiating Extrinsic Calibration ####')
		cam_img_dict = {}

		fnames = glob(join(extrin_pth,'*.*'))
		#print(fnames)
		num_cams = 0
		
		for nm in fnames:
			
			cam_num = list(nm.split('/')[-1].split('.')[0])[-1]
			cam_img_dict[int(cam_num)-1] = []

			if nm.split('/')[-1].split('.')[-1] in acceptable_suffixes:

				#print(nm)
				cap = cv2.VideoCapture(nm)
				#cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
				ret, frm = cap.read()
				width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
				height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
				fps = cap.get(cv2.CAP_PROP_FPS)
				#print(ret)
				if ret:
					cam_img_dict[int(cam_num)-1].append(frm)
			num_cams +=1
		video_properties = [width,height,fps]
		for i in range(num_cams):
			print('Number of images for camera ' + str(i+1) + ':', len(cam_img_dict[i]))
		print()

		return cam_img_dict, num_cams, video_properties

	@staticmethod
	def import_intrinsics_data(intrin_pth, num_cams):

		print()
		print('##### Initiating Intrinsic Calibration ####')
		cam_img_dict = {}
		for i in range(num_cams):
			
			cam_img_dict[i] = []
			fnames = glob(join(join(intrin_pth,str(i+1)),'*'))

			for nm in fnames:
				if nm.split('/')[-1].split('.')[-1] in acceptable_suffixes:

					#print(nm)
					cap = cv2.VideoCapture(nm)
					#cap.set(cv2.CAP_PROP_POS_FRAMES, 61)
					ret, frm = cap.read()
					#print(ret)
					if ret:
						cam_img_dict[i].append(frm)

		print()
		for i in range(num_cams):
			print('Number of images for camera ' + str(i+1) + ':', len(cam_img_dict[i]))
		print()

		return cam_img_dict

	@staticmethod
	def plot_intrinsics(cam_mat, dist, aruco_dict, image):
		fig, axs = plt.subplots(1,2)
		axs[0].imshow(image)
		axs[0].set_title('Original')
		
		#axs[1].imshow(image)
		axs[1].set_title('Undistort')

		image = CalibraterUtility.undistort_img(cam_mat, dist, image)

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		parameters =  aruco.DetectorParameters_create()
		
			
		corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict,
														  parameters=parameters)
		# SUB PIXEL DETECTION
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
		for corner in corners:
			cv2.cornerSubPix(gray, corner, winSize = (3,3), zeroZone = (-1,-1), criteria = criteria)

		frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

		size_of_marker =  0.0285 # side lenght of the marker in meter
		dist = np.array([0.,0.,0.,0.,0.])
		rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, size_of_marker , cam_mat, dist)

		length_of_axis = 0.1
		imaxis = aruco.drawDetectedMarkers(frame_markers.copy(), corners, ids)

		for i in range(len(tvecs)):
			imaxis = aruco.drawAxis(imaxis, cam_mat, dist, rvecs[i], tvecs[i], length_of_axis)
		axs[1].imshow(imaxis, interpolation = "nearest")

		plt.show()



	@staticmethod
	def frontal_view(cam_mat, dist, rvec, tvec, img, board_dict, debug):
		# 功能：评价第i张棋盘格图像标定的相机内参是否准确
		# 输入棋盘格图像，根据相机内参对图像进行畸变矫正，在已知棋盘格世界坐标系情况下，pnp计算相机外参，
		# 将棋盘格世界坐标系根据求解的内外参投影到图像坐标系，计算他的投影图像坐标和归一化后的世界坐标系H变换矩阵
		# 将畸变矫正后的图像经过H矩阵变换到归一化世界坐标系，输出（畸变矫正图像，归一化世界坐标系下的图像，他们俩的H变换矩阵）
		# 矫正后的图像变换到归一化世界坐标系后，其与我们放置在世界坐标系下的棋盘格归一化坐标进行比较，坐标误差越小，相机内参标定越准确

		if debug:
			fig, axs = plt.subplots(1,2)
		undis_img = CalibraterUtility.undistort_img(cam_mat, dist, img)

		num_rows = board_dict['num_rows']
		num_cols = board_dict['num_cols']
		board = board_dict['board']
		aruco_dict = board_dict['aruco_dict']
		m_chess = board_dict['m_chess']
		dist = np.array([0.,0.,0.,0.,0.])
		axis = np.float32([[0,0,0], [0,num_cols,0], [num_rows,num_cols,0], [num_rows,0,0]]) * m_chess
		gray = cv2.cvtColor(undis_img, cv2.COLOR_BGR2GRAY)
		
		num_charuco, charuco_corners, charuco_ids = CalibraterUtility.detect_corner_char(gray, aruco_dict, board)

		ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, cam_mat, dist, None, None, useExtrinsicGuess=False)
		#计算去畸变后的图像和世界坐标系下的棋盘格坐标变换矩阵，即该相机外参
		imgpts, jac = cv2.projectPoints(axis, rvec, tvec, cam_mat, dist)
		#获得图像坐标系下的理想关键点

		img_pts = imgpts.reshape(4,2)
		#plt.imshow(undis_img)
		#plt.scatter(img_pts[:,0], img_pts[:,1])
		#plt.show()
		
		axis = np.float32([[0,0,0], [num_cols,0,0], [num_cols,num_rows,0], [0,num_rows,0]]) * 100
		h, status = cv2.findHomography(imgpts, axis)
		im_dst = cv2.warpPerspective(undis_img, h, (num_cols * 100,num_rows * 100))
		#把去畸变的图像变换到归一化后的世界坐标系下
		if debug:
			axs[0].imshow(undis_img)
			axs[1].imshow(im_dst)
			plt.show()

		return undis_img, im_dst, h

	@staticmethod
	def intersection(l1, l2):
		return(list(set(l1)&set(l2)))

	@staticmethod
	def extrinsic_error(cam_mat, dist, rvec, tvec, img, board_dict, ref_img, display):
		# 作用：给定相机内外参，计算相机内外参求解的error
		# 去畸变后的图像变换到归一化世界坐标系（这个变换过程使用了求解的相机内外参），并将世界坐标系下的棋盘格也变换到归一化世界坐标系，
		# 如果相机内参和外参都求解准确，则他们变换后的坐标误差小于阈值。

		if display:
			fig =  plt.figure()
		undis_im, im_frontal, h = CalibraterUtility.frontal_view(cam_mat, dist, rvec, tvec, img, board_dict, False)
		gray = cv2.cvtColor(undis_im, cv2.COLOR_BGR2GRAY)
		if display:
			plt.imshow(gray)
			plt.show()
		board = board_dict['board']
		aruco_dict = board_dict['aruco_dict']
		m_chess = board_dict['m_chess']
		num_charuco_u, charuco_corners_u, charuco_ids_u = CalibraterUtility.detect_corner_char(gray, aruco_dict, board)
		#print(charuco_ids)
		gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
		ref_num_charuco_u, ref_charuco_corners_u, ref_charuco_ids_u = CalibraterUtility.detect_corner_char(gray_ref, aruco_dict, board)
		ref_charuco_corners_u = np.array(ref_charuco_corners_u).reshape((len(ref_charuco_ids_u),2))
		#print(ref_charuco_ids_u)
		idx = [0, 28, 34, 6]
		#plt.imshow(gray_ref)
		#plt.scatter(ref_charuco_corners_u[idx,0], ref_charuco_corners_u[idx,1])
		#plt.show()

		num_rows = board_dict['num_rows']
		num_cols = board_dict['num_cols']
		imgpts = ref_charuco_corners_u[idx,:]

		axis = np.float32([[1,1,0], [num_cols-1,1,0], [num_cols-1,num_rows-1,0], [1,num_rows-1,0]]) * 100
		h_n, status = cv2.findHomography(imgpts, axis)
		#计算放置在世界坐标系下的棋盘格坐标特征点到归一化后世界坐标系棋盘格的坐标变换H矩阵

		ref_im_dst = cv2.warpPerspective(gray_ref, h_n, (num_cols * 100,num_rows * 100))
		#plt.imshow(ref_im_dst)
		#plt.title('Ground Truth')
		#plt.show()
		HI = 255
		LO = 120
		th, ref_im_gray_th_otsu = cv2.threshold(ref_im_dst, LO, HI, cv2.THRESH_OTSU)

		gray = cv2.cvtColor(im_frontal, cv2.COLOR_BGR2GRAY)

		num_charuco, charuco_corners, charuco_ids = CalibraterUtility.detect_corner_char(gray, aruco_dict, board)
		#print(charuco_ids)

		a = CalibraterUtility.intersection((charuco_ids_u.flatten()).tolist(),(charuco_ids.flatten()).tolist() )
		first_set = np.where(charuco_ids_u == a)[0]
		sec_set = np.where(charuco_ids == a)[0]

		new_u = charuco_corners_u.reshape(num_charuco_u,2)
		#tmp_u = np.ones((num_charuco_u,3))
		#tmp_u[:,0:2] = new_u
		res = CalibraterUtility.apply_hom_to_pts(h, new_u)
		res = res.T
		new = charuco_corners.reshape(num_charuco,2)

		res = res * m_chess
		new = new * m_chess
		errs = np.sqrt(((new[sec_set,0] - res[0,first_set])**2 + (new[sec_set,1] - res[1,first_set])**2))
		#print('Average Error (cm):', np.mean(errs))
		#print('Std Dev Error (cm):', np.std(errs))
		
		res = res / m_chess
		new = new / m_chess
		th, im_gray_th_otsu = cv2.threshold(gray, LO, HI, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		#th3 = cv2.adaptiveThreshold(gray,190,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
		th3 = np.int32((gray>th + 10)*255)
		#plt.imshow(th3)
		#plt.title('Predicted')
		#plt.show()
		mismatch = len(np.where(th3-np.int32(ref_im_gray_th_otsu) !=0)[0]) / (num_cols * 100 * num_rows * 100) * 100

		if display:
			plt.imshow(th3-ref_im_gray_th_otsu)
			plt.scatter(new[sec_set,0], new[sec_set,1])
			plt.title('Percent Janked Px:' + str(mismatch))
		
			plt.scatter(res[0,first_set], res[1,first_set])
			plt.show()
		return mismatch

	@staticmethod
	def get_camera_grnd_plane_pts(cam_mat, dist, rvec, tvec, board_dict):
		"""
		计算世界坐标系下的棋盘格在图像坐标系下的投影
		"""
		num_rows = board_dict['num_rows']
		num_cols = board_dict['num_cols']
		m_chess = board_dict['m_chess']
		axis = np.float32([[0,0,0], [0,num_cols,0], [num_rows,num_cols,0], [num_rows, 0,0]]) * m_chess

		imgpts, jac = cv2.projectPoints(axis, 
										rvec, 
										tvec,
										cam_mat,
										dist)

		img_pts = imgpts.reshape(4,2)

		return img_pts

	@staticmethod
	def setup_cameras(cam_dict):
		cam_model_dict = {}
		for k in cam_dict.keys():
			curr_cam =  cam_dict[k]
			cam_model_dict[k] = CameraModel(curr_cam.cam_mat, curr_cam.ext_mat, curr_cam.no_dist)

		return cam_model_dict

