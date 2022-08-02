"""Module for operations on Camera."""
from typing import List
import cv2
import numpy as np
from numpy import linalg


class CameraModel:  # pylint: disable=too-many-instance-attributes
    """Representation of operations on Camera Model."""

    def __init__(self, camera_intrinsic: np.ndarray, camera_matrix: np.ndarray, camera_distortion: np.ndarray) -> None:
        """Initialize Camera Model with calibration data.

        :param camera_intrinsic: 3x3 matrix with intrinsic camera parameters
        :param camera_matrix: 4x4 matrix with rotation and translation camera parameters
        :param camera_distortion: 1x5 matrix with radial and tangential distortion parameters
        """
        self.extrinsic = camera_matrix
        self.intrinsic = camera_intrinsic
        self.distortion = camera_distortion
        self.inverted_extrinsic = linalg.pinv(camera_matrix) #相机外参矩阵的逆
        self.inverted_intrinsic = linalg.pinv(camera_intrinsic)
        self.inverted_rotation = self.inverted_extrinsic[:3, :3]
        self.rotation = self.inverted_rotation.T
        self.translation = -1 * self.rotation.dot(self.inverted_extrinsic[:3, 3].reshape((-1, 1)))
        self.projection = np.hstack([self.rotation, self.translation])
        self.projection = camera_intrinsic.dot(self.projection)
        self.position = self._compute_position()
        self.focal_length = camera_intrinsic[0, 0]

    def world_to_image(self, world_coordinates: List[float]) -> np.array:
        """Translate world coordinates into image space pixels.

        :param world_coordinates: 3D point in world space as a list of pixel positions [X, Y, Z]
        :return: 2D point as a numpy array [X, Y]
        """
        world_point = np.array([world_coordinates[0], world_coordinates[1],
                                world_coordinates[2], 1]).reshape((-1, 1))
        image_camera = np.dot(self.projection, world_point)
        image_px = image_camera / image_camera[2]  # Rescale to the image plane
        image_px = image_px[: 2].reshape((-1,))  # Remove homogeneous component
        return image_px

    def image_to_world(self, image_px: List[int], z: float = 1.0, vector_length: int = 120, im_size: tuple = (720,1280))\
            -> np.ndarray:
        """Translate image space pixels into world space.
        输入单个图像特征点，可以计算他在世界坐标系下的射线。（因为需要两对点才能确定3D点位置）给定射线长度的情况下可以获得3D点
        :param image_px: 2D point as a list of pixel positions [X, Y]
        :param z: (optional) distance to the plane on which point is projected in camera space
        :param vector_length: (optional) length of the ray vector
        :return: 3D point in world space as a numpy array
        """
        #new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(self.intrinsic, self.distortion, im_size, 1, im_size)
        #image_px = cv2.undistortPoints(np.array([[image_px]]), self.intrinsic, self.distortion, None, new_cam_mat)
        #image_px = np.squeeze(image_px).tolist()
        
        img_point = np.array([image_px[0], image_px[1], 1]).reshape((-1, 1))
        img_point = z * self.inverted_intrinsic.dot(img_point)
        img_point = np.vstack([img_point, [[1]]])

        world_point = np.dot(self.inverted_extrinsic, img_point)
        world_point = world_point[:3].reshape((-1,))  # Remove homogeneous component

        ray_vector = world_point - self.position
        ray_vector = self._normalize(ray_vector)

        return self.position + ray_vector * vector_length

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """Normalize a given vector."""
        length = np.sqrt(np.dot(vector, vector))
        if length > 0:
            vector /= length
        return vector

    def _compute_position(self) -> np.ndarray:
        """Compute position in 3D world based on camera calibration.

        :return: position as a 3-element vector [X, Y, Z]
        """
        return self.inverted_extrinsic[:3, 3].reshape((-1,))
