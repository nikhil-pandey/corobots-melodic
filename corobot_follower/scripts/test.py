import joblib
import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_z(T_cam_world, T_world_pc, K):
    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]
    proj_mat = np.dot(K, np.hstack((R, t[:, np.newaxis])))
    xyz_hom = np.hstack((T_world_pc, np.ones((T_world_pc.shape[0], 1))))
    xy_hom = np.dot(proj_mat, xyz_hom.T).T
    z = xy_hom[:, -1]
    z = np.asarray(z).squeeze()
    return z

ranges, cloud, frame, obj_points, transformation_matrix, K, D, rotation_vector, translation_vector = joblib.load('test.dump')
frame_n = np.array(frame, dtype=np.uint8)
Z = get_z(transformation_matrix, obj_points, K)
obj_points_nz = obj_points[Z > 0]
img_points, _ = cv2.projectPoints(obj_points_nz, rotation_vector, translation_vector, K, D)
img_points = np.squeeze(img_points)