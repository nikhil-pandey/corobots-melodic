#!/usr/bin/env python

import rospy
import yaml
import numpy as np
from corobot_follower.srv import LocationSrv, LocationSrvResponse
from corobot_follower.msg import HumanLocationList, HumanLocation
from pyquaternion import Quaternion
import cv2
from sensor_msgs import point_cloud2
import laser_geometry.laser_geometry as lg


def read_transformation_matrix(file):
    with open(file, 'r') as f:
        data = f.read().split()
        qx, qy, qz, qw, tx, ty, tz = tuple(map(float, data))
    q = Quaternion(qw, qx, qy, qz).transformation_matrix
    q[0, 3] = tx
    q[1, 3] = ty
    q[2, 3] = tz
    return q


def read_instrinsic_calibration(file):
    with open(file, 'r') as f:
        f.readline()
        config = yaml.load(f)
        lens = config['lens']
        fx = float(config['fx'])
        fy = float(config['fy'])
        cx = float(config['cx'])
        cy = float(config['cy'])
        k1 = float(config['k1'])
        k2 = float(config['k2'])
        p1 = float(config['p1/k3'])
        p2 = float(config['p2/k4'])
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]])
        D = np.array([k1, k2, p1, p2])
        return lens, K, D


def get_z(T_cam_world, T_world_pc, K):
    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]
    proj_mat = np.dot(K, np.hstack((R, t[:, np.newaxis])))
    xyz_hom = np.hstack((T_world_pc, np.ones((T_world_pc.shape[0], 1))))
    xy_hom = np.dot(proj_mat, xyz_hom.T).T
    z = xy_hom[:, -1]
    z = np.asarray(z).squeeze()
    return z


class LocationService:
    def __init__(self):
        rospy.init_node('location_service')
        self.lp = lg.LaserProjection()
        self.transformation_matrix = read_transformation_matrix(rospy.get_param('~laser_calibration'))
        self.translation_vector = self.transformation_matrix[:3, 3]
        self.rotation_matrix = self.transformation_matrix[:3, :3]
        self.rotation_vector, _ = cv2.Rodrigues(self.rotation_matrix)
        self.lens, self.K, self.D = read_instrinsic_calibration(rospy.get_param('~camera_calibration'))
        rospy.Service(rospy.get_param('~service_name', 'location'), LocationSrv, self.project_location)

    def project_location(self, message):
        cloud = self.lp.projectLaser(message.scan)
        points = point_cloud2.read_points(cloud)
        obj_points = np.array(map(lambda point: [point[0], point[1], point[2]], points))
        in_view = get_z(self.transformation_matrix, obj_points, self.K)
        obj_points = obj_points[in_view > 0]
        distances = (obj_points[:, 0] ** 2 + obj_points[:, 1] ** 2) ** 0.5
        img_points, _ = cv2.projectPoints(obj_points, self.rotation_vector, self.translation_vector, self.K, self.D)
        img_points = np.squeeze(img_points)
        laser_points = np.squeeze(img_points[:, 0])
        humans = message.human_list.human_list
        response = HumanLocationList()
        locations = []
        for human in humans:
            bbox = human.bounding_box
            mask = np.where((laser_points > bbox.x) & (laser_points < (bbox.x + bbox.width)))
            distance = np.median(distances[mask])
            location = np.median(obj_points[mask, :], axis=1)[0]
            locations.append(HumanLocation(location[0], location[1], distance))
        response.location_list = locations
        return response

    def run(self):
        rospy.spin()

    def cleanup(self):
        rospy.loginfo('Cleaning up')


if __name__ == '__main__':
    rospy.loginfo('Starting location service...')
    LocationService().run()
