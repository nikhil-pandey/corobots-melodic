#!/usr/bin/env python

import sys
import time
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import yaml
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from openpose_ros_msgs.msg import OpenPoseHumanList
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
from pyquaternion import Quaternion
import yaml
import laser_geometry.laser_geometry as lg
import message_filters
import joblib

# rospy.get_param('~name')
ODOM_TOPIC = '/odom'
CAMERA_TOPIC = '/camera/image_raw'
LIDAR_TOPIC = '/scan'
HUMAN_LIST = '/openpose_ros/human_list'
LASER_CALIBRATION_FILE = '/home/np7803/create_ws/src/camera_2d_lidar_calibration/data/calibration_result.txt'
CAMERA_INTRINSIC_CALIBRATION_FILE = '/home/np7803/create_ws/src/corobot_follower/data/camera_intrinsic_calibration.txt'
TIME_DIFFERENCE_IMAGE_LASER = 1
LASER_POINT_RADIUS = 3
HUMAN_LIST_WITH_DISTANCE_PUBLISHER = '/openpose_ros/human_list_distance'


def read_transformation_matrix(file):
    with open(file, 'r') as f:
        data = f.read().split()
        qx = float(data[0])
        qy = float(data[1])
        qz = float(data[2])
        qw = float(data[3])
        tx = float(data[4])
        ty = float(data[5])
        tz = float(data[6])
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


class Follower(object):
    def __init__(self):
        self.node_name = 'Follower_Node'
        rospy.init_node(self.node_name)

        self.cv_window_name = self.node_name
        cv2.namedWindow(self.cv_window_name)
        cv2.moveWindow(self.cv_window_name, 25, 75)

        rospy.loginfo('Setting things up')
        self.bridge = CvBridge()
        rospy.loginfo('Setting laser projection')
        self.lp = lg.LaserProjection()
        rospy.loginfo('Reading transformation matrix')
        self.transformation_matrix = read_transformation_matrix(LASER_CALIBRATION_FILE)
        self.translation_vector = self.transformation_matrix[:3, 3]
        self.rotation_matrix = self.transformation_matrix[:3, :3]
        self.rotation_vector, _ = cv2.Rodrigues(self.rotation_matrix)
        rospy.loginfo('Reading intrinsic calibration')
        self.lens, self.K, self.D = read_instrinsic_calibration(CAMERA_INTRINSIC_CALIBRATION_FILE)

        self.odom_subscriber = rospy.Subscriber(ODOM_TOPIC, Odometry, self.odometry_callback)
        # self.image_subscriber = rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback)
        self.openpose_subscriber = rospy.Subscriber(HUMAN_LIST, OpenPoseHumanList, self.openpose_callback)

        scan_sub = message_filters.Subscriber(LIDAR_TOPIC, LaserScan, queue_size=1)
        image_sub = message_filters.Subscriber(CAMERA_TOPIC, Image, queue_size=1)
        ts = message_filters.ApproximateTimeSynchronizer([scan_sub, image_sub], 10, TIME_DIFFERENCE_IMAGE_LASER)
        ts.registerCallback(self.image_laser_callback)

        self.human_list_publisher = rospy.Publisher(HUMAN_LIST_WITH_DISTANCE_PUBLISHER, OpenPoseHumanList, queue_size=1)

        rospy.on_shutdown(self.cleanup)
        self.frame = None
        self.laser_points = None
        self.distances = None
        self.visualize = False
        rospy.loginfo('Done initializing')

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

    def odometry_callback(self, msg):
        pass

    def image_laser_callback(self, lidar_msg, camera_msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(camera_msg)
        except CvBridgeError as e:
            print('Error converting the frame', e)

        cloud = self.lp.projectLaser(lidar_msg)
        points = pc2.read_points(cloud)
        obj_points = np.array(map(lambda point: [point[0], point[1], point[2]], points))
        Z = get_z(self.transformation_matrix, obj_points, self.K)
        obj_points = obj_points[Z > 0]
        distances = (obj_points[:, 0] ** 2 + obj_points[:, 1] ** 2) ** 0.5
        img_points, _ = cv2.projectPoints(obj_points, self.rotation_vector, self.translation_vector, self.K, self.D)
        img_points = np.squeeze(img_points)
        self.laser_points = img_points[:, 0]
        self.distances = distances
        if self.visualize:
            for i in range(len(img_points)):
                try:
                    xy = (int(round(img_points[i][0])), int(round(img_points[i][1])))
                    cv2.circle(frame, xy, LASER_POINT_RADIUS, (0, 255, 255), 1)
                    cv2.putText(frame, str(np.round(distances[i], 2)), xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                except OverflowError:
                    continue
        self.frame = np.array(frame, dtype=np.uint8)
        rospy.loginfo('Done laser camera callback')

    def openpose_callback(self, msg):
        number_of_people = msg.num_humans
        frame = self.frame

        if frame is None:
            return
        if number_of_people > 0:
            for person in msg.human_list:
                kp_count = person.num_body_key_points_with_non_zero_prob
                bbox = person.body_bounding_box
                if self.frame is not None:
                    if self.visualize:
                        cv2.rectangle(frame, (int(bbox.x), int(bbox.y + bbox.height)),
                                      (int(bbox.x + bbox.width), int(bbox.y)),
                                      (0, 255, 0), 2)
                    if self.laser_points is not None:
                        distances = self.distances[
                            np.where((self.laser_points > bbox.x) & (self.laser_points < (bbox.x + bbox.width)))]
                        distance = np.median(distances)
                        person.distance = distance
                        if self.visualize:
                            cv2.putText(frame, str(np.round(distance, 2)), (int(bbox.x), int(bbox.y)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 0, 255))
        if frame is not None:
            cv2.imshow(self.node_name, frame)

        self.human_list_publisher.publish(msg)

        keystroke = cv2.waitKey(5)
        if 32 <= keystroke < 128:
            cc = chr(keystroke).lower()
            if cc == 'q':
                rospy.loginfo('You quit')
                rospy.signal_shutdown('You wanted to quit')

    def cleanup(self):
        rospy.loginfo('Cleaning up')
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        Follower().run()
    except Exception as e:
        raise e
