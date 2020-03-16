#!/usr/bin/env python
import math
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
from segment import Segment

# rospy.get_param('~name')
ODOM_TOPIC = '/odom'
CAMERA_TOPIC = '/camera/image_raw'
LIDAR_TOPIC = '/scan'
HUMAN_LIST = '/openpose_ros/human_list'
LASER_CALIBRATION_FILE = '/home/np7803/create_ws/src/camera_2d_lidar_calibration/data/calibration_result.txt'
CAMERA_INTRINSIC_CALIBRATION_FILE = '/home/np7803/create_ws/src/corobot_follower/data/camera_intrinsic_calibration.txt'
TIME_DIFFERENCE_IMAGE_LASER_OPENPOSE = 3
LASER_POINT_RADIUS = 3
HUMAN_LIST_WITH_DISTANCE_PUBLISHER = '/openpose_ros/human_list_distance'

C_MOUTH = 0
C_NECK = 1
C_LEFT_SHOULDER = 2
C_LEFT_ARM = 3
C_LEFT_WRIST = 4
C_RIGHT_SHOULDER = 5
C_RIGHT_ARM = 6
C_RIGHT_WRIST = 7
C_LEFT_HIP = 8
C_LEFT_KNEE = 9
C_LEFT_FOOT = 10
C_RIGHT_HIP = 11
C_RIGHT_KNEE = 12
C_RIGHT_FOOT = 13
C_LEFT_EYE = 14
C_RIGHT_EYE = 15
C_LEFT_EAR = 16
C_RIGHT_EAR = 17


def angle_between(p0, p1, p2):
    x1 = p1.x - p0.x
    y1 = p1.y - p0.y
    x2 = p1.x - p2.x
    y2 = p1.y - p2.y
    x3 = p2.x - p0.x
    y3 = p2.y - p0.y
    b = x1 * x1 + y1 * y1
    a = x2 * x2 + y2 * y2
    c = x3 * x3 + y3 * y3
    return math.acos((a + b - c) / math.sqrt(4 * a * b))


def has_all_keypoints(person, keypoints):
    for keypoint in keypoints:
        if person.body_key_points_with_prob[keypoint].prob == 0:
            return False
    return True


def is_standing(person):
    try:
        kps = person.body_key_points_with_prob
        return angle_between(kps[C_LEFT_FOOT], kps[C_LEFT_KNEE], kps[C_LEFT_HIP]) >= 2.5 or angle_between(
            kps[C_RIGHT_FOOT], kps[C_RIGHT_KNEE], kps[C_RIGHT_HIP]) >= 2.5
    except:
        return False


def is_facing_towards_robot(person):
    try:
        kps = person.body_key_points_with_prob
        return kps[C_LEFT_SHOULDER].x < kps[C_RIGHT_SHOULDER].x
    except:
        return False


def has_raised_hand(person):
    try:
        if not has_all_keypoints(person,
                                 [C_LEFT_SHOULDER, C_RIGHT_SHOULDER, C_LEFT_FOOT, C_LEFT_KNEE, C_LEFT_HIP, C_RIGHT_FOOT,
                                  C_RIGHT_KNEE, C_RIGHT_HIP, C_LEFT_WRIST, C_LEFT_ARM, C_RIGHT_WRIST, C_RIGHT_ARM]):
            rospy.loginfo('Dont have all the required key points!')
            return False

        kps = person.body_key_points_with_prob

        if not is_facing_towards_robot(person):
            rospy.loginfo('Dont turn your back on me!')
            return False

        if not is_standing(person):
            rospy.loginfo('Not standing straight!')
            return False

        if kps[C_LEFT_WRIST].y < kps[C_LEFT_ARM].y and \
                kps[C_LEFT_WRIST].x < kps[C_LEFT_SHOULDER].x \
                and kps[C_LEFT_WRIST].x < kps[C_LEFT_ARM].x:
            if angle_between(kps[C_LEFT_WRIST], kps[C_LEFT_ARM],
                             kps[C_LEFT_SHOULDER]) < 1:
                return True
            rospy.loginfo('Left hand angle not so small! %s' % (angle_between(kps[C_LEFT_WRIST], kps[C_LEFT_ARM],
                                                                              kps[C_LEFT_SHOULDER])))
        rospy.loginfo('Left wrist not in right place')

        if kps[C_RIGHT_WRIST].y < kps[C_RIGHT_ARM].y and \
                kps[C_RIGHT_WRIST].x > kps[C_RIGHT_SHOULDER].x \
                and kps[C_RIGHT_WRIST].x > kps[C_RIGHT_ARM].x:
            if angle_between(kps[C_RIGHT_WRIST], kps[C_RIGHT_ARM],
                             kps[C_RIGHT_SHOULDER]) < 1:
                return True
            rospy.loginfo('right hand angle not so small %s' % (angle_between(kps[C_RIGHT_WRIST], kps[C_RIGHT_ARM],
                                                                              kps[C_RIGHT_SHOULDER])))
        rospy.loginfo('right hand not in right place %s %s %s' % (
            kps[C_RIGHT_WRIST].y < kps[C_RIGHT_ARM].y, kps[C_RIGHT_WRIST].x > kps[C_RIGHT_SHOULDER].x,
            kps[C_RIGHT_WRIST].x > kps[C_RIGHT_ARM].x))
        return False
    except:
        return False


def intersect((p1, p2), (p3, p4)):
    return Segment((p1.x, p1.y), (p2.x, p2.y)).intersection(Segment((p3.x, p3.y), (p4.x, p4.y))) is not None


def has_crossed_hand(person):
    try:
        if not has_all_keypoints(person, [C_LEFT_WRIST, C_RIGHT_WRIST, C_LEFT_ARM, C_RIGHT_ARM, C_LEFT_SHOULDER,
                                          C_RIGHT_SHOULDER]):
            rospy.loginfo('Dont have all the required key points!')
            return False

        kps = person.body_key_points_with_prob
        left_wrist = kps[C_LEFT_WRIST]
        right_wrist = kps[C_RIGHT_WRIST]
        left_arm = kps[C_LEFT_ARM]
        right_arm = kps[C_RIGHT_ARM]

        if not is_facing_towards_robot(person):
            rospy.loginfo('Dont turn your back on me!')
            return False

        if not intersect((left_wrist, left_arm), (right_wrist, right_arm)):
            rospy.loginfo('Does not intersect')
            return False

        return True
    except:
        return False


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

        self.frame = None
        self.laser_points = None
        self.distances = None
        self.visualize = True

        # self.image_subscriber = rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback)
        # self.openpose_subscriber = rospy.Subscriber(HUMAN_LIST, OpenPoseHumanList, self.openpose_callback)

        scan_sub = message_filters.Subscriber(LIDAR_TOPIC, LaserScan, queue_size=20)
        image_sub = message_filters.Subscriber(CAMERA_TOPIC, Image, queue_size=20)
        openpose_sub = message_filters.Subscriber(CAMERA_TOPIC, Image, queue_size=20)
        ts = message_filters.ApproximateTimeSynchronizer([scan_sub, image_sub, openpose_sub], 10,
                                                         TIME_DIFFERENCE_IMAGE_LASER_OPENPOSE)
        ts.registerCallback(self.image_laser_openpose_callback)

        self.human_list_publisher = rospy.Publisher(HUMAN_LIST_WITH_DISTANCE_PUBLISHER, OpenPoseHumanList, queue_size=1)

        rospy.on_shutdown(self.cleanup)
        rospy.loginfo('Done initializing')

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

    def image_callback(self, camera_msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(camera_msg)
        except CvBridgeError as e:
            print('Error converting the frame', e)
        self.frame = np.array(frame, dtype=np.uint8)
        rospy.loginfo('Done laser camera callback')

    def image_laser_openpose_callback(self, lidar_msg, camera_msg, openpose_msg):
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
        laser_points = img_points[:, 0]

    def openpose_callback(self, msg):
        number_of_people = msg.num_humans

        if self.frame is None:
            return
        frame = np.array(self.frame, dtype=np.uint8)
        if number_of_people > 0:
            for person in msg.human_list:
                kp_count = person.num_body_key_points_with_non_zero_prob
                bbox = person.body_bounding_box
                person.waiving_hand = has_raised_hand(person)
                person.crossing_hand = has_crossed_hand(person)
                mask = np.where((laser_points > bbox.x) & (laser_points < (bbox.x + bbox.width)))
                distances = distances[mask]
                locations = obj_points[mask]
                distance = np.median(distances)
                location = np.median(locations, axis=0)
                person.distance = distance
                person.x = location[0]
                person.y = location[1]

        self.human_list_publisher.publish(openpose_msg)
        rospy.loginfo('Done laser camera openpose callback')

    def cleanup(self):
        rospy.loginfo('Cleaning up')
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        Follower().run()
    except Exception as e:
        raise e
