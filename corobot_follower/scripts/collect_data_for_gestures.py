#!/usr/bin/env python
import os
import sys
import time
from os.path import join

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
CAMERA_TOPIC = '/camera/image_raw'
DATASET_SAVE_DIR = '/home/np7803/create_ws/src/corobot_follower/data/dataset/'
HUMAN_LIST = '/openpose_ros/human_list'


class GestureDataCollectorNode(object):
    def __init__(self):
        self.node_name = 'GestureDataCollectorNode'
        rospy.init_node(self.node_name)

        self.cv_window_name = self.node_name
        cv2.namedWindow(self.cv_window_name)
        cv2.moveWindow(self.cv_window_name, 25, 75)

        rospy.loginfo('Setting things up')
        self.bridge = CvBridge()

        openpose_sub = message_filters.Subscriber(HUMAN_LIST, OpenPoseHumanList, queue_size=1)
        image_sub = message_filters.Subscriber(CAMERA_TOPIC, Image, queue_size=1)
        ts = message_filters.ApproximateTimeSynchronizer([openpose_sub, image_sub], 10, 1)
        ts.registerCallback(self.image_openpose_callback)

        rospy.on_shutdown(self.cleanup)
        rospy.loginfo('Done initializing')
        if not os.path.exists(DATASET_SAVE_DIR):
            os.makedirs(DATASET_SAVE_DIR)
        self.feature_file = open(join(DATASET_SAVE_DIR, 'features.csv'), 'a')

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

    def odometry_callback(self, msg):
        pass

    def image_openpose_callback(self, openpose_msg, camera_msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(camera_msg)
        except CvBridgeError as e:
            print('Error converting the frame', e)

        number_of_people = openpose_msg.num_humans
        vp = np.array(frame, dtype=np.uint8)
        if number_of_people > 0:
            file_name = str(time.time()) + '.' + str(np.random.randint(1, 100)) + '.png'
            cv2.imwrite(join(DATASET_SAVE_DIR, file_name), frame)
            for person in openpose_msg.human_list:
                kp_count = person.num_body_key_points_with_non_zero_prob
                bbox = person.body_bounding_box
                cv2.rectangle(vp, (int(bbox.x), int(bbox.y + bbox.height)),
                              (int(bbox.x + bbox.width), int(bbox.y)),
                              (0, 255, 0), 2)
                features = [file_name, kp_count, bbox.x, bbox.y, bbox.height, bbox.width]
                for kp in person.body_key_points_with_prob:
                    cv2.circle(vp, (int(kp.x), int(kp.y)), 3, (0, 255, 255), 1)
                    features.append(kp.x)
                    features.append(kp.y)
                self.feature_file.write(','.join(map(str, features)))
                self.feature_file.write('\n')

        cv2.imshow(self.node_name, vp)

        keystroke = cv2.waitKey(5)
        if 32 <= keystroke < 128:
            cc = chr(keystroke).lower()
            if cc == 'q':
                rospy.loginfo('You quit')
                rospy.signal_shutdown('You wanted to quit')

    def cleanup(self):
        rospy.loginfo('Cleaning up')
        cv2.destroyAllWindows()
        self.feature_file.close()


if __name__ == '__main__':
    try:
        GestureDataCollectorNode().run()
    except Exception as e:
        raise e
