#!/usr/bin/env python

import sys
import time
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from openpose_ros_msgs.msg import OpenPoseHumanList
from nav_msgs.msg import Odometry

ODOM_TOPIC = '/odom'
CAMERA_TOPIC = '/camera/image_raw'
HUMAN_LIST = '/openpose_ros/human_list'


class Follower(object):
    def __init__(self):
        self.node_name = 'Follower_Node'
        rospy.init_node(self.node_name)

        self.cv_window_name = self.node_name
        cv2.namedWindow(self.cv_window_name)
        cv2.moveWindow(self.cv_window_name, 25, 75)

        self.bridge = CvBridge()

        self.odom_subscriber = rospy.Subscriber(ODOM_TOPIC, Odometry, self.odometry_callback)
        self.image_subscriber = rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback)
        self.openpose_subscriber = rospy.Subscriber(HUMAN_LIST, OpenPoseHumanList, self.openpose_callback)
        rospy.on_shutdown(self.cleanup)
        self.frame = None

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

    def odometry_callback(self, msg):
        pass

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            print('Error converting the frame', e)

        self.frame = np.array(frame, dtype=np.uint8)

    def openpose_callback(self, msg):
        # with open('/home/np7803/create_ws/src/corobot_follower/scripts/test.txt', 'w') as f:
        #     f.write(str(msg))
        number_of_people = msg.num_humans
        frame = self.frame
        for person in msg.human_list:
            kp_count = person.num_body_key_points_with_non_zero_prob
            bbox = person.body_bounding_box
            if self.frame is not None:
                cv2.rectangle(frame, (int(bbox.x), int(bbox.y)), (int(bbox.x + bbox.width), int(bbox.height)), (0, 255, 0), 2)

        cv2.imshow(self.node_name, frame)
        self.frame = None

        keystroke = cv2.waitKey(5)
        if 32 <= keystroke < 128:
            cc = chr(keystroke).lower()
            if cc == 'q':
                rospy.signal_shutdown('You wanted to quit')

    def cleanup(self):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        Follower().run()
    except:
        pass
