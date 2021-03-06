#!/usr/bin/env python
import math
import time

import message_filters
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, CompressedImage
from corobot_openpose.srv import EstimatePoseSrv
from corobot_openpose.msg import OpenPoseHumanList, OpenPoseHuman, BoundingBox, PointWithProb
from corobot_follower.srv import GesturesSrv, LocationSrv
from corobot_follower.msg import HumanLocation
from sklearn.neighbors import KNeighborsClassifier


class Tracker(object):

    def __init__(self):
        self.node_name = 'Follower_Node'
        rospy.init_node(self.node_name)

        self.cv_window_name = self.node_name
        cv2.namedWindow(self.cv_window_name)
        cv2.moveWindow(self.cv_window_name, 25, 75)

        rospy.loginfo('Setting things up')
        self.bridge = CvBridge()

        self.following = False
        self.lost_person = False
        self.lost_person_time = 0
        self.following_locations = None
        self.following_gestures = None
        self.last_person_feature_vector = None

        self.lost_person_threshold = int(rospy.get_param('~time_wait_person_lost'))

        self.gotoPublisher = rospy.Publisher(rospy.get_param('~goto_command_topic'), HumanLocation, queue_size=1)

        rospy.loginfo('Waiting for services...')
        rospy.wait_for_service(rospy.get_param('~openpose_service'))
        rospy.loginfo('Openpose service at: %s' % (rospy.get_param('~openpose_service')))
        rospy.wait_for_service(rospy.get_param('~gesture_service'))
        rospy.loginfo('Gesture service at: %s' % (rospy.get_param('~gesture_service')))
        rospy.wait_for_service(rospy.get_param('~location_service'))
        rospy.loginfo('Location service at: %s' % (rospy.get_param('~location_service')))
        self.openposeService = rospy.ServiceProxy(rospy.get_param('~openpose_service'), EstimatePoseSrv)
        self.gestureService = rospy.ServiceProxy(rospy.get_param('~gesture_service'), GesturesSrv)
        self.locationService = rospy.ServiceProxy(rospy.get_param('~location_service'), LocationSrv)

        rospy.loginfo('Subscribing to camera and lidar...')
        lidar_sub = message_filters.Subscriber(rospy.get_param('~laser_topic'), LaserScan)
        rospy.loginfo('Lidar topic: %s' % (rospy.get_param('~laser_topic')))
        image_sub = message_filters.Subscriber(rospy.get_param('~image_topic'), CompressedImage)
        rospy.loginfo('Image topic: %s' % (rospy.get_param('~image_topic')))
        odom_sub = message_filters.Subscriber(rospy.get_param('~odometry_topic'), Odometry)
        rospy.loginfo('Odometry Topic: %s' % (rospy.get_param('~odometry_topic')))
        message_filters.ApproximateTimeSynchronizer([image_sub, lidar_sub, odom_sub], 10,
                                                    int(rospy.get_param('~lidar_image_sync_diff'))) \
            .registerCallback(self.image_lidar_callback)

        rospy.on_shutdown(self.cleanup)
        rospy.loginfo('Done initializing')

    def image_lidar_callback(self, image_msg, laser_msg, odom_msg):
        try:
            humans = self.openposeService(image_msg).human_list
            gestures = self.gestureService(humans).gestures
            locations = self.locationService(humans, laser_msg).location_list
            # img = self.bridge.compressed_imgmsg_to_cv2(image_msg)
            # self.visualize(img, humans, gestures, locations)
            self.track(humans, gestures, self.to_global_locations(locations, self.decode_odometry(odom_msg)))
        except rospy.ServiceException as e:
            print('Service down!', e)

    def run(self):
        rospy.spin()

    def cleanup(self):
        rospy.loginfo('Cleaning up')
        cv2.destroyAllWindows()

    def visualize(self, img, humans, gestures, locations):
        img = np.array(img, dtype=np.uint8)

        for idx, person in enumerate(humans.human_list):
            bbox = person.bounding_box
            cv2.rectangle(img, (int(bbox.x), int(bbox.y + bbox.height)),
                          (int(bbox.x + bbox.width), int(bbox.y)),
                          (0, 255, 0), 2)
            rospy.loginfo(bbox)
            gesture = gestures.gesture_list[idx]
            location = locations.location_list[idx]
            text = []
            if gesture.standing:
                text.append('Standing')

            if gesture.facing_robot:
                text.append('Facing robot')

            if gesture.waiving:
                text.append('Waiving')
            if gesture.crossed_hand:
                text.append('Crossed Hand')

            cv2.putText(img, ', '.join(text), (int(bbox.x), int(bbox.y + bbox.height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255),
                        thickness=2)

            lcText = '%s, %s at %s' % (np.round(location.x, 2), np.round(location.y, 2), np.round(location.distance, 2))

            cv2.putText(img, lcText, (int(bbox.x), int(bbox.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        cv2.imshow(self.cv_window_name, img)
        cv2.waitKey(1)

    def track(self, humans, gestures, locations):
        humans_list = humans.human_list
        X, y = self.make_feature_vector(humans, gestures, locations)
        if len(humans_list) == 0:
            if not self.following:
                return
            # Wait for some time before stoping to track
            if self.lost_person:
                rospy.loginfo('Looks like we lost the person')
                if time.time() - self.lost_person_time > self.lost_person_threshold:
                    rospy.loginfo('Stopping following the person')
                    self.reset_following()
                    return
            rospy.loginfo('we lost the person, but going to wait for some time')
            self.lost_person = True
            self.lost_person_time = time.time()
            return

        if not self.following:
            waves = []
            for idx, gesture in enumerate(gestures.gesture_list):
                if gesture.waiving and gesture.standing and gesture.facing_robot:
                    rospy.loginfo("user %s is waiving" % (idx))
                    waves.append(idx)
            if len(waves) > 1:
                rospy.loginfo("Multiple people waiving! Ignoring!")
                self.reset_following()
                return
            elif len(waves) == 0:
                rospy.loginfo("Noone is waiving")
                self.reset_following()
                return
            person_to_follow = waves[0]
            self.following = True
        else:
            if humans.num_humans > 1:
                rospy.loginfo('More than 1 person, stopping to follow for now')
                knn = KNeighborsClassifier(n_neighbors=1, p=1)
                knn.fit(X, y)
                p = knn.predict(self.last_person_feature_vector.reshape(1, -1))
                person_to_follow = p[0]
            else:
                person_to_follow = 0

        feature_vector = X[np.where(y == person_to_follow)]
        following_gesture = gestures.gesture_list[person_to_follow]
        following_location = locations.location_list[person_to_follow]

        if following_gesture.facing_robot and following_gesture.crossed_hand:
            rospy.loginfo("Crossed hand detected, stopping following")
            self.reset_following()
            return

        rospy.loginfo("Going to location")
        self.last_person_feature_vector = feature_vector
        self.gotoPublisher.publish(following_location)

    def reset_following(self):
        self.following = False
        self.lost_person = False
        self.following_gestures = None
        self.following_locations = None
        self.last_person_feature_vector = None
        self.gotoPublisher.publish(HumanLocation(None, None, None))

    @staticmethod
    def decode_odometry(msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        z = orientation.z
        w = orientation.w
        theta = 2 * math.atan2(z, w)
        return x, y, theta

    @staticmethod
    def to_global_locations(locations, odometry):
        x, y, theta = odometry
        tf_matrix = np.array([
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta), np.cos(theta), y],
            [0, 0, 1]

        ])
        for location in locations.location_list:
            result = np.dot(tf_matrix, np.array([location.x, location.y, 1]))
            location.x = result[0]
            location.y = result[1]
        return locations

    def make_feature_vector(self, humans, gestures, locations):
        X = []
        y = []

        for idx in range(len(humans.human_list)):
            f = [gestures.gesture_list[idx].facing_robot, gestures.gesture_list[idx].standing,
                 locations.location_list[idx].x, locations.location_list[idx].y,
                 locations.location_list[idx].distance]
            if np.isnan(f[-1]):
                continue
            X.append(f)
            y.append(idx)
        return np.array(X), np.array(y)


if __name__ == '__main__':
    try:
        Tracker().run()
    except Exception as e:
        raise e
