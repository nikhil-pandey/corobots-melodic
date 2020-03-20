#!/usr/bin/env python
import math

import rospy
from corobot_follower.srv import GesturesSrv, GesturesSrvResponse
from corobot_follower.msg import OpenPoseGestureList, OpenPoseGesture
from segment import Segment

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


def is_standing(person):
    try:
        kps = person.key_points
        return angle_between(kps[C_LEFT_FOOT], kps[C_LEFT_KNEE], kps[C_LEFT_HIP]) >= 2.5 or angle_between(
            kps[C_RIGHT_FOOT], kps[C_RIGHT_KNEE], kps[C_RIGHT_HIP]) >= 2.5
    except:
        return False


def is_facing_towards_robot(person):
    try:
        kps = person.key_points
        return kps[C_LEFT_SHOULDER].x < kps[C_RIGHT_SHOULDER].x
    except:
        return False


def is_waiving(person):
    try:
        if not has_all_keypoints(person,
                                 [C_LEFT_SHOULDER, C_RIGHT_SHOULDER, C_LEFT_FOOT, C_LEFT_KNEE, C_LEFT_HIP, C_RIGHT_FOOT,
                                  C_RIGHT_KNEE, C_RIGHT_HIP, C_LEFT_WRIST, C_LEFT_ARM, C_RIGHT_WRIST, C_RIGHT_ARM]):
            rospy.loginfo('Dont have all the required key points!')
            return False

        kps = person.key_points

        # if not is_facing_towards_robot(person):
        #     rospy.loginfo('Dont turn your back on me!')
        #     return False
        #
        # if not is_standing(person):
        #     rospy.loginfo('Not standing straight!')
        #     return False

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
        if person.key_points[keypoint].prob == 0:
            return False
    return True


def intersect((p1, p2), (p3, p4)):
    return Segment((p1.x, p1.y), (p2.x, p2.y)).intersection(Segment((p3.x, p3.y), (p4.x, p4.y))) is not None


def is_crossing_hands(person):
    try:
        if not has_all_keypoints(person, [C_LEFT_WRIST, C_RIGHT_WRIST, C_LEFT_ARM, C_RIGHT_ARM, C_LEFT_SHOULDER,
                                          C_RIGHT_SHOULDER]):
            rospy.loginfo('Dont have all the required key points!')
            return False

        kps = person.key_points
        left_wrist = kps[C_LEFT_WRIST]
        right_wrist = kps[C_RIGHT_WRIST]
        left_arm = kps[C_LEFT_ARM]
        right_arm = kps[C_RIGHT_ARM]

        # if not is_facing_towards_robot(person):
        #     rospy.loginfo('Dont turn your back on me!')
        #     return False

        if not intersect((left_wrist, left_arm), (right_wrist, right_arm)):
            rospy.loginfo('Does not intersect')
            return False

        return True
    except:
        return False


class GestureService:
    def __init__(self):
        rospy.init_node('openpose_gesture_service')
        rospy.Service(rospy.get_param('service_name', 'gesture'), GesturesSrv, self.find_gestures)

    def find_gestures(self, message):
        humans = message.human_list.human_list
        response = OpenPoseGestureList()
        gestures = []
        for human in humans:
            gesture = OpenPoseGesture()
            gesture.standing = is_standing(human)
            gesture.facing_robot = is_facing_towards_robot(human)
            gesture.waiving = is_waiving(human)
            gesture.crossed_hand = is_crossing_hands(human)
            gestures.append(gesture)
        response.gesture_list = gestures
        return response

    def run(self):
        rospy.spin()

    def cleanup(self):
        rospy.loginfo('Cleaning up')


if __name__ == '__main__':
    rospy.loginfo('Starting gesture service...')
    GestureService().run()
