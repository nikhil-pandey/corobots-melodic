#!/usr/bin/env python

import math
import time
from collections import deque
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from corobot_follower.msg import HumanLocation

TOLERANCE = 1
T360 = math.pi * 2


class GOTO(object):
    def __init__(self):
        rospy.init_node('goto', anonymous=True)
        self.odom_subscriber = rospy.Subscriber(rospy.get_param('odometry_topic'), Odometry, self.odom_callback)
        self.destination_subscriber = rospy.Subscriber(rospy.get_param('goto_command_topic'), HumanLocation,
                                                       self.destination_callback)
        self.velocity_publisher = rospy.Publisher(rospy.get_param('velocity_topic'), Twist, queue_size=10)
        self.destinations = deque()
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def destination_callback(self, msg):
        if msg.x != 0 and msg.y != 0:
            self.destinations = deque([(msg.x, msg.y)])
            rospy.loginfo('Received new destination: %s' % (self.destinations))
            return
        self.destinations = deque([])

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        z = orientation.z
        w = orientation.w
        self.theta = 2 * math.atan2(z, w)

    def run(self):
        rate = rospy.Rate(60)
        last_time = time.time()

        while not rospy.is_shutdown():
            if not self.destinations:
                self.stop()
                rate.sleep()
                continue

            x, y = self.destinations[0]
            if self.at_destination(x, y):
                rospy.loginfo('Reached %s, %s. Actual Position: %s, %s' % (x, y, self.x, self.y))
                self.destinations.popleft()
                if not self.destinations:
                    self.stop()
                    rate.sleep()
                    continue

            speed = self.get_speed(x, y)

            if time.time() - last_time > 3:
                rospy.loginfo('Current location %s, %s, %s' % (self.x, self.y, self.theta))
                last_time = time.time()

            self.velocity_publisher.publish(speed)
            rate.sleep()

        speed = Twist()
        speed.linear.x = 0
        speed.angular.z = 0
        self.velocity_publisher.publish(speed)

    def at_destination(self, x, y):
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2) <= TOLERANCE

    def rotation(self, theta1, theta2):
        theta1 %= T360
        theta2 %= T360
        delta = (theta1 - theta2) % T360
        if delta > math.pi:
            return delta - T360
        return delta

    def get_speed(self, x, y):
        fa = 1
        fl = 2

        speed = Twist()
        speed.linear.x = 0
        speed.angular.z = 0

        delta_x = x - self.x
        delta_y = y - self.y

        angle_to_goal = math.atan2(delta_y, delta_x)
        delta_theta = self.rotation(angle_to_goal, self.theta)
        delta_distance = math.sqrt(delta_x ** 2 + delta_y ** 2)

        if abs(delta_theta) > 0.3:
            speed.angular.z = max(-0.3, min(fa * delta_theta, 0.3))
            return speed

        if abs(delta_distance) > 1:
            speed.linear.x = max(-0.5, min(fl * delta_distance, 0.5))
            return speed

        if abs(delta_theta) > TOLERANCE:
            speed.angular.z = delta_theta
        if abs(delta_distance) > TOLERANCE:
            speed.linear.x = delta_distance
        return speed

    def stop(self):
        rospy.loginfo('Stopping the robot')
        speed = Twist()
        speed.linear.x = 0
        speed.angular.z = 0
        self.velocity_publisher.publish(speed)


if __name__ == '__main__':
    try:
        GOTO().run()
    except rospy.ROSInterruptException:
        pass
