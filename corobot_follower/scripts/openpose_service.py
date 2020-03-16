#!/usr/bin/env python

ros_mode = True
try:
    import pyopenpose as op
    import cv2
    import time
    import sys
    import numpy as np
    from cv_bridge import CvBridge, CvBridgeError
    import rospy
    from corobot_follower.srv import EstimatePoseSrv, EstimatePoseSrvResponse
    from corobot_follower.msg import OpenPoseHumanList, OpenPoseHuman, BoundingBox, PointWithProb
except Exception as e:
    print(e)
    ros_mode = False

params = {
    'model_folder': '/home/np7803/openpose/models/',
    'body': 1,
    'model_pose': 'COCO',
    'net_resolution': '368x368',
}


class OpenposeService:
    def __init__(self):
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

    def estimate_pose(self, img):
        datum = op.Datum()
        datum.cvInputData = img
        self.opWrapper.emplaceAndPop([datum])
        return datum


class OpenposeROSWrapper:
    def __init__(self):
        self.node_name = 'openpose_service'
        rospy.init_node(self.node_name)

        self.cv_window_name = self.node_name
        cv2.namedWindow(self.cv_window_name)
        cv2.moveWindow(self.cv_window_name, 25, 75)

        self.openposeService = OpenposeService()
        self.bridge = CvBridge()
        rospy.Service('openpose', EstimatePoseSrv, self.estimate_pose)

    def estimate_pose(self, message):
        frame = self.bridge.imgmsg_to_cv2(message.image)
        data = self.openposeService.estimate_pose(frame)
        self.visualize(data)
        human_list_msg = OpenPoseHumanList()
        human_list = []
        if len(data.poseKeypoints.shape):
            for person in data.poseKeypoints:
                human = OpenPoseHuman()
                human.key_points_count = int(np.count_nonzero(person[:, 2]))
                min_x, max_x, min_y, max_y = None, None, None, None
                key_points = []
                for key_point in person:
                    key_points.append(PointWithProb(float(key_point[0]), float(key_point[1]), float(key_point[2])))
                    if key_point[2] == 0:
                        continue
                    if min_x is None or key_point[0] < min_x:
                        min_x = key_point[0]
                    if max_x is None or key_point[0] > max_x:
                        max_x = key_point[0]
                    if min_y is None or key_point[1] < min_y:
                        min_y = key_point[1]
                    if max_y is None or key_point[1] > max_y:
                        max_y = key_point[1]
                while len(key_points) < 25:
                    key_points.append(PointWithProb(0, 0, 0))
                human.key_points = key_points
                human.bounding_box = BoundingBox(float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y))
                human_list.append(human)
        human_list_msg.num_humans = len(human_list)
        human_list_msg.human_list = human_list
        return human_list_msg

    def run(self):
        rospy.spin()

    def cleanup(self):
        rospy.loginfo('Cleaning up')
        cv2.destroyAllWindows()

    def visualize(self, data):
        cv2.imshow(self.cv_window_name, data.cvOutputData)
        cv2.waitKey(1)


if __name__ == '__main__':
    if ros_mode:
        rospy.loginfo('Starting openpose service...')
        OpenposeROSWrapper().run()
    else:
        print('Running non ros mode....')
        service = OpenposeService()
        imagePaths = op.get_images_on_directory('/home/np7803/openpose/examples/media')
        for imagePath in imagePaths:
            d = service.estimate_pose(cv2.imread(imagePath))
            break