import math
from os.path import join
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

from segment import Segment

DATASET_SAVE_DIR = '/home/np7803/create_ws/src/corobot_follower/data/dataset/'
FEATURE_FILE = join(DATASET_SAVE_DIR, 'features.csv')

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
    x1 = p1[0] - p0[0]
    y1 = p1[1] - p0[1]
    x2 = p1[0] - p2[0]
    y2 = p1[1] - p2[1]
    x3 = p2[0] - p0[0]
    y3 = p2[1] - p0[1]
    b = x1 * x1 + y1 * y1
    a = x2 * x2 + y2 * y2
    c = x3 * x3 + y3 * y3
    return math.acos((a + b - c) / math.sqrt(4 * a * b))


def intersect((p1, p2), (p3, p4)):
    return Segment(p1, p2).intersection(Segment(p3, p4)) is not None


class Feature:
    def __init__(self, file_name, kp_count, x, y, height, width):
        self.file_name = file_name
        self.img = None
        self.kp_count = kp_count
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.key_points = {}

    def show(self):
        img = self.get_bb_image()
        for key in self.key_points:
            self.draw_key_point(img, key)
        plt.imshow(img)
        plt.show()

    def show_kp(self, kp_num):
        img = self.get_bb_image()
        self.draw_key_point(img, kp_num)
        plt.imshow(img)
        plt.show()

    def draw_key_point(self, img, kp_num):
        x = self.key_points[kp_num][0] - self.x
        y = self.key_points[kp_num][1] - self.y
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 255), 1)

    def get_bb_image(self):
        if self.img is None:
            self.img = cv2.imread(join(DATASET_SAVE_DIR, self.file_name))

        img = np.array(self.img, dtype=np.uint8)
        img = img[int(self.y):int(self.y + self.height), int(self.x):int(self.x + self.width)]
        # cv2.rectangle(img, (int(self.x), int(self.y + self.height)),
        #               (int(self.x + self.width), int(self.y)),
        #               (0, 255, 0), 2)
        return img


features = []
with open(FEATURE_FILE, 'r') as f:
    for line in f:
        items = line.strip().split(',')
        file_name = items[0]
        feature = Feature(file_name, int(items[1]), *map(float, items[2:6]))
        c = 0
        for idx in range(6, len(items), 2):
            feature.key_points[c] = float(items[idx]), float(items[idx + 1])
            c += 1
        features.append(feature)

fts = [f for f in features if f.kp_count >= 18]
ins = fts[-1]


def has_raised_hand(feature):
    if feature.key_points[C_LEFT_SHOULDER][0] > feature.key_points[C_RIGHT_SHOULDER][0]:
        # print('Dont turn your back on me!')
        return False

    if angle_between(feature.key_points[C_LEFT_FOOT], feature.key_points[C_LEFT_KNEE],
                     feature.key_points[C_LEFT_HIP]) < 2.5 or \
            angle_between(feature.key_points[C_RIGHT_FOOT], feature.key_points[C_RIGHT_KNEE],
                          feature.key_points[C_RIGHT_HIP]) < 2.5:
        # print('Not standing straight!')
        return False

    if all([True if x != 0 else False for x in
            [feature.key_points[C_LEFT_WRIST][1], feature.key_points[C_LEFT_ARM][1],
             feature.key_points[C_LEFT_WRIST][0], feature.key_points[C_LEFT_SHOULDER][0]]]):
        if feature.key_points[C_LEFT_WRIST][1] < feature.key_points[C_LEFT_ARM][1] and \
                feature.key_points[C_LEFT_WRIST][0] < feature.key_points[C_LEFT_SHOULDER][0] \
                and feature.key_points[C_LEFT_WRIST][0] < feature.key_points[C_LEFT_ARM][0]:
            if angle_between(feature.key_points[C_LEFT_WRIST], feature.key_points[C_LEFT_ARM],
                             feature.key_points[C_LEFT_SHOULDER]) < 0.5:
                return True
            # print('Left hand angle not so small')
        # print('Left wrist not in right place')

    if all([True if x != 0 else False for x in
            [feature.key_points[C_RIGHT_WRIST][1], feature.key_points[C_RIGHT_ARM][1],
             feature.key_points[C_RIGHT_WRIST][0], feature.key_points[C_RIGHT_SHOULDER][0]]]):
        if feature.key_points[C_RIGHT_WRIST][1] > feature.key_points[C_RIGHT_ARM][1] and \
                feature.key_points[C_RIGHT_WRIST][0] > feature.key_points[C_RIGHT_SHOULDER][0] \
                and feature.key_points[C_RIGHT_WRIST][0] > feature.key_points[C_RIGHT_ARM][0]:
            if angle_between(feature.key_points[C_RIGHT_WRIST], feature.key_points[C_RIGHT_ARM],
                             feature.key_points[C_RIGHT_SHOULDER]) < 0.5:
                return True
            # print('right hand angle not so small')
        # print('right hand not in right place')
    return False


def has_crossed_hand(feature):
    if feature.key_points[C_LEFT_SHOULDER][0] > feature.key_points[C_RIGHT_SHOULDER][0]:
        print('Dont turn your back on me!')
        return False

    if angle_between(feature.key_points[C_LEFT_FOOT], feature.key_points[C_LEFT_KNEE],
                     feature.key_points[C_LEFT_HIP]) < 2.5 or \
            angle_between(feature.key_points[C_RIGHT_FOOT], feature.key_points[C_RIGHT_KNEE],
                          feature.key_points[C_RIGHT_HIP]) < 2.5:
        print('Not standing straight!')
        return False

    left_wrist = feature.key_points[C_LEFT_WRIST]
    right_wrist = feature.key_points[C_RIGHT_WRIST]
    left_arm = feature.key_points[C_LEFT_ARM]
    right_arm = feature.key_points[C_RIGHT_ARM]

    if not all([True if x != 0 else False for x in [left_wrist, left_arm, right_wrist, right_arm]]):
        print('one of the required key point is missing')
        return False

    if not intersect((left_wrist, left_arm), (right_wrist, right_arm)):
        print('Does not intersect')
        return False

    return True


a = [f for f in fts if has_crossed_hand(f)]
