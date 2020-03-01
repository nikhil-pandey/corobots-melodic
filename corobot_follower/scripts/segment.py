from fractions import Fraction

from line import Line
from point import Point


def make_line(point1, point2):
    """
    Make line using 2-points.
    :param point1: The first point.
    :param point2: The second point.
    :return:
    """
    slope = Fraction(point2.y - point1.y, point2.x - point1.x)
    intercept = point1.y - slope * point1.x
    return Line(slope, intercept)


class Segment:
    """
    A line segment.
    """

    def __init__(self, point1, point2):
        if point1[0] > point2[0]:
            # Orders them based on x-coordinate
            point1, point2 = point2, point1
        self.left_point = Point(point1[0], point1[1], self)
        self.right_point = Point(point2[0], point2[1], self)
        self.line = make_line(self.left_point, self.right_point)

    def __eq__(self, other):
        return other is not None and self.left_point == other.left_point and self.right_point == other.right_point

    def __repr__(self):
        return '%s, %s -> %s' % (self.line, self.left_point, self.right_point)

    def __lt__(self, other):
        return self.left_point.y < other.left_point.y

    def __gt__(self, other):
        return self.left_point.y > other.left_point.y

    def __ge__(self, other):
        return self.left_point.y >= other.left_point.y

    def intersection(self, segment=None, line=None):
        if segment:
            point = self.line.intersection(segment.line)
            return None if point is None or (not self.contains(point)) or (not segment.contains(point)) else point
        if line:
            point = self.line.intersection(line)
            return None if point is None or (not self.contains(point)) else point

    def __hash__(self):
        return (self.left_point.x, self.left_point.y, self.right_point.x, self.right_point.y).__hash__()

    def contains(self, point):
        return self.left_point == point or self.right_point == point or \
               (self.line.contains(point) and \
                (self.left_point.x <= point.x <= self.right_point.x or \
                 self.left_point.x >= point.x >= self.right_point.x) and \
                (self.left_point.y <= point.y <= self.right_point.y or \
                 self.left_point.y >= point.y >= self.right_point.y))
