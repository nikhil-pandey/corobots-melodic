"""
file: line.py
language: python3
author: np7803@rit.edu Nikhil Pandey
description: The line class.
"""

from point import Point


class Line:
    """
    The line class.
    """

    def __init__(self, slope, intercept):
        """
        Creates a line in slope intercept form.
        y = Mx + C
        where M is the slope and C is the intercept.
        :param slope: The slope of the line.
        :param intercept: The intercept of the line.
        """
        self.slope = slope
        self.intercept = intercept

    def intersection(self, line):
        """
        Computes intersection with another line.
        :param line: Another line.
        :return: Point or None
        """
        if line is None or self.slope == line.slope:
            return

        x_intercept = self.x_intercept(line)
        y_intercept = (self.slope * x_intercept) + self.intercept

        return Point(x_intercept, y_intercept)

    def contains(self, point):
        """
        Checks if the point satisfies the line.
        :param point: The point.
        :return: True if line contains the point else False
        """
        if self.slope == float('inf'):
            return self.intercept == point.x

        return self.slope * point.x + self.intercept == point.y

    def x_intercept(self, line):
        """
        Intercept with the given line.
        :param line: The line.
        :return: The intercept.
        """
        if self.slope == float('inf'):
            return -self.intercept

        if line.slope == float('inf'):
            return -line.intercept

        return (self.intercept - line.intercept) / (line.slope - self.slope)

    def __repr__(self):
        return 'y = %sx %s %s' % (self.slope, '-' if self.intercept < 0 else '+', abs(self.intercept))
