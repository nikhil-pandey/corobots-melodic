from fractions import Fraction


class Point:
    """
    The point class.
    """

    def __init__(self, x, y, segment=None):
        """
        (x, y) points. Uses fraction to prevent round-off error.
        :param x: The x-coordinate.
        :param y: The y-coordinate.
        :param segment: The segment point belongs to.
        """
        self.x = Fraction(x)
        self.y = Fraction(y)
        self.segment = segment

    def is_left_point(self):
        return self.segment is not None and self.segment.left_point == self

    def on_right_of(self, line):
        if line.contains(self):
            return False

        if line.slope == float('inf'):
            return self.x > -line.intercept

        raise ValueError("Not implemented")

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return '(%s, %s)' % (float(self.x), float(self.y))

    def __hash__(self):
        return (self.x, self.y).__hash__()
