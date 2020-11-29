import cv2
import numpy as np
import math

BLACK = (0, 0, 0)
_TRIANGLE_HEIGHT =  30
_TRIANGLE_BASE = 18
_WINDOW_NAME = "MAPPING"

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin[0], origin[1]
    px, py = point[0], point[1]

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return (int(round(qx)), int(round(qy)))

def robot_triangle_coordinates(point, angle):
    """
    generate robot coordinate
    args: 
        point: (x, y)
        angle: degree angle, anti-clockwise deflection is 
                postive and clockwise deflection is neagative

    returns:  (x_top, y_top), (x_right, y_right), (x_left, y_left) 
    """
    angle = math.radians(-angle)
    top = (point[0], point[1] - _TRIANGLE_HEIGHT / 2)
    base_point = (point[0], point[1] + _TRIANGLE_HEIGHT / 2)
    left = (base_point[0] - _TRIANGLE_BASE / 2, base_point[1])
    right = (base_point[0] + _TRIANGLE_BASE / 2, base_point[1])
    return rotate(point,top, angle), rotate(point,right, angle), rotate(point,left, angle)

def draw_robot(loc_point, angle):
    """
    draws the robot triangle in the image
    args: loc_point : (x, y)
          angle : angular deflection in degree
    """
    im = cv2.imread("field.png", cv2.IMREAD_UNCHANGED)
    pt1, pt2, pt3 = robot_triangle_coordinates(loc_point, angle)
    cv2.fillPoly(im, np.array([[pt1, pt2, pt3]]), BLACK)
    return im
    

def test():
    """
    Test module functionality
    """
    image = draw_robot((100, 400), -45)
    cv2.imshow(_WINDOW_NAME, image)
    cv2.waitKey(3000)
