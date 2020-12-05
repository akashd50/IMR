import cv2
import numpy as np
import math

_BLACK = (0, 0, 0)
_RED = (0, 0, 255)
_TRIANGLE_HEIGHT =  30
_TRIANGLE_BASE = 18
_WINDOW_NAME = "MAPPING"
_OBSTACLE_SIZE = 40

_IMAGE_HEIGHT = 600
_IMAGE_WIDTH = 300

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

def draw_robot(loc_point, angle, img):
    """
    draws the robot triangle in the image
    args: loc_point : (x, y)
          angle : angular deflection in degree
    """
    
    pt1, pt2, pt3 = robot_triangle_coordinates(loc_point, angle)
    cv2.fillPoly(img, np.array([[pt1, pt2, pt3]]), _BLACK)
    return img


def generate_obstacle_point(start, end):
    """
    Genetate 4 points for the obstacle representation
    args: start
          end
    returns: (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    """
    top_left = (start[0], start[1] - _OBSTACLE_SIZE)
    top_right = (end[0], end[1] - _OBSTACLE_SIZE)
    return start, end, top_right, top_left




def draw_obstacle(start, end, img):
    """
    Draw the obstacle box 
    args: 
            start   : left point
            end     : Right point
    """
    start, end, top_right, top_left = generate_obstacle_point(start, end)
    cv2.fillPoly(img, np.array([[start, end, top_right, top_left]]), _RED)
    return img


def show_map_window(image):
    """
        Draw the map image
        args:
                image : image of the map
    """
    cv2.imshow(_WINDOW_NAME, image)


def test():
    """
    Test module functionality
    """
    image = cv2.imread("field.png", cv2.IMREAD_UNCHANGED)
    image = draw_robot((100, 400), -45, image)
    image = draw_obstacle((100, 350), (100 + _OBSTACLE_SIZE, 350), image)
    cv2.imshow(_WINDOW_NAME, image)
    cv2.waitKey(3000)
