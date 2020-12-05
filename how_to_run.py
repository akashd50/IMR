# coding=utf-8
import rospy
from geometry_msgs.msg import Twist
from math import radians
import math
import cv2
from robot import Robot
import assignment1 as a1
import numpy as np
from MapDisplayThread import MapDisplayThread
import display


def Run(robot):
    # Open the map image and create a new thread for display
    image = cv2.imread("field.png", cv2.IMREAD_UNCHANGED)
    map_thread = MapDisplayThread(1, "map_thread", image)
    map_thread.start()

    # Testing updating the image in the thread, seems to be working fine
    # Can remove this code later on
    # new_image = image.copy()
    # new_image = display.draw_robot((100, 400), -45, image)
    # map_thread.set_image(new_image)

    do_robot_stuff(robot, map_thread, image)


def do_robot_stuff(robot, map_thread, map_image):
    global DEF_LARGE_VAL
    DEF_LARGE_VAL = 999999

    angle_to_goal = 0
    rotated_due_to_obstacle = False
    rotated_due_to_obstacle_dir = 0
    is_goal_visible = False

    # get the image from robot
    image = robot.get_image()
    frame_height, frame_width, _ = image.shape
    # Video writer for debugging
    # out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 4, (frame_width, frame_height))
    while True:
        # rotate the bot in the direction of goal

        # find obstacles in current direction
        image = robot.get_image()

        # pass the image to extract the obstacle and goal data
        frame_for_drawing, obstacles, blue_goal_data = photo_mode(image)
        rows, cols, channels = image.shape
        goal_x, goal_y, goal_area = blue_goal_data

        # check if the goal is visible
        if (goal_area <= 0) | (goal_area >= rows * cols):
            is_goal_visible = False
        else:
            is_goal_visible = True

        # Find the possible obstacles in range
        obstacles_in_collision_range = find_obstacles_in_collision_range(obstacles, rows, cols)

        num_obs_in_coll_range = len(obstacles_in_collision_range)

        if num_obs_in_coll_range == 1:
            current_obstacle = obstacles_in_collision_range[0]

            if rotated_due_to_obstacle:
                # If the robot was previously in the middle of rotation due to an obstacle, keep rotating
                # unless there's enough space to pass through

                if (abs(current_obstacle.closest_x_to_center_nrm) > 0.8) & (current_obstacle.size_ratio < 0.2):
                    # check if there's enough space to pass through
                    # move straight
                    move_forward(robot, 0.2, 1)
                else:
                    # otherwise keep rotating in the same direction
                    angle_to_rotate = rotated_due_to_obstacle_dir * 10
                    rotate_z(robot, angle_to_rotate, 1)
                    angle_to_goal += angle_to_rotate + (angle_to_rotate * 0.1)
            else:
                # otherwise find the angle of rotation for the current obstacle and rotate the robot
                angle_to_rotate = get_angle_to_rotate(current_obstacle)
                rotate_z(robot, angle_to_rotate, 1)
                angle_to_goal += angle_to_rotate + (angle_to_rotate * 0.1)
                rotated_due_to_obstacle = True
                rotated_due_to_obstacle_dir = angle_to_rotate / abs(angle_to_rotate)

            # If we ever see a potential collide-able obstacle in the current iteration,
            # we always skip the entire rest of loop
            continue
        elif num_obs_in_coll_range >= 2:
            angle_to_rotate = get_angle_to_rotate_multiple(obstacles_in_collision_range)
            rotate_z(robot, angle_to_rotate, 1)
            angle_to_goal += angle_to_rotate + (angle_to_rotate * 0.1)
            rotated_due_to_obstacle = True
            rotated_due_to_obstacle_dir = angle_to_rotate / abs(angle_to_rotate)
            continue

        # correct direction
        # print("Goal Data: ", blue_goal_data)
        if not is_goal_visible:
            # cant find goal

            if rotated_due_to_obstacle:
                # if the robot previously rotated due to an obstacle and there's no obstacle in the path,
                # move forward first
                move_forward(robot, 0.5, 1)
                rotated_due_to_obstacle = False
            else:
                # else continue rotating to find the goal
                rotate_z(robot, 40, 1)
                angle_to_goal += 40 + 40 * 0.1
        else:
            # if goal is visible

            if rotated_due_to_obstacle:
                # if the robot previously rotated due to an obstacle and there's no obstacle in the path,
                # move forward first

                move_forward(robot, 0.2, 1)
                rotated_due_to_obstacle = False
            else:
                # else correct the direction to the goal or move forward

                bottom = [float(cols) / 2, rows]
                vec_to_goal = [goal_x - bottom[0], abs(goal_y - bottom[1])]
                normalized_vec = a1.normalize(vec_to_goal)
                r_angle = math.degrees(math.atan2(normalized_vec[1], normalized_vec[0])) - 90

                # if the angle to goal is between 10 degrees move forward otherwise correct direction and continue
                if abs(r_angle) > 5:
                    angle_to_rotate = r_angle
                    rotate_z(robot, angle_to_rotate * 0.8, 1)
                    angle_to_goal += angle_to_rotate * 0.8 + (angle_to_rotate * 0.8) * 0.1
                else:
                    move_forward(robot, 0.2, 1)

        # out.write(frame_for_drawing)

    # out.release()
    # print("Angle to goal: ", angle_to_goal)
    # cv2.waitKey(0)


def find_obstacles_in_collision_range(obstacle_data, rows, cols):
    # From the obstacles in the list finds which obstacles can potentially collide with the robot.
    # Does so using a size threshold which is estimated using the position of the obstacle in
    # the frame and the height of the obstacle and I take certain percentage of the total frame size as the threshold
    # depending on the parameters described above

    frame_size = rows * cols
    obstacles_in_collision_range = []
    for obstacle in obstacle_data:
        x = obstacle.center_x
        y = obstacle.center_y
        size = obstacle.area
        width = obstacle.width
        height = obstacle.height

        size_ratio = float(size) / float(frame_size)
        height_ratio = float(height) / float(rows)
        center_normalized_x, center_angle = find_normalized_pos_and_angle(x, y, rows, cols)
        closest_x_to_view = find_closest_obstacle_x_to_view(obstacle, rows, cols)
        normalized_x, angle = find_normalized_pos_and_angle(closest_x_to_view, y, rows, cols)

        size_threshold = frame_size * max(abs(normalized_x), 0.1) * (1.6 - height_ratio)
        size_threshold_passed = size > size_threshold

        if size_threshold_passed:
            obstacle.angle = angle
            obstacle.size_ratio = size_ratio
            obstacle.center_x_nrm = center_normalized_x
            obstacle.closest_x_to_center = closest_x_to_view
            obstacle.closest_x_to_center_nrm = normalized_x
            obstacles_in_collision_range.append(obstacle)

    return obstacles_in_collision_range


def get_angle_to_rotate(obstacle_data):
    # Find the anlge of rotation if there's one obstacle in the view
    # for single obstacle

    additional_angle_offset = 0
    if (abs(obstacle_data.size_ratio) > 0.25) & (abs(obstacle_data.center_x_nrm) > 0.6):
        additional_angle_offset = 20

    if obstacle_data.center_x_nrm > 0:
        return 10 + additional_angle_offset
    else:
        return -(10 + additional_angle_offset)


def get_angle_to_rotate_multiple(obstacles):
    # Finds the angle to rotate to if there are more than one obstacle in the view
    # for multiple obstacles

    def size_sort(obs):
        return obs.size_ratio

    if len(obstacles) > 2:
        obstacles.sort(size_sort)

    obs1 = obstacles[0]
    obs2 = obstacles[1]
    dist_bw_obs = abs(obs1.closest_x_to_center_nrm - obs2.closest_x_to_center_nrm)
    if dist_bw_obs < 1.0:
        return 60
    else:
        return 0


def find_closest_obstacle_x_to_view(obstacle, frame_rows, frame_cols):
    #  finds which x (cols) in pixels, the obstacle is closest to the camera i.e. finds the closest
    #  point of an obstacle to the camera

    normalized_x, angle = find_normalized_pos_and_angle(obstacle.center_x, obstacle.center_y, frame_rows, frame_cols)
    if normalized_x < 0:
        return obstacle.center_x + (obstacle.width / 2)
    else:
        return obstacle.center_x - (obstacle.width / 2)


def find_normalized_pos_and_angle(x, y, rows, cols):
    # normalizes the x position between -1 and 1 (-1 being the left and 1 being the right, 0 being the center)
    # and finds the angle from the center-bottom of the view to the x and y

    normalized_x = (float(x * 2) / float(cols)) - 1
    bottom = [float(cols) / 2, rows]
    vec_to_obs = [x - bottom[0], abs(y - bottom[1])]
    normalized_vec = a1.normalize(vec_to_obs)
    angle = math.degrees(math.atan2(normalized_vec[1], normalized_vec[0])) - 90
    return normalized_x, angle


def rotate_z(robot, deg_per_sec, seconds):
    # helper function for rotating the robot forward
    rate = rospy.Rate(10)
    twist = Twist()
    twist.angular.z = radians(deg_per_sec)
    iter_time = 10 * seconds
    for i in xrange(0, iter_time):
        robot.publish_twist(twist)
        rate.sleep()

    # instantly stops rotation as soon as the loop finishes
    # this allows to avoid the discrepancies in extra rotation
    twist.angular.z = 0
    robot.publish_twist(twist)


def move_forward(robot, meters_per_sec, seconds):
    # helper function for moving the robot forward
    rate = rospy.Rate(10)
    twist = Twist()
    twist.linear.x = meters_per_sec
    iter_time = 10 * seconds
    for i in xrange(0, iter_time):
        robot.publish_twist(twist)
        rate.sleep()


def photo_mode(image):
    frame = image
    frame_for_drawing = frame.copy()
    rows, cols, channels = frame.shape
    # Get the field parameters line the horizon height, and the convex hull of the field
    horizon_height, outer_field_edges = a1.do_field_ops(frame)

    #  Do Obstacle Detection. positional_data_obstacles contains the position and size of obstacles
    positional_data_obstacles, obstacle_contours = a1.find_obstacles(frame, frame_for_drawing)

    obstacle_objects = []
    for obs in positional_data_obstacles:
        obstacle_objects.append(Obstacle(obs))

    # always keep track of the angle wtr to the goal post. Update it everytime a rotational turn is made

    # find blue goal
    pos_x_blue, pos_y_blue, blue_area = a1.find_goal(frame, np.asarray([100, 180, 180]), np.asarray([130, 255, 255]),
                                                     frame_for_drawing)

    # find yellow goal
    pos_x_yellow, pos_y_yellow, yellow_area = a1.find_goal(frame, np.asarray([10, 80, 230]), np.asarray([40, 180, 255]),
                                                           frame_for_drawing)

    # Not Doing line detection since we don't need it for this assignment

    # frame_for_drawing = draw_field(frame, obstacle_contours, horizon_height, outer_field_edges, frame_for_drawing)
    return frame_for_drawing, obstacle_objects, [pos_x_blue, pos_y_blue, blue_area]


def find_edges_close_to_horizon_height(frame, horizon_height, field_edges):
    # Finds the edges close to the horizon of the field in order so essentially finds the highest lines in frame
    # which tells us at what point the robot needs to rotate to not go out of the boundary line'

    closest_to_horizon = []
    rows, cols, _ = frame.shape

    for line in field_edges:
        if not is_frame_line(rows, cols, line) & (a1.get_line_len(line) > (rows * 0.1)):
            closest_to_horizon.append(line)

    return closest_to_horizon


def is_frame_line(rows, cols, line):
    # Returns whether the line around the goal field edge is a line along the frame's
    # boundary or an actual field edge line
    dist_bw_y1 = abs(rows - line[1])
    dist_bw_y2 = abs(rows - line[3])
    dist_bw_x1 = abs(cols - line[0])
    dist_bw_x2 = abs(cols - line[2])

    is_frame_line = ((dist_bw_y1 < 10) & (dist_bw_y2 < 10)) | ((dist_bw_x1 < 10) & (dist_bw_x2 < 10))
    return is_frame_line


# Obstacle class for easier organization
class Obstacle:
    center_x = 0
    center_y = 0
    area = 0
    width = 0
    height = 0
    angle_from_center = 0
    size_ratio = 0
    center_x_nrm = 0
    closest_x_to_center = 0
    closest_x_to_center_nrm = 0

    def __init__(self, position_data):
        self.center_x, self.center_y, self.area, self.width, self.height = position_data


if __name__ == '__main__':
    robot = Robot()
    Run(robot)
