import display
import cv2
import math
from MapDisplayThread import MapDisplayThread


class MapHelper:
    _FIELD_IMAGE = "field.png"
    _IMAGE_HEIGHT = 600
    _IMAGE_WIDTH = 300

    def __init__(self):
        self.robot_rotation = 0
        self.robot_translation = (0, 0)
        self.map = cv2.imread(self._FIELD_IMAGE, cv2.IMREAD_UNCHANGED)
        self.map_thread = MapDisplayThread(1, "map_thread", self.map)
        self.map_thread.start()
        self.recorded_obstacles = []

    def set_robot_rotation(self, new_rotation):
        """
            :param new_rotation: (float)
            new rotation of the robot
        """
        self.robot_rotation = new_rotation

    def update_robot_rotation(self, rotation_diff):
        """
            :param rotation_diff: (float)
            change in rotation of the robot
        """
        self.robot_rotation += rotation_diff

    def set_robot_translation(self, new_translation):
        """
            :param new_translation: (x, y)
            new location of the robot in the field
        """
        self.robot_translation = new_translation

    def translate_robot(self, translation_amount):
        """
            :param translation_amount: (float)
            change in the robot's position in the direction of the current rotation
        """
        new_pos_x, new_pos_y = display.translate_point(self.robot_translation, translation_amount, self.robot_rotation)
        self.set_robot_translation((new_pos_x, new_pos_y))

    def refresh_map(self):
        """
            Copy from the original map image and redraw
            the robot at new position with new translation
        """
        new_image = self.map.copy()
        new_image = display.draw_robot(self.robot_translation, self.robot_rotation, new_image)
        for obs in self.recorded_obstacles:
            display.draw_obstacle((obs.center_x, obs.center_y), (0, 0), new_image)
        self.map_thread.set_image(new_image)

    def get_distance(self, dist_func, obstacle):
        """
        given a obstacle_height returns its distance
        parameters:
            obstacle_height: obstacle height
        """
        if dist_func == 1:
            dist_for_ratio_one = 0.5
            if (abs(obstacle.center_x_nrm) > 0.6) & (obstacle.height_ratio > 0.8):
                dist_for_ratio_one = 0.7
            return 2 * (1.0 - obstacle.height_ratio) + dist_for_ratio_one
        elif dist_func == 2:
            return 0.7 - obstacle.height_ratio + 1.0

    def map_obstacles(self, obstacles):
        for obstacle in obstacles:
            distance = self.get_distance(1, obstacle)
            # distance2 = self.get_distance(2, obstacle)
            angle_to_obstacle = -(obstacle.center_x_nrm * 45)
            if (abs(obstacle.center_x_nrm) > 0.7) & (obstacle.height_ratio > 0.8):
                angle_to_obstacle += 10
            x, y = display.translate_point(self.robot_translation, distance,
                                           self.robot_rotation + angle_to_obstacle)
            # x2, y2 = display.translate_point(self.robot_translation, distance2, self.robot_rotation + angle_to_obstacle)
            # new_obs_data = ObstacleObservationData(int((x + x2) / 2), int((y + y2) / 2), 0)
            new_obs_data = ObstacleObservationData(int(x), int(y), 1.0, obstacle.size_ratio)
            new_obs_data.max_size_ratio = obstacle.size_ratio

            merged_with_old = False
            for old_data in self.recorded_obstacles:
                if old_data.check_intersection(new_obs_data):

                    if new_obs_data.max_size_ratio > old_data.max_size_ratio:
                        old_data.max_size_ratio = new_obs_data.max_size_ratio

                    old_data.merge_two_obstacles(new_obs_data)
                    merged_with_old = True

            if not merged_with_old:
                self.recorded_obstacles.append(new_obs_data)
        # self.map_thread.set_image(new_image)


class ObstacleObservationData:

    def __init__(self, cx, cy, weight, size_rat):
        self.center_x = cx
        self.center_y = cy
        self.weight = weight
        self.size_ratio = size_rat
        self.max_size_ratio = 0.0
        self.probability = 0.0

    def check_intersection(self, obsObvData):
        # checks to see if this obstacle rectangle overlaps with another one
        l1x, l1y, r1x, r1y = self.center_x - 25, self.center_y - 25, self.center_x + 25, self.center_y + 25
        l2x, l2y, r2x, r2y = obsObvData.center_x - 25, obsObvData.center_y - 25, obsObvData.center_x + 25, obsObvData.center_y + 25

        if (l1x >= r2x) or (l2x >= r1x):
            return False

        if (l1y >= r2y) or (l2y >= r1y):
            return False
        return True

    def merge_two_obstacles(self, obsObvData):
        x_diff = self.center_x - obsObvData.center_x
        y_diff = self.center_y - obsObvData.center_y
        # ratio_to_use = obsObvData.size_ratio/self.max_size_ratio
        self.center_x = self.center_x - int(float(x_diff) * obsObvData.size_ratio)
        self.center_y = self.center_y - int(float(y_diff) * obsObvData.size_ratio)

        # self.center_x = (self.center_x + obsObvData.center_x) / 2
        # self.center_y = (self.center_y + obsObvData.center_y) / 2
