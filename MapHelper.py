import display
import cv2
from MapDisplayThread import MapDisplayThread


class MapHelper:
    _FIELD_IMAGE = "field.png"

    def __init__(self):
        self.robot_rotation = 0
        self.robot_translation = (0, 0)
        self.map = cv2.imread(self._FIELD_IMAGE, cv2.IMREAD_UNCHANGED)
        self.map_thread = MapDisplayThread(1, "map_thread", self.map)
        self.map_thread.start()

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
        self.map_thread.set_image(new_image)

    def get_distance(self, obstacle_height):
        """
        given a obstacle_height returns its distance
        parameters:
            obstacle_height: obstacle height

        """
        return (int((199- obstacle_height) / 27))