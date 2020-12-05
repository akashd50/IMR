import display
import cv2
from MapDisplayThread import MapDisplayThread


class MapHelper:
    def __init__(self):
        self.robot_rotation = 0

        # defaulting the translation for now
        self.robot_translation = (150, 300)
        self.map = cv2.imread("field.png", cv2.IMREAD_UNCHANGED)
        self.map_thread = MapDisplayThread(1, "map_thread", self.map)
        self.map_thread.start()

    def update_rotation(self, new_rotation):
        self.robot_rotation = new_rotation
        self.update_map()

    def update_translation(self, new_translation):
        self.robot_translation = new_translation
        self.update_map()

    def update_map(self):
        new_image = self.map.copy()
        new_image = display.draw_robot(self.robot_translation, self.robot_rotation, new_image)
        self.map_thread.set_image(new_image)
