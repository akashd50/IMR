import display
import cv2
from MapDisplayThread import MapDisplayThread


class MapHelper:
    def __init__(self):
        self.map = cv2.imread("field.png", cv2.IMREAD_UNCHANGED)
        self.map_thread = MapDisplayThread(1, "map_thread", self.map)
        self.map_thread.start()

    def update_rotation(self, rotation):
        new_image = self.map.copy()
        new_image = display.draw_robot((100, 400), rotation, new_image)
        self.map_thread.set_image(new_image)
