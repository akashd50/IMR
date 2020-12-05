import threading
import display
import cv2


class MapDisplayThread(threading.Thread):
    def __init__(self, thread_id, name, image):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.image = image

    def run(self):
        print("Starting thread: ", self.thread_id)
        display_map(self.image)

    def set_image(self, image):
        self.image = image


def display_map(image):
    while True:
        display.show_map_window(image)
        if cv2.waitKey(0):
            break
    cv2.destroyAllWindows()
