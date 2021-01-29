import threading
import display
import cv2


class MapDisplayThread(threading.Thread):
    def __init__(self, thread_id, name, image):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.image = image
        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())

    def run(self):
        print("Starting thread: ", self.thread_id)
        while True:
            with self.pause_cond:
                while self.paused:
                    self.pause_cond.wait()

            display.show_map_window(self.image)

            if cv2.waitKey(1) == ord('q'):
                break
        cv2.destroyAllWindows()

    def set_image(self, img):
        self.pause()
        self.image = img
        self.resume()

    def pause(self):
        self.paused = True
        self.pause_cond.acquire()

    def resume(self):
        self.paused = False
        self.pause_cond.notify()
        self.pause_cond.release()
