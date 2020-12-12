# Obstacle class for easier organization
IMAGE_CENTRE = 320      # IMAGE WIDTH  = 640

LEFT = -1
CENTRE = 0
RIGHT = 1

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


    def orientation(self):
        """
        detects if the obstacle is left right or centrelined
        """
        if abs(self.center_x -  IMAGE_CENTRE) < self.width:
            # We have a cere lined
            return CENTRE
        else:
            if closest_x_to_center >= IMAGE_CENTRE:
                return RIGHT
            else:
                return LEFT
