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

