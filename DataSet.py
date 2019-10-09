import numpy as np
import cv2 as cv
import os


class DataSet:
    """
    shujuji
    """
    def __init__(self, image_path):
        self.image_path = image_path

    def data_resize(self, resize: tuple, if_gray: bool):
        if if_gray is True:
            img = cv.imread(self.image_path, 0)
        else:
            img = cv.imread(self.image_path, -1)
        img = cv.resize(img, resize, interpolation=cv.INTER_CUBIC)
        return img





