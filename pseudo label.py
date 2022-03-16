import numpy as np
import cv2
from random import *

def param():
    num1 = randrange(0, width + 1)
    num2 = randrange(0, height + 1)
    line_count = randint(1, 3)
    interval = randrange(50, 400)
    return num1, num2, line_count, interval


class pseudo_lable():

    def __init__(self, width, height, line_width):
        self.blank = np.zeros((width, height, 3), np.uint8)

    def lines(self):
        for j in range(line_count):
            cv2.line(self.blank, (0, num1 + interval * j),
                     (height, num2 + interval * j), (255, 255, 255), line_width)
            cv2.imwrite(f'./prpr/{i + 1}.bmp', self.blank)

    def lines_vertical(self):
        for j in range(line_count):
            cv2.line(self.blank, (num1 + interval * j, 0),
                     (num2 + interval * j, width), (255, 255, 255), line_width)
            cv2.imwrite(f'./prpr/{k}.bmp', self.blank)


width = 512
height = 512
line_width = 3
cnt = 125
for i in range(0, cnt):
    num1, num2, line_count, interval = param()
    lables = pseudo_lable(width, height, line_width)
    lables.lines()
for k in range(cnt, cnt * 2):
    num1, num2, line_count, interval = param()
    lables = pseudo_lable(width, height, line_width)
    lables.lines_vertical()