import numpy as np
import cv2
from random import *
import os

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
        for line_cnt in range(line_count):
            cv2.line(self.blank, (0, num1 + interval * line_cnt),
                     (height, num2 + interval * line_cnt), (255, 255, 255), line_width)
            cv2.imwrite(save_dir + f'/{i+1}.bmp', self.blank)

    def lines_vertical(self):
        for line_cnt in range(line_count):
            cv2.line(self.blank, (num1 + interval * line_cnt, 0),
                     (num2 + interval * line_cnt, width), (255, 255, 255), line_width)
            cv2.imwrite(save_dir + f'/{j+1}.bmp', self.blank)

width = 512
height = 512
line_width = 3
cnt = 125
save_dir = './pseudo_lables'
os.makedirs(save_dir, exist_ok=True)

for i in range(0, cnt):
    num1, num2, line_count, interval = param()
    lables = pseudo_lable(width, height, line_width)
    lables.lines()
for j in range(cnt, cnt * 2):
    num1, num2, line_count, interval = param()
    lables = pseudo_lable(width, height, line_width)
    lables.lines_vertical()