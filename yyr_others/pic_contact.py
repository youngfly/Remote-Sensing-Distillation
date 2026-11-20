import cv2
import numpy as np
import os

path1 = '/home/airstudio/code/wpj/h_1/'
path2 = '/home/airstudio/code/wpj/h_2/'
path3 = '/home/airstudio/code/wpj/h_3/'

pic =  os.listdir(path1)
for i, img in enumerate(pic):
    img1 = cv2.imread(path1 + img)
    img2 = cv2.imread(path2 + img)
    img3 = img1 + img2
    cv2.imwrite(path3 + img,img3)
    # exit(0)