import os
from PIL import Image
import numpy as np
from tqdm import tqdm
# from scipy.misc import imread

R = 0
G = 0
B = 0
R_S = 0
G_S = 0
B_S = 0
num=0
pix_num = 0

# todo 统计所有的
dior_path = '/home/airstudio/data/datapre/pic/'
dota_path = '/home/airstudio/data/datapre/pic_dota2/'
fair_path = '/home/airstudio/data/datapre/pic_fair/'
pathlist = [dior_path, dota_path, fair_path]

nwpu1 = '/home/airstudio/data/NWPU/train2017/'
nwpu2 = '/home/airstudio/data/NWPU/val2017/'
dota = '/home/airstudio/data/dota_data/train1024/images/'
dota2 = '/home/airstudio/data/dota_data/val1024/images/'
dota3 = '/home/airstudio/data/dota_data/test1024/images/'
dior = '/home/airstudio/data/DIOR/JPEGImages/'
pathlist2 = [dior]

voc_2007 = '/home/airstudio/data/VOCdevkit/VOC2007/JPEGImages/'
voc_2012 = '/home/airstudio/data/VOCdevkit/VOC2012/JPEGImages/'
pathlist3 = [voc_2007, voc_2012]

for j in pathlist3:
    pic = os.listdir(j)
    # for i, im in enumerate(pic):
    for im in tqdm(pic):
        # print(i)
        image = Image.open(j + im)
        img = image.convert('RGB')
        w, h = image.size
        pix_num = w*h
        img = np.array(img)
        R_temp = img[:,:,0]
        R += np.sum(R_temp)
        R_S += np.sum(np.power(R_temp, 2.0))

        G_temp = img[:,:,1]
        G += np.sum(G_temp)
        G_S += np.sum(np.power(G_temp, 2.0))

        B_temp = img[:,:,2]
        B += np.sum(B_temp)
        B_S += np.sum(np.power(B_temp, 2.0))
        num = pix_num + num

R_mean = R / num
G_mean = G / num
B_mean = B / num

R_std = np.sqrt(R_S/num-R_mean*R_mean)
G_std = np.sqrt(G_S/num-G_mean*G_mean)
B_std = np.sqrt(B_S/num-B_mean*B_mean)

print(R_mean,G_mean,B_mean)
print(R_std,G_std,B_std)