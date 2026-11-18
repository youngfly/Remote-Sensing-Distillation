_base_ = './faster_rcnn_r50_nwpu.py'
model = dict(pretrained='/home/airstudio/pretrained model/resnet101-5d3b4d8f.pth', backbone=dict(depth=101))