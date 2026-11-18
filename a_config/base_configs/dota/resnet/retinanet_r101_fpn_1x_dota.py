_base_ = './retinanet_r50_fpn_1x_dota.py'
#model = dict(pretrained='/workspace/mmdet2.0/pretained/resnet101-5d3b4d8f.pth', backbone=dict(depth=101))
model = dict(pretrained='/home/airstudio/pretrained model/resnet101-5d3b4d8f.pth', backbone=dict(depth=101))