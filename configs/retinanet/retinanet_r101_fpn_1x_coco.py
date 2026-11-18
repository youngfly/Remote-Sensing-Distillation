_base_ = './retinanet_r50_fpn_1x_nwpu.py'
#model = dict(pretrained='/workspace/mmdet2.0/pretained/resnet101-5d3b4d8f.pth', backbone=dict(depth=101))
model = dict(pretrained='/yyr/pretained/resnet101-5d3b4d8f.pth', backbone=dict(depth=101))
