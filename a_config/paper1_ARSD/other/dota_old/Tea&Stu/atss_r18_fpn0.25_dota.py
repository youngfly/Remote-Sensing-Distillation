_base_ = './atss_r18_fpn0.5_dota.py'
model = dict(neck=dict(out_channels=64), bbox_head=dict(in_channels=64, feat_channels=64,))