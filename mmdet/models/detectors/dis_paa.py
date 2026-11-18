import torch
from mmdet.models import build_detector
from mmdet.models import build_loss
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmdet.core import bbox_overlaps
from ..builder import DETECTORS, LOSSES
from .paa import PAA
import cv2
import matplotlib.pyplot as plt
import matplotlib
from mmcv.cnn import normal_init
import torch.nn as nn
import mmcv
import numpy as np
import os

EPS = 1e-12
try:
    import sklearn.mixture as skm
except ImportError:
    skm = None

def levels_to_images(mlvl_tensor):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)

    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]

@DETECTORS.register_module()
class Distill_PAA(PAA):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 tea_model,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 tea_pretrained=None,
                 ):
        super(Distill_PAA, self).__init__(backbone, neck, bbox_head, train_cfg,
                                          test_cfg, pretrained)
        self.tea_model = build_detector(tea_model, train_cfg=train_cfg, test_cfg=test_cfg)
        self.load_weights(pretrained=tea_pretrained)
        self.freeze_models()
        # todo 采用giou作为回归的损失函数
        loss_dis_reg_cfg = dict(type='NIoULoss', loss_weight=2.0)
        self.loss_distill_reg = build_loss(loss_dis_reg_cfg)
        # todo 针对学生的fpn的特征层的尺度变为原来的一半，此处从configs获取学生fpn通道数
        self.in_channels_adapt = bbox_head.in_channels
        self.out_channels = 256
        self.adapt_conv = nn.Conv2d(self.in_channels_adapt, self.out_channels, 1, padding=0)
        # todo 初始化适应层的权重
        self._init_adapt_weights()

        # todo 是否关系
        self.if_cat = True
        self.cat_wei = 0.01
        self.if_geo = False
        self.cos_wei = 0.1
        self.dia_wei = 0.0001

        # todo 是否响应
        self.if_fea = False
        self.fea_wei = 1
        self.if_cls = False
        self.cls_wei = 1
        self.if_reg = False
        self.reg_wei = 0.1

        # todo 延迟蒸馏
        self.if_delay = True
        # todo dior训练集的图片数
        self.num_images_dior = 11725
        self.num_images_voc = 16552
        self.num_images_nwpu = 520
        # todo 当前的iter以及epoch数
        self.iter = 0
        self.epoch = 0
        # todo 从第 n+1 个的epoch开始蒸馏
        self.ss_epoch = 10

    def _init_adapt_weights(self):
        normal_init(self.adapt_conv, std=0.01)

    def freeze_models(self):

        self.tea_model.eval()
        for param in self.tea_model.parameters():
            param.requires_grad = False

    def load_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self.tea_model, pretrained, strict=False, logger=logger)
            print("load teacher model success")
        else:
            raise TypeError('pretrained must be a str')

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        # todo 获取教师网络的输出
        self.tea_model.eval()
        with torch.no_grad():
            tea_x = self.tea_model.extract_feat(img)  # 教师网络的输出
            tea_outs = self.tea_model.bbox_head(tea_x)  # tea_outs[0]分类 tea_outs[1]回归
            tea_loss_inputs = tea_outs + (gt_bboxes, gt_labels, img_metas)  # 获取教师网络的输入，方便后面的获取最后的输出

        # todo 获取学生的网络的输出
        stu_x = self.extract_feat(img)
        stu_outs = self.bbox_head(stu_x)

        loss_inputs = stu_outs + (gt_bboxes, gt_labels, img_metas,)
        losses, pos_ind_flatten = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # todo 采用延迟监督策略
        # todo 获取 batchsize 计算当前的iter轮数， 以及epoch数,
        # todo 全时域监督, 默认为Ture
        flag = True
        if self.if_delay:
            batch_nums = len(gt_bboxes)
            self.iter += 1
            # todo num_images = ['dior', 'voc' , 'nwpu']
            self.epoch = int((self.iter * batch_nums) / self.num_images_nwpu) + 1
            flag = self.epoch > self.ss_epoch

        # todo 导出结果为了教师和学生的结果对齐，导出学生的pos_inds
        pos_cls_stu, pos_reg_stu, pos_labels, iou_bbox, \
        pos_index, pos_gt, pos_inds_flatten, \
        cls_loss_stu, reg_loss_stu = self.bbox_head.loss_fea(*loss_inputs, pos_ind_flatten,
                                                             gt_bboxes_ignore=gt_bboxes_ignore)

        # todo 导出教师组的分类和回归结果平滑组
        pos_cls_tea, pos_reg_tea, \
        cls_loss_tea, reg_loss_tea = self.bbox_head.loss_tea(*tea_loss_inputs, pos_inds_flatten, pos_labels,
                                                             gt_bboxes_ignore=gt_bboxes_ignore)

        # todo 样本集A
        mask_A = cls_loss_tea < cls_loss_stu
        mask_A = mask_A+0
        mask_B = reg_loss_tea < reg_loss_stu
        mask_B = mask_B+0

        # todo 获取教师和学生特征层的信息，展平方便提取关键信息
        ft = levels_to_images(tea_x)
        ft = torch.cat(ft, 0).view(-1, ft[0].size(-1))
        fs = levels_to_images(stu_x)
        fs = torch.cat(fs, 0).view(-1, fs[0].size(-1))
        # todo 获取正样本
        ft = ft[pos_inds_flatten]
        fs = fs[pos_inds_flatten]

        # todo 隐式
        if self.if_cat and flag:
            sup_cate = self.dis_cate(mask_A, pos_cls_stu, pos_labels, pos_index, pos_gt, pos_cls_tea)
            sup_cate = sup_cate * self.cat_wei
            sup_loss_cls = torch.clamp(sup_cate, 0, 10)
            losses.update(dict(loss_cate=sup_loss_cls))
        if self.if_geo and flag:
            sup_geo_cos, sup_geo_dia = self.dis_geo(mask_B, pos_reg_stu, pos_reg_tea, pos_index)
            sup_geo_cos = sup_geo_cos * self.cos_wei
            sup_geo_dia = sup_geo_dia * self.dia_wei
            sup_geo = sup_geo_cos + sup_geo_dia
            sup_loss_reg = torch.clamp(sup_geo, 0, 10)
            losses.update(dict(loss_geo=sup_loss_reg))

        # todo 显示
        if self.if_fea and flag:
            sup_fea = self.dis_fea(ft, fs)
            sup_fea = sup_fea * self.fea_wei
            sup_fea = torch.clamp(sup_fea, 0, 10)
            losses.update(dict(loss_sup_fea=sup_fea))
        if self.if_cls and flag:
            sup_cls = self.dis_cls(mask_A, pos_cls_stu, pos_cls_tea)
            sup_cls = sup_cls * self.cls_wei
            sup_cls = torch.clamp(sup_cls, 0, 10)
            losses.update(dict(loss_sup_cls=sup_cls))
        if self.if_reg and flag:
            sup_reg = self.dis_reg(mask_B, pos_reg_stu, pos_reg_tea)
            sup_reg = sup_reg * self.reg_wei
            sup_reg = torch.clamp(sup_reg, 0, 10)
            losses.update(dict(loss_sup_reg=sup_reg))

        # todo ************************绘图函数*******************************
        # todo 可视化展示不同的mask的结果, 需要选择path, 以及不同的mask_batch
        # path = "/home/airstudio/code/yyr/masks/mask0/"
        # path1 = "/home/airstudio/code/yyr/masks/mask1/"
        # path2 = "/home/airstudio/code/yyr/masks/mask2/"
        # path3 = "/home/airstudio/code/yyr/masks/mask3/"
        # path4 = "/home/airstudio/code/yyr/masks/mask4/"

        # todo 热力图区域 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # todo 热力图的存储路径
        # path_tea_fpn = "/home/airstudio/code/yyr/heatmap/tea_fpn/"
        # path_stu_fpn = "/home/airstudio/code/yyr/heatmap/stu_fpn/"

        # todo 可视化热力图
        # self.show_heatmap(img_metas, tea_x, path_tea_fpn)
        # todo 同时绘制教师和学生的热力图
        # self.show_heatmap_all(img_metas, stu_x, path_stu_fpn, tea_x, path_tea_fpn)

        # todo 为每一个图片的存储一个文件夹
        # self.heatmap_level_t(img_metas, tea_x, path_tea_fpn)
        # self.heatmap_level_all(img_metas, stu_x, path_stu_fpn, tea_x, path_tea_fpn)

        # todo 检测框区域 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # todo 将带有检测框的结果可视化
        # self.show_bboxes(gt_bboxes, img_metas, path4, thickness=3)
        # todo ************************绘图函数*******************************

        return losses

    # todo rela
    def dis_cate(self, mask_A, pos_cls_stu, pos_labels, pos_index, pos_gt, pos_cls_tea):
        # todo pos_index: 每个样本属于的哪一张图片上的结果
        # todo pos_gt: 每一个样本的分配的gt
        # todo pos_labels： 每一个样本的对应的标签
        sup_loss = 0.0
        for i in range(len(pos_cls_stu)):
            # todo 计算每一个样本与其他所有的样本之间的关系
            # todo 计算相同图片上的结果
            batch = (pos_index == pos_index[i]).cuda()
            batch = batch + 1
            mask = batch * mask_A
            s_feats = pos_cls_stu[i].repeat(len(pos_cls_stu), 1)
            loss_stu = torch.sqrt((torch.pow(s_feats - pos_cls_stu, 2) * mask.unsqueeze(1)).sum())
            t_feats = pos_cls_tea[i].repeat(len(pos_cls_tea), 1)
            loss_tea = torch.sqrt((torch.pow(t_feats - pos_cls_tea, 2) * mask.unsqueeze(1)).sum())
            # todo 计算 smooth L1 loss
            loss = self.sml1(loss_stu, loss_tea)
            loss_norm = loss/len(pos_cls_stu)
            sup_loss = sup_loss + loss_norm
        return sup_loss

    def dis_geo(self, mask_B, pos_reg_stu, pos_reg_tea, pos_index):
        sup_loss_cos = 0.0
        sup_loss_dia = 0.0
        # todo 计算每个样本的蕴含的几何知识
        s_cos, s_dia = self.geo_kn(pos_reg_stu)
        t_cos, t_dia = self.geo_kn(pos_reg_tea)
        # todo
        for i in range(len(s_cos)):
            # todo 计算每一个样本与其他所有的样本之间的关系
            # todo 获取一张图片的里面的样本
            batch = (pos_index == pos_index[i]).cuda()
            batch = batch + 1
            mask = mask_B * batch
            # todo 计算余弦相似度损失
            s_feats = s_cos[i].repeat(len(s_cos), 1)
            loss_stu_cos = torch.sqrt((torch.pow(s_feats - s_cos.unsqueeze(1), 2) * mask.unsqueeze(1)).sum())
            t_feats = t_cos[i].repeat(len(t_cos), 1)
            loss_tea_cos = torch.sqrt((torch.pow(t_feats - t_cos.unsqueeze(1), 2) * mask.unsqueeze(1)).sum())
            loss_cos = self.sml1(loss_stu_cos, loss_tea_cos)
            loss_cos_norm = loss_cos / len(s_cos)
            sup_loss_cos = sup_loss_cos + loss_cos_norm
            # todo 计算对角相似度损失
            s_f = s_dia[i].repeat(len(s_dia), 1)
            loss_stu_dia = torch.sqrt((torch.pow(s_f - s_dia.unsqueeze(1), 2) * mask_B.unsqueeze(1)).sum())
            t_f = t_dia[i].repeat(len(t_dia), 1)
            loss_tea_dia = torch.sqrt((torch.pow(t_f - t_dia.unsqueeze(1), 2) * mask_B.unsqueeze(1)).sum())
            loss_dia = self.sml1(loss_stu_dia, loss_tea_dia)
            loss_dia_norm = loss_dia/ len(s_dia)
            sup_loss_dia = sup_loss_dia + loss_dia_norm
        return sup_loss_cos, sup_loss_dia

    def dis_cls(self, mask_A, pos_cls_stu, pos_cls_tea):
        loss_cls = torch.sqrt((torch.pow(pos_cls_stu - pos_cls_tea, 2) * mask_A.unsqueeze(1)).sum())
        sup_loss = loss_cls/len(pos_cls_stu)
        return sup_loss

    def dis_reg(self, mask_B, pos_reg_stu, pos_reg_tea):
        loss_reg = self.loss_distill_reg(pos_reg_tea, pos_reg_stu, weight=mask_B, avg_factor=1.0)
        sup_loss2 = loss_reg/len(pos_reg_stu)
        return sup_loss2

    def dis_fea(self, ft, fs):
        fea_expand = self.adapt_fc(fs, fs.size(1), ft.size(1))
        loss_fea = torch.sqrt((torch.pow(ft - fea_expand, 2)).sum())
        sup_loss = loss_fea / len(ft)
        return sup_loss

    # todo smooth L1oss
    def sml1(self, a, b):
        x = a - b
        if x.abs() < 1:
            y = 0.5 * x * x
        else:
            y = x.abs() - 0.5
        return y

    # todo 几何知识提炼
    def geo_kn(self, bb):
        s_w = bb[:, 2] - bb[:, 0]
        # todo 计算样本的高
        s_h = bb[:, 3] - bb[:, 1]
        # todo 计算对角线距离
        # s_dia = torch.sqrt()
        s_w2 = s_w ** 2
        s_h2 = s_h ** 2
        s_dia = torch.sqrt(s_w2 + s_h2)
        # todo 计算余弦
        s_cos = s_w / s_dia
        return s_cos, s_dia

    # todo 适应层的定义**************************************************************************
    # todo 通道转换函数,适用于有长宽的特征图进行通道转换
    def adapt_conv(self, feature, in_channel, out_channel):
        # todo 定义通道函数
        a_conv = nn.Conv2d(in_channel, out_channel, 1, padding=0).cuda()
        normal_init(a_conv, std=0.01)
        adapt_new = a_conv(feature)
        return adapt_new

    # todo 通道转换函数，单维度的特征进行通道转换
    def adapt_fc(self, feature, in_channel, out_channel):
        # todo 定义通道函数
        a_fc = nn.Linear(in_channel, out_channel, bias=True).cuda()
        adapt_new = a_fc(feature)
        return adapt_new

    # todo 小工具**************************************************************
    # todo 生成文件夹
    def mkdir(self, path):
        isexists = os.path.exists(path)
        if not isexists:
            os.makedirs(path)
            print("file makes successfully")
        else:
            print("file is exist")

    # todo 适应层的网络函数
    def adapt(self, x):
        stu_adapt = []
        for i in range(len(x)):
            stu_ada = []
            stu_ada = self.adapt_conv(x[i])
            stu_adapt.append(stu_ada)
        return stu_adapt
