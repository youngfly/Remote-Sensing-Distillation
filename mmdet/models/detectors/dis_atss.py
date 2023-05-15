import torch
from mmdet.models import build_detector
from mmdet.models import build_loss
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmdet.core import bbox_overlaps
from ..builder import DETECTORS, LOSSES
from .atss import ATSS
import cv2
import matplotlib.pyplot as plt
import matplotlib
from mmcv.cnn import normal_init
import torch.nn as nn
import torch.nn.functional as F
import mmcv
import numpy as np
import os

eps = 1e-12

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
class Dis_ATSS(ATSS):
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
        super(Dis_ATSS, self).__init__(backbone, neck, bbox_head, train_cfg,
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
        self.if_cat = False
        self.cat_wei = 1
        self.if_geo = False
        self.cos_wei = 1.0
        self.dia_wei = 0.001
        self.cen_wei = 0.001
        self.gro_wei = 1

        # todo 是否响应
        self.if_fea = False
        self.fea_wei = 1.0
        self.if_cls = False
        self.cls_wei = 1.0
        self.if_reg = False
        self.reg_wei = 1.0

        # todo 延迟蒸馏
        # todo dior训练集的图片数
        self.num_images_dior = 11725
        self.num_images_voc = 16552
        self.num_images_nwpu = 520
        self.num_images_dota = 15749
        # todo 当前的iter以及epoch数
        self.iter = 0
        self.epoch = 0
        # todo 延迟蒸馏， 延迟轮数， 延迟数据集
        self.if_delay = True
        self.ss_epoch = 12
        self.images = self.num_images_dota

        # todo 初始化适应层的权重
        self._init_adapt_weights()

        print('the delay distillation is', self.if_delay, ' + ', 'the epoch is', self.ss_epoch)
        # print('the mode is', self.if_fea,)

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
            tea_x = self.tea_model.extract_feat(img)
            tea_outs = self.tea_model.bbox_head(tea_x)
            tea_loss_inputs = tea_outs + (gt_bboxes, gt_labels, img_metas)

        # todo 获取学生的网络的输出
        stu_x = self.extract_feat(img)
        stu_outs = self.bbox_head(stu_x)

        # todo stu_x=[5,[n,c,h,w]]
        # todo stu_out=[3,[5,[n,c,h,w]]]
        # todo stu_out[0]分类， stu_outs[1]回归, stu_outs[2]centness
        # todo tupe (3, 5, n, c, h ,w), 3代表三个输出的分支头， 5代表5个尺度，n代表batch_size,
        # todo 分类头的c=15, 代表类别， 检测头c=4, 代表4个偏移量， 点头c=1代表， 代表一个中心点
        # todo [h,w] 128, 64, 32 ,16, 8

        loss_inputs = stu_outs + (gt_bboxes, gt_labels, img_metas,)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # todo 采用延迟监督策略
        # todo 获取 batchsize 计算当前的iter轮数， 以及epoch数,
        # todo 全时域监督, 默认为Ture
        flag = True
        if self.if_delay:
            batch_nums = len(gt_bboxes)
            self.iter += 1
            # todo num_images = ['dior', 'voc' , 'nwpu', 'dota']
            self.epoch = int((self.iter * batch_nums) / self.images) + 1
            flag = self.epoch > self.ss_epoch

        # todo 导出新的结果
        pos_cls_stu, pos_reg_stu, pos_labels, \
        pos_index, pos_gt, pos_inds_flatten, \
        cls_loss_stu, reg_loss_stu, \
        pos_inds_flatten_old, \
        pos_bbox_targets = self.bbox_head.loss_stu(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # todo 导出教师组的分类和回归结果平滑组
        pos_cls_tea, pos_reg_tea, \
        cls_loss_tea, reg_loss_tea = self.bbox_head.loss_tea(*tea_loss_inputs, pos_inds_flatten,
                                                             pos_labels, pos_bbox_targets,
                                                             gt_bboxes_ignore=gt_bboxes_ignore)
        # todo 统计样本选择
        # with open("number.txt", "w") as f:
        #     f.write(str((len(pos_inds_flatten)/len(pos_inds_flatten_old)))+ "\n")
        print(len(pos_inds_flatten)/len(pos_inds_flatten_old))

        # todo code检测，保证教师和学生的预测可以对齐, 以及有结果
        assert len(pos_reg_tea) == len(pos_reg_stu)
        assert len(pos_cls_tea) == len(pos_cls_stu)
        assert len(pos_reg_stu) > 0
        assert len(pos_cls_stu) > 0

        # todo 样本集A 和 B
        mask_A = cls_loss_tea < cls_loss_stu
        mask_A = mask_A + 0
        mask_B = reg_loss_tea < reg_loss_stu
        mask_B = mask_B + 0

        # todo 学生的特征进行提前通道变换
        stu_adapt = self.adapt_old(stu_x)

        # todo 特征层的样本选取*********************************
        # todo 获取教师和学生特征层的信息，展平方便提取关键信息
        ft = levels_to_images(tea_x)
        ft_flatten = torch.cat(ft, 0).view(-1, ft[0].size(-1))
        fs = levels_to_images(stu_x)
        fs_flatten = torch.cat(fs, 0).view(-1, fs[0].size(-1))
        fs_adapt = levels_to_images(stu_adapt)
        fs_adapt_fla = torch.cat(fs_adapt, 0).view(-1, ft[0].size(-1))

        # todo 获取正样本
        ft_s3 = ft_flatten[pos_inds_flatten]
        fs_s3 = fs_flatten[pos_inds_flatten]
        fs_ad_s3 = fs_adapt_fla[pos_inds_flatten]

        # todo 全局正样本
        ft_all = ft_flatten[pos_inds_flatten_old]
        fs_all = fs_flatten[pos_inds_flatten_old]
        fs_ad_all = fs_adapt_fla[pos_inds_flatten_old]

        # todo 分类结果的样本的选取****************************
        cls_t = levels_to_images(tea_outs[0])
        cls_t_flatten = torch.cat(cls_t, 0).view(-1, cls_t[0].size(-1))
        cls_s = levels_to_images(stu_outs[0])
        cls_s_flatten = torch.cat(cls_s, 0).view(-1, cls_s[0].size(-1))

        # todo 全局正样本
        cls_t_all = cls_t_flatten[pos_inds_flatten_old]
        cls_s_all = cls_s_flatten[pos_inds_flatten_old]

        # todo 回归的样本的选取******************************
        reg_t = levels_to_images(tea_outs[1])
        reg_t_flatten = torch.cat(reg_t, 0).view(-1, reg_t[0].size(-1))
        reg_s = levels_to_images(stu_outs[1])
        reg_s_flatten = torch.cat(reg_s, 0).view(-1, reg_s[0].size(-1))
        # todo 全局正样本
        reg_t_all = reg_t_flatten[pos_inds_flatten_old]
        reg_s_all = reg_s_flatten[pos_inds_flatten_old]

        # todo 为全局生成的权重矩阵
        mask_all = torch.ones([len(reg_t_all)]).cuda()

        # todo 获取mask4，并展平化
        mask_batch = self.get_masks_4(stu_x, gt_bboxes, img_metas, img)
        mask_flatten_all = torch.tensor([]).cuda()
        for i in range (len(mask_batch)):
            for j in range(len(mask_batch[i])):
                mask_flatten = mask_batch[i][j].view(-1, mask_batch[i][j].size(0) * mask_batch[i][j].size(1)).float().cuda(0)
                mask_flatten_all = torch.cat((mask_flatten_all, mask_flatten), 1)
        mask_flatten_all = mask_flatten_all.view(mask_flatten_all.size(-1) ,-1)
        mask_flatten_pos = mask_flatten_all[pos_inds_flatten]

        # todo 隐式
        if self.if_cat and flag:
            sup_cate = self.dis_cate4(mask_A, pos_cls_stu, pos_labels, pos_index, pos_gt, pos_cls_tea)
            # sup_cate = self.dis_cate3(mask_A, cls_loss_stu, pos_labels, pos_index, pos_gt, cls_loss_tea)
            sup_cate = sup_cate * self.cat_wei
            # print(sup_cate)
            sup_cate = torch.clamp(sup_cate, 0, 10)
            losses.update(dict(loss_cate=sup_cate))
        if self.if_geo and flag:
            # sup_geo_cos, sup_geo_dia = self.dis_geo(mask_B, pos_reg_stu, pos_reg_tea, pos_index, pos_gt, pos_bbox_targets)
            # sup_geo_cos = sup_geo_cos * self.cos_wei
            # sup_geo_dia = sup_geo_dia * self.dia_wei
            # sup_geo = sup_geo_cos + sup_geo_dia
            # sup_loss_geo = torch.clamp(sup_geo, 0.01, 10)
            # losses.update(dict(loss_geo=sup_loss_geo))

            sup_geo_gro = self.dis_geo3(mask_B, reg_loss_stu, pos_index, reg_loss_tea)
            # print(sup_geo_gro)
            sup_geo_gro = sup_geo_gro * self.gro_wei
            sup_geo_gro = torch.clamp(sup_geo_gro, 0, 10)
            losses.update(dict(loss_geo=sup_geo_gro))

        # todo 显示
        if self.if_fea:
            # todo s3正样本
            # sup_fea = self.dis_fea(ft_s3, fs_s3)
            # todo 全局正样本
            # sup_fea = self.dis_fea(ft_all, fs_all)
            # todo ad的全局正样本
            # sup_fea = self.dis_fea_ada(ft_all, fs_ad_all)
            # todo ad的s3正样本
            sup_fea = self.dis_fea_ada(ft_s3, fs_ad_s3)

            # todo 采用gt的s3正样本
            # sup_fea = self.dis_fea_ada_gt(ft_s3, fs_ad_s3, mask_flatten_pos)

            sup_fea = sup_fea * self.fea_wei
            sup_fea = torch.clamp(sup_fea, 0, 10)
            losses.update(dict(loss_sup_fea=sup_fea))
        if self.if_cls:
            # todo s3正样本
            # sup_cls = self.dis_cls(mask_A, pos_cls_stu, pos_cls_tea)
            # todo 全局正样本
            sup_cls = self.dis_cls(mask_all, cls_t_all, cls_s_all)

            sup_cls = sup_cls * self.cls_wei
            sup_cls = torch.clamp(sup_cls, 0, 10)
            losses.update(dict(loss_sup_cls=sup_cls))
        if self.if_reg :
            sup_reg = self.dis_reg(mask_B, pos_reg_stu, pos_reg_tea)
            sup_reg = sup_reg * self.reg_wei
            sup_reg = torch.clamp(sup_reg, 0, 10)
            losses.update(dict(loss_sup_reg=sup_reg))

        # todo ************************绘图函数*******************************
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
        # self.show_bboxes(gt_bboxes, img_metas, path_stu_fpn, thickness=3)
        # todo ************************绘图函数*******************************

        return losses

    # todo exp
    # todo cate
    # todo 最早的错误版本
    def dis_cate(self, mask_A, pos_cls_stu, pos_labels, pos_index, pos_gt, pos_cls_tea):
        # todo pos_index: 每个样本属于的哪一张图片上的结果
        # todo pos_gt: 每一个样本的分配的gt
        # todo pos_labels： 每一个样本的对应的标签
        sup_loss = 0.0
        if not len(pos_cls_stu):
            return torch.tensor([0.0]).cuda()
        pos_cls_stu.softmax()
        pos_cls_tea.softmax()
        for i in range(len(pos_cls_stu)):
            # todo 计算每一个样本与其他所有的样本之间的关系
            # todo 计算相同图片上的结果
            batch = (pos_index == pos_index[i]).cuda()
            batch = batch + 0
            mask = batch * mask_A
            s_feats = pos_cls_stu[i].repeat(len(pos_cls_stu), 1)
            loss_stu = torch.sqrt((torch.pow(s_feats - pos_cls_stu, 2) * mask.unsqueeze(1)).sum())
            t_feats = pos_cls_tea[i].repeat(len(pos_cls_tea), 1)
            loss_tea = torch.sqrt((torch.pow(t_feats - pos_cls_tea, 2) * mask.unsqueeze(1)).sum())

            # todo 计算 smooth L1 loss
            loss = self.sml1(loss_stu, loss_tea)
            loss_norm = loss / len(pos_cls_stu)
            sup_loss = sup_loss + loss_norm
        if torch.isnan(sup_loss):
            sup_loss = torch.tensor([0.0]).cuda()
        return sup_loss

    # todo 自己构建的距离计算矩阵
    def dis_cate2(self, mask_A, pos_cls_stu, pos_labels, pos_index, pos_gt, pos_cls_tea):
        sup_loss = 0.0
        # todo 正样本的batch数
        nums = pos_index.max()
        # todo 过滤所有的正样本
        batch = []
        for j in range(nums):
            images_per_batch = (pos_index == (j + 1)).cuda()
            batch.append(images_per_batch)
        # todo 计算一个批次图片所有样本的之间的关系
        for i in range(len(batch)):
            batch_pos_cls_stu = pos_cls_stu[batch[i]]
            target_batch_stu_single = batch_pos_cls_stu.unsqueeze(0)
            target_batch_stu = target_batch_stu_single.repeat((len(batch_pos_cls_stu), 1, 1), 0)
            cate_stu = torch.sqrt((torch.pow(batch_pos_cls_stu.unsqueeze(1) - target_batch_stu, 2))).sum(-1)
            stu_norm = cate_stu.sum()
            cate_stu_norm = cate_stu / (stu_norm + 1e-6)

            batch_pos_cls_tea = pos_cls_tea[batch[i]]
            target_batch_tea_single = batch_pos_cls_tea.unsqueeze(0)
            target_batch_tea = target_batch_tea_single.repeat((len(batch_pos_cls_tea), 1, 1), 0)
            cate_tea = torch.sqrt((torch.pow(batch_pos_cls_tea.unsqueeze(1) - target_batch_tea, 2))).sum(-1)
            tea_norm = 1 / 2 * (cate_tea ** 2).sum()
            cate_tea_norm = cate_tea / (tea_norm + 1e-6)

            cate_all = F.smooth_l1_loss(cate_tea_norm, cate_stu, reduction='mean')
            cate_all = cate_tea_norm.sum()

            sup_loss = sup_loss + cate_all
        if torch.isnan(sup_loss):
            sup_loss = torch.tensor([0.0]).cuda()

        return sup_loss

    # todo 其实是对对应的cls_loss 按照批次计算sml1
    def dis_cate3(self, mask_A, cls_loss_stu, pos_labels, pos_index, pos_gt, cls_loss_tea):
        # todo pos_index: 每个样本属于的哪一张图片上的结果
        # todo pos_gt: 每一个样本的分配的gt
        # todo pos_labels： 每一个样本的对应的标签
        sup_loss_all = 0.0
        # todo 正样本的batch数
        nums = pos_index.max()
        # todo 过滤所有的正样本,
        batch = []
        mask = []
        for j in range(nums):
            # todo 找到属于一张图片的所属的群组
            images_per_batch = (pos_index == (j + 1)).cuda()
            images_per_mask = mask_A[images_per_batch].cuda()
            batch.append(images_per_batch)
            mask.append(images_per_mask)
        for i in range(len(batch)):
            batch_pos_cls_stu = cls_loss_stu[batch[i]]
            batch_pos_cls_tea = cls_loss_tea[batch[i]]
            sup_loss = F.smooth_l1_loss(batch_pos_cls_stu, batch_pos_cls_tea, reduction='none')
            sup_loss = (sup_loss * mask[i]).sum() / len(batch[i])
            sup_loss_all = sup_loss_all + sup_loss
        sup_loss_all = sup_loss_all / len(batch)
        return sup_loss_all

    # todo 根据的RKD改造的距离计算矩阵
    def dis_cate4(self, mask_A, pos_cls_stu, pos_labels, pos_index, pos_gt, pos_cls_tea):
        # todo pos_index: 每个样本属于的哪一张图片上的结果
        # todo pos_gt: 每一个样本的分配的gt
        # todo pos_labels： 每一个样本的对应的标签
        sup_loss_all = 0.0
        # todo 正样本的batch数
        nums = pos_index.max()
        # todo 过滤所有的正样本,
        batch = []
        mask = []
        for j in range(nums):
            # todo 找到属于一张图片的所属的群组
            images_per_batch = (pos_index == (j + 1)).cuda()
            images_per_mask = mask_A[images_per_batch].cuda()
            batch.append(images_per_batch)
            mask.append(images_per_mask)
        for i in range(len(batch)):
            batch_pos_cls_stu = pos_cls_stu[batch[i]]
            s_d = self.pdist(batch_pos_cls_stu, squared=False)
            mean_sd = s_d[s_d > 0].mean()
            s_d = s_d / mean_sd

            batch_pos_cls_tea = pos_cls_tea[batch[i]]
            t_d = self.pdist(batch_pos_cls_tea, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td
            loss = F.smooth_l1_loss(s_d, t_d, reduction='elementwise_mean')
            sup_loss_all = sup_loss_all + loss
        sup_loss_all = sup_loss_all / 4
        if torch.isnan(sup_loss_all):
            sup_loss_all = torch.tensor([0.0]).cuda()
        return sup_loss_all

    # todo geo
    # todo 计算每个边框的角度以及的对角距离
    def dis_geo(self, mask_B, pos_reg_stu, pos_reg_tea, pos_index, pos_gt, pos_bbox_targets):
        sup_loss_cos = 0.0
        sup_loss_dia = 0.0
        # todo 计算每个样本的蕴含的几何知识
        s_cos, s_dia = self.geo_kn(pos_reg_stu)
        t_cos, t_dia = self.geo_kn(pos_reg_tea)
        for i in range(len(s_cos)):
            # todo 计算每一个样本与其他所有的样本之间的关系
            # todo 获取一张图片的里面的样本
            batch = (pos_index == pos_index[i]).cuda()
            batch = batch + 0
            mask = batch
            # mask = mask_B * batch

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
            loss_stu_dia = torch.sqrt((torch.pow(s_f - s_dia.unsqueeze(1), 2) * mask.unsqueeze(1)).sum())
            t_f = t_dia[i].repeat(len(t_dia), 1)
            loss_tea_dia = torch.sqrt((torch.pow(t_f - t_dia.unsqueeze(1), 2) * mask.unsqueeze(1)).sum())
            loss_dia = self.sml1(loss_stu_dia, loss_tea_dia)
            loss_dia_norm = loss_dia / len(s_dia)
            sup_loss_dia = sup_loss_dia + loss_dia_norm

        if torch.isnan(sup_loss_cos):
            sup_loss_cos = torch.tensor([0.0]).cuda()
        if torch.isnan(sup_loss_dia):
            sup_loss_dia = torch.tensor([0.0]).cuda()
        return sup_loss_cos, sup_loss_dia

    # todo 计算中心点之间的所在距离
    def dis_geo2(self, mask_B, pos_reg_stu, pos_reg_tea, pos_index):
        sup_loss_cen = 0.0
        # todo 计算每个样本的中心点
        s_center = self.geo_cen(pos_reg_stu)
        t_center = self.geo_cen(pos_reg_tea)
        for i in range(len(s_center)):
            # todo 筛选出当前图片的样本
            batch = (pos_index == pos_index[i]).cuda()
            batch = batch + 0
            # todo 值计算教师样本的中质量好于学生
            # mask = mask_B * batch
            mask = batch
            # todo 计算空间位置距离损失
            s_feats = s_center[i].repeat(len(s_center), 1)
            s_dis = torch.sqrt((s_center[:, 0] - s_feats[:, 0]) ** 2 + (s_center[:, 1] - s_feats[:, 1]) ** 2)
            s_dis = s_dis / s_dis.max()
            loss_stu_cen = (s_dis * mask).sum()

            t_feats = t_center[i].repeat(len(t_center), 1)
            t_dis = torch.sqrt((t_center[:, 0] - t_feats[:, 0]) ** 2 + (t_center[:, 1] - t_feats[:, 1]) ** 2)
            t_dis = t_dis / t_dis.max()
            loss_tea_cen = (t_dis * mask).sum()

            loss_cen = self.sml1(loss_stu_cen, loss_tea_cen)
            loss_cen_norm = loss_cen / len(s_center)
            sup_loss_cen = sup_loss_cen + loss_cen_norm
        if torch.isnan(sup_loss_cen):
            sup_loss_cen = torch.tensor([0.0]).cuda()
        print(sup_loss_cen)
        return sup_loss_cen

    # todo 计算对应的reg_loss之间的sml1
    def dis_geo3(self, mask_B, reg_loss_stu, pos_index, reg_loss_tea):
        sup_loss_all = 0.0
        # todo 正样本的batch数
        nums = pos_index.max()
        # todo 过滤所有的正样本,
        batch = []
        mask = []
        for j in range(nums):
            # todo 找到属于一张图片的所属的群组
            images_per_batch = (pos_index == (j + 1)).cuda()
            images_per_mask = mask_B[images_per_batch].cuda()
            batch.append(images_per_batch)
            mask.append(images_per_mask)
        for i in range(len(batch)):
            batch_pos_reg_stu = reg_loss_stu[batch[i]]
            batch_pos_reg_tea = reg_loss_tea[batch[i]]
            sup_loss = F.smooth_l1_loss(batch_pos_reg_stu, batch_pos_reg_tea, reduction='none')
            sup_loss = (sup_loss * mask[i]).sum() / len(batch[i])
            sup_loss = (sup_loss * mask[i]).sum()
            sup_loss_all = sup_loss_all + sup_loss
        sup_loss_all = sup_loss_all / len(batch)
        return sup_loss_all

    # todo imp
    def dis_cls(self, mask_A, pos_cls_stu, pos_cls_tea):
        loss_cls = torch.sqrt((torch.pow(pos_cls_stu - pos_cls_tea, 2) * mask_A.unsqueeze(1)).sum())
        sup_loss = loss_cls / len(pos_cls_stu)
        return sup_loss

    def dis_reg(self, mask_B, pos_reg_stu, pos_reg_tea):
        loss_reg = self.loss_distill_reg(pos_reg_tea, pos_reg_stu, weight=mask_B, avg_factor=1.0)
        sup_loss2 = loss_reg / len(pos_reg_stu)
        return sup_loss2

    def dis_fea(self, ft, fs):
        fea_expand = self.adapt_fc(fs, fs.size(1), ft.size(1))
        loss_fea = torch.sqrt((torch.pow(ft - fea_expand, 2)).sum())
        sup_loss = loss_fea / len(ft)
        return

    def dis_fea_ada(self, ft, fs):
        loss_fea = torch.sqrt((torch.pow(ft - fs, 2)).sum())
        sup_loss = loss_fea / len(ft)
        return sup_loss

    def dis_fea_ada_gt(self, ft, fs, gtpos):
        loss_fea = torch.sqrt((torch.pow(ft - fs, 2) * gtpos.unsqueeze(1)).sum())
        sup_loss = loss_fea / len(ft)
        return sup_loss

    def get_masks_4(self, cls_scores, gt_bboxes, img_metas, img):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        # todo 获取一个批次图片的尺寸
        img_ori = img.size()[-2:]
        img_ori = torch.tensor([img_ori])
        # todo img_ori=[1024,1024]
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(featmap_sizes, img_metas, device=device)
        mask_batch = []

        batch_size = len(gt_bboxes)
        levels = len(anchor_list[0])
        for i in range(batch_size):
            mask_per_im = []
            # todo gt_bboxes=[gt_num,4]
            # todo 根据gt_box计算其面积
            area = torch.sqrt((gt_bboxes[i][:, 2] - gt_bboxes[i][:, 0]) * (gt_bboxes[i][:, 3] - gt_bboxes[i][:, 1]))
            # todo arae=[gt_num], 值=[0~1024]
            # print(area)
            # todo 计算原始图片的尺寸
            area_image = torch.sqrt((img_ori[:, 0] * img_ori[:, 1]).to(torch.float32)).cuda()
            # todo area_image=[1],值=[1024]
            # print(area_image)
            # todo 获得不同尺寸的gt应该分配给不同的层级,类似于fpn的操作
            target_lvls = torch.floor(4 + torch.log2(area / area_image + 1e-6)).cuda()
            # todo 将level限制5个fpn层级
            target_lvls = target_lvls.clamp(min=0, max=4).long()
            # todo target_lvls=[gt_num], 值=[0~4]
            # todo 根据真值的面积对mask的不同位置进行加权
            area_weight = torch.exp(-area / (area_image / 2)) + 1
            area_weight1 = torch.exp(-2 * area / (area_image / 2)) + 1
            area_weight2 = 2 - area / (area_image / 2)
            area_weight3 = 1.5 * torch.exp(-4 * area / area_image / 2) + 0.5
            for j in range(levels):
                height, width = featmap_sizes[j][0], featmap_sizes[j][1]
                mask_per_level = torch.zeros([height, width], dtype=torch.int64).cuda()
                # todo 获取每个特征图的尺寸
                fea_scale = featmap_sizes[j]
                # todo 将特征图尺寸转换为tensor
                fea_scale = torch.tensor([fea_scale])
                # todo fea_scale=[1,1] 值=[128,128],[64,64],[32,32],[16,16],[8,8]
                # todo 计算出每一层特征图相对于原始图片的缩放比率。
                # todo 对于dota刚好是长款等比，但是nwpu长宽比例不同，可能后面需要调整顺序
                percent = torch.div(fea_scale.float(), img_ori.float()).cuda()
                # todo percent=[1,1] 值=[0.125], [0.0625],[0.0312],[0.0156],[0.0078]
                # print(percent)
                # todo 构建按比例缩放后的gt坐标组
                gt_trans = torch.zeros(gt_bboxes[i].shape, dtype=torch.float32).cuda()
                gt_trans[:, :2] = gt_bboxes[i][:, :2] * percent
                gt_trans[:, 2:] = gt_bboxes[i][:, 2:] * percent
                # todo gt坐标放缩到对应的特征图上的位置
                # todo gt_trans=[gt_nums,4]
                # print(gt_trans.shape)
                # todo gt_mask尺寸为和gt_bboxes尺寸相同，gt_mask=[gt_num,4]
                gt_mask = torch.zeros(gt_bboxes[i].shape, dtype=torch.float32).cuda()
                # todo 根据当前图片里每所有的gt，生成对应当前层级的mask
                for k in range(gt_bboxes[i].shape[0]):
                    # todo 如果第k个gt分配的level层级等于当前的level层级，则将该层级的gt的gt_mask置为1
                    if target_lvls[k] == j:
                        gt_mask[k, :] = 1
                # todo 将mask和gt_trans相乘，得到对应的gt
                gt_trans = gt_trans * gt_mask
                # todo 根据不同的层级
                for k in range(gt_bboxes[i].shape[0]):
                    # todo 如果
                    if torch.sum(gt_bboxes[i][k]) == 0.:
                        break
                    # todo 为该层级生成特征图尺寸的mask
                    mask_per_gt = torch.zeros(height, width, dtype=torch.int64).cuda()
                    # # todo 根据真值的面积对mask的不同位置进行加权
                    # area_weight = torch.exp(-area / (area_image / 2)) + 1
                    # area_weight1 = torch.exp(-2 * area / (area_image / 2)) + 1
                    # area_weight2 = 2 - area / (area_image / 2)
                    # area_weight3 = 1.5 * torch.exp(-4 * area / area_image / 2) + 0.5
                    # todo 将mask 里面gt位置的值置为对应的加权权重
                    mask_per_gt[gt_trans[k][1].int():gt_trans[k][3].int(), gt_trans[k][0].int():gt_trans[k][2].int()] = \
                        area_weight[k]
                    # todo 将这个level里面所有的gt位置叠加起来，得到一个层级特征mask
                    mask_per_level += mask_per_gt
                # todo 得到一张图片的特征mask
                mask_per_im.append(mask_per_level)
            # todo 得到一个batch所有的mask
            mask_batch.append(mask_per_im)
        return mask_batch

    # todo distance 距离
    def pdist(self, e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            rest = res.sqrt()
        rest[range(len(e)), range(len(e))] = 0

        return res

    # todo 几何知识提炼
    def geo_kn(self, bb):
        s_w = (bb[:, 2] - bb[:, 0]) + 0.01
        # todo 计算样本的高
        s_h = (bb[:, 3] - bb[:, 1]) + 0.01
        # todo 计算对角线距离
        s_w2 = s_w ** 2
        s_h2 = s_h ** 2
        s_dia = torch.sqrt(s_w2 + s_h2) + 0.01
        # todo 计算余弦
        s_cos = s_w / s_dia
        return s_cos, s_dia

    def geo_cen(self, bb):
        cx = (bb[:, 2] + bb[:, 0]) / 2
        cy = (bb[:, 3] + bb[:, 1]) / 2
        center = torch.stack([cx, cy], dim=1)
        return center

    # todo 适应层的定义*********************************************************
    # todo 通道转换函数,适用于有长宽的特征图进行通道转换
    def adapt(self, feature, in_channel, out_channel):
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
    def adapt_old(self, x):
        stu_adapt = []
        for i in range(len(x)):
            stu_ada = []
            stu_ada = self.adapt_conv(x[i])
            # print(stu_ada.shape)
            stu_adapt.append(stu_ada)
        return stu_adapt

    # todo 绘图函数相关*********************************************************
    # todo 绘制带有bboxes的图片
    def show_bboxes(self,
                    bboxes,
                    img_metas,
                    path,
                    thickness=10):
        self.mkdir(path)
        # todo 选择你喜欢的颜色
        red = (0, 0, 255)
        green = (0, 255, 0)
        blue = (255, 0, 0)
        cyan = (255, 255, 0)
        yellow = (0, 255, 255)
        magenta = (255, 0, 255)
        white = (255, 255, 255)
        black = (0, 0, 0)
        chartreuse = (127, 255, 0)
        colors = chartreuse
        for i, img_meta in enumerate(img_metas):
            filename = img_meta['filename']
            path_pic = path + filename.split("/")[-1].split(".")[0]
            det_path = path_pic + "/det.png"
            flip = img_meta['flip']
            img = cv2.imread(filename)
            if flip:
                img = cv2.flip(img, 1)
            box = bboxes[i]
            for j in range(len(box)):
                left_top = (box[j, 0], box[j, 1])
                right_bottom = (box[j, 2], box[j, 3])
                cv2.rectangle(img, left_top, right_bottom, colors, thickness=thickness)
            cv2.imwrite(det_path, img)

    # todo 此函数调用了自己的重新设计的绘制函数，对应不同的类别拥有不同的颜色
    # todo mmcv.imshow_det_bboxes2
    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=10,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i]
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        mmcv.imshow_det_bboxes(
            img,
            bboxes,
            labels,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            thickness=thickness,
            font_scale=font_scale,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

    # todo 绘制为每一张图片生成一个文件夹
    def heatmap_level_t(self, img_metas, heatmaps, path):
        levels = len(heatmaps)
        for i, img_meta in enumerate(img_metas):
            filename = img_meta['filename']
            path_pic = path + filename.split("/")[-1].split(".")[0]
            self.mkdir(path_pic)
            flip = img_meta['flip']
            img = cv2.imread(filename)
            if flip:
                img = cv2.flip(img, 1)
            plt.imshow(img)
            plt.xticks(())
            plt.yticks(())
            plt.savefig(path_pic + "/image.png")
            for j in range(levels):
                heat = torch.sum(heatmaps[j][i], dim=0).cpu().numpy()
                plt.imshow(heat, interpolation='bilinear', cmap='jet')
                # cm 可选参数 cool hot, gray, cmap=plt.cm.cool。 没有的话默认是绿蓝的图
                plt.xticks(())
                plt.yticks(())
                path_level = path_pic + "/level" + str(j + 1) + ".png"
                plt.savefig(path_level)
                plt.close("all")

    def heatmap_level_s(self, img_metas, heatmaps, path):
        levels = len(heatmaps)
        for i, img_meta in enumerate(img_metas):
            filename = img_meta['filename']
            path_pic = path + filename.split("/")[-1].split(".")[0]
            self.mkdir(path_pic)
            flip = img_meta['flip']
            img = cv2.imread(filename)
            if flip:
                img = cv2.flip(img, 1)
            plt.imshow(img)
            plt.xticks(())
            plt.yticks(())
            plt.savefig(path_pic + "/image.png")
            for j in range(levels):
                heat = torch.sum(heatmaps[j][i], dim=0).cpu().clone().detach().numpy()
                plt.imshow(heat, interpolation='bilinear', cmap='jet')
                # cm 可选参数 cool hot, gray, cmap=plt.cm.cool。 没有的话默认是绿蓝的图
                plt.xticks(())
                plt.yticks(())
                path_level = path_pic + "/level" + str(j + 1) + ".png"
                plt.savefig(path_level)
                plt.close("all")

    def heatmap_level_all(self, img_metas, stu_x, path1, tea_x, path2):
        self.heatmap_level_s(img_metas, stu_x, path1)
        self.heatmap_level_t(img_metas, tea_x, path2)
