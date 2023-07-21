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
import mmcv
import numpy as np
import os


@DETECTORS.register_module()
class Distill_Atss(ATSS):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 tea_model,
                 imitation_loss_weight=0.001,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 tea_pretrained=None
                 ):
        super(Distill_Atss, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)
        self.tea_model = build_detector(tea_model, train_cfg=train_cfg, test_cfg=test_cfg)
        self.load_weights(pretrained=tea_pretrained)
        self.freeze_models()

        # todo 原来的cls_weight为0.01
        self.imitation_loss_weight = 0.01
        self.dis_cls_weight = 0.01
        self.dis_reg_weight = 0.1

        # todo 采用giou作为回归的损失函数
        loss_dis_reg_cfg = dict(type='NIoULoss', loss_weight=2.0)
        self.loss_distill_reg = build_loss(loss_dis_reg_cfg)

        # todo 针对学生的fpn的特征层的尺度变为原来的一半，此处从configs获取学生fpn通道数,手动设定可以被抛弃了
        self.in_channels_adapt = bbox_head.in_channels
        # todo 手动设定学生网络fpn的通道数
        self.in_channels = 128
        self.out_channels = 256
        self.adapt_conv = nn.Conv2d(self.in_channels_adapt, self.out_channels, 1, padding=0)

        self.if_adapt = True
        self.if_adapt2 = False
        self.if_cls = True
        self.if_cls2 = False

        # todo 是否蒸馏backbone
        self.dis_backbone = True
        # todo 是否蒸馏最终的分类头
        self.dis_cls = True
        # todo 是否蒸馏最终的回归头
        self.dis_reg = True

        # todo 对于蒸馏回归的输出，监督方式
        # todo 采用adapt原始论文的监督方式,iou(t,gt)>iou(a,gt)     mask0
        self.if_ori = False
        # todo 强监督   iou(t,gt)>iou(s,gt)>iou(a,gt)           mask3
        self.if_strict = True
        # todo 半强监督 iou(t,gt)>iou(s,gt), iou(t,gt)>iou(a,gt)
        self.if_half_strict = False
        # todo 中等监督 iou(t,gt)>iou(s,gt)                      mask1
        self.if_middle = False
        # todo 初始化适应层的权重
        self._init_adapt_weights()

    def _init_adapt_weights(self):
        normal_init(self.adapt_conv, std=0.01)

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

        # todo stu_x=[5,[n,c,h,w]]
        # todo stu_out=[3,[5,[n,c,h,w]]]
        # todo stu_out[0]分类， stu_outs[1]回归, stu_outs[2]centness
        # todo tupe (3, 5, n, c, h ,w), 3代表三个输出的分支头， 5代表5个尺度，n代表batch_size,
        # todo 分类头的c=15, 代表类别， 检测头c=4, 代表4个偏移量， 点头c=1代表， 代表一个中心点
        # todo [h,w] 128, 64, 32 ,16, 8

        # todo 添加适应层，因为采用的学生网络的fpn层通道数发生了改变，因此采用适应层为学生网络的通道数进行调整。
        if self.if_adapt:
            stu_adapt = []
            stu_adapt = self.adapt(stu_x)
        if self.if_adapt2:
            stu_adapt = []
            stu_adapt = self.adapt(stu_x)

        # todo 变为一个包含5个元素的tupe
        # todo 获取学生网络的输入
        loss_inputs = stu_outs + (gt_bboxes, gt_labels, img_metas,)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # todo 针对分类头的输出进行蒸馏
        # mask_batch_cls = self.get_masks(stu_outs[0], gt_bboxes, img_metas)
        # mask_batch_cls = self.get_masks_2(stu_outs[0], gt_bboxes, img_metas)
        # mask_batch_cls = self.get_masks_3(stu_outs[0], gt_bboxes, img_metas, img)
        mask_batch_cls = self.get_masks_4(stu_outs[0], gt_bboxes, img_metas, img)
        # mask_batch_cls = self.get_masks_fitnet(stu_x, gt_bboxes, img_metas)

        # todo 全新的分类头输出蒸馏，考虑到为不同的通道数设置不同的mask,或者说,仅仅在分类正确的通道设置mask
        # mask_batch_cls = self.get_masks_5(stu_outs[0], gt_bboxes, gt_labels, img_metas, img)
        # mask_batch_cls = self.get_masks_6(stu_outs[0], gt_bboxes, gt_labels, img_metas, img)

        # todo 针对fpn 出来的特征图进行蒸馏
        # mask_batch = self.get_masks(stu_x, gt_bboxes, img_metas)
        # mask_batch = self.get_masks_2(stu_x, gt_bboxes, img_metas)
        # mask_batch = self.get_masks_3(stu_x, gt_bboxes, img_metas, img)
        mask_batch = self.get_masks_4(stu_x, gt_bboxes, img_metas, img)
        # mask_batch = self.get_masks_fitnet(stu_x, gt_bboxes, img_metas)

        # todo 解耦蒸馏
        # mask_batch, mask_batch_fg = self.get_masks_defeat(stu_x, gt_bboxes, img_metas, img)

        # todo ************************绘图函数*******************************
        # todo 可视化展示不同的mask的结果, 需要选择path, 以及不同的mask_batch
        # path = "/home/airstudio/code/yyr/masks/mask0/"
        # path1 = "/home/airstudio/code/yyr/masks/mask1/"
        # path2 = "/home/airstudio/code/yyr/masks/mask2/"
        # path3 = "/home/airstudio/code/yyr/masks/mask3/"
        # path4 = "/home/airstudio/code/yyr/masks/mask4/"

        # todo 同时可视化所有的mask
        # mask_batch1 = self.get_masks(stu_x, gt_bboxes, img_metas)
        # mask_batch2 = self.get_masks_2(stu_x, gt_bboxes, img_metas)
        # mask_batch3 = self.get_masks_3(stu_x, gt_bboxes, img_metas, img)
        # mask_batch4 = self.get_masks_4(stu_x, gt_bboxes, img_metas, img)
        # todo 可视化其中的一个mask或者所有的mask
        # self.show_masks(img_metas, mask_batch4, path4)
        # self.show_mask4s(img_metas, mask_batch4, path4)
        # self.show_masks_all(img_metas, mask_batch1, path1, mask_batch2, path2, mask_batch3, path3, mask_batch4, path4)

        # todo 为每一个每一个图片单独存储一个文件夹
        # self.mask_level(img_metas, mask_batch4, path2)
        # self.mask_level_for_mask4(img_metas, mask_batch4, path4)
        # self.mask_level_all(img_metas, mask_batch1, path1, mask_batch2, path2, mask_batch3, path3, mask_batch4, path4）

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

        # todo 对于fpn层特征图进行蒸馏
        if self.dis_backbone:
            sup_loss_fpn = 0.0
            norm = 0
            norm1 = 0
            for j, mask_per_im in enumerate(mask_batch):
                for i, mask_per_level in enumerate(mask_per_im):
                    mask_per_level = (mask_per_level > 0).float().unsqueeze(0)
                    norm += mask_per_level.sum() * 2

                    # todo 解耦蒸馏
                    # mask_per_level_bg = ((1-mask_per_level) > 0).float().unsqueeze(0)
                    # norm1 += mask_per_level_bg.sum() * 2

                    # todo 此处的损失相当于对学生和教师的网络增加适应层添加监督，一般是学生网络FPN出来的特征图通道数和教师不同
                    # todo 此处有问题，对于mask是按照每一个bach->level,但是下式计算的时候，一个bach的一个level的mask似乎与所有bach
                    # todo 同一个level的图片进行计算，也就是多余计算其他bach与某一个bach的图片的mask的损失
                    if self.if_adapt:
                        sup_loss_fpn += (torch.pow(stu_adapt[i] - tea_x[i], 2) * mask_per_level).sum()
                        # todo 解耦蒸馏
                        # sup_loss_fpn += (torch.pow(stu_adapt[i] - tea_x[i], 2) * (1-mask_per_level_bg)).sum() * 4
                    # todo 在此处进行修正，所有的bach的mask对应图片
                    elif self.if_adapt2:
                        sup_loss_fpn += (torch.pow(stu_adapt[i][j] - tea_x[i][j], 2) * mask_per_level).sum()
                    # todo 此处相当于对学生和教师的FPN出来的特征图增加监督
                    else:
                        sup_loss_fpn += (torch.pow(stu_x[i] - tea_x[i], 2) * mask_per_level).sum()
            sup_loss_fpn = sup_loss_fpn * self.imitation_loss_weight / (norm + 1 +norm1)
            losses.update(dict(loss_sup_fpn=sup_loss_fpn))

        # todo 对最后的分类头进行蒸馏
        if self.dis_cls:
            sup_loss_cls = 0.0
            norm_cls = 0
            for j, mask_per_im in enumerate(mask_batch_cls):
                for i, mask_per_level in enumerate(mask_per_im):
                    mask_per_level = (mask_per_level > 0).float().unsqueeze(0)
                    norm_cls += mask_per_level.sum() * 2
                    # 此处相当于对学生和教师的最后的分类输出增加监督,分类输出为5个尺度，输出的通道数为15
                    # todo 老版的蒸馏策略，存在着batchsize的mask和图片没有对应的问题
                    if self.if_cls:
                        sup_loss_cls += (torch.pow(tea_outs[0][i] - stu_outs[0][i], 2) * mask_per_level).sum()
                    # todo 新型的蒸馏策略，mask和每个图片完全对应上
                    if self.if_cls2:
                        sup_loss_cls += (torch.pow(tea_outs[0][i][j] - stu_outs[0][i][j], 2) * mask_per_level).sum()
            sup_loss_cls = sup_loss_cls * self.dis_cls_weight / (norm_cls + 1)
            losses.update(dict(loss_sup_cls=sup_loss_cls))

        # todo 对回归头进行蒸馏
        if self.dis_reg:
            # todo 获取学生网络和教师网络与真值的的overlaps以及学生和教师网络的各自的bbox_pred
            # todo 首先对教师和学生的overlaps进行比较，此后针对bbox进行损失计算
            # todo stu_reg = tupe[4]
            # todo stu_reg[0] = pos_pred_overlaps_list = [5, [pos_decode_pred]], pos_num代表着该批此正样本的数量, decode_box和gt的iou
            # todo stu_reg[1] = pos_decode_bbox_pred = [5,[pos_decode_pred,4]], pos_anchors = n*h*w[pos]
            # todo stu_reg[2] = pos_bbox_pred = [5,[pos_pred,4]
            # todo stu_reg[3] = pos_anchors_overlaps_list=[5,[pos_nums,1] 正样本anchor和gt的iou
            # todo stu_reg[4] = pos_centerness = [5,[pos_nums]]
            # todo stu_reg[5] = centerness_targets = [5,[pos_nums]]
            # todo stu_reg[6] = pos_anchors = [5,[pos_nums,4]
            stu_reg = self.bbox_head.loss_reg(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            tea_reg = self.tea_model.bbox_head.loss_reg(*tea_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            sup_loss_reg = 0.0
            for i in range(len(stu_reg[0])):

                if len(tea_reg[0][i]) != len(stu_reg[0][i]):
                    continue
                # todo 生成教师的iou大于anchor的矩阵 iou(t,gt)>iou(a,gt)
                mask0_bool = tea_reg[0][i] > tea_reg[3][i].squeeze()
                # todo 生成教师的iou大于学生iou的矩阵 iou(t,gt)>iou(s,gt)
                mask1_bool = tea_reg[0][i] > stu_reg[0][i]
                # todo 生成学生的iou大于原本anchor的矩阵 iou(s,gt)>iou(a,gt)
                mask2_bool = stu_reg[0][i] > stu_reg[3][i].squeeze()
                # todo 严格约束，学习的对象不仅要是教师还要比基础的anhcor还要好，学习样本的数量下降, iou(t,gt)>iou(s,gt)>iou(a,gt)
                mask3_bool = mask1_bool * mask2_bool
                # todo 半严格约束，iou(t,gt)>iou(s,gt), iou(t,gt)>iou(a,gt)
                mask4_bool = mask1_bool + mask0_bool

                # todo 将bool矩阵转换为0,1矩阵
                # todo 全监督
                mask = torch.ones(stu_reg[0][i].shape, dtype=torch.int64).cuda()
                # todo adapt 论文     iou(t,gt)>iou(a,gt)
                mask0 = mask0_bool + 0
                # todo 中等监督        iou(t,gt)>iou(s,gt)
                mask1 = mask1_bool + 0
                # todo 强监督          iou(t,gt)>iou(s,gt)>iou(a,gt)
                mask3 = mask3_bool + 0
                # todo 半强监督        iou(t,gt)>iou(s,gt), iou(t,gt)>iou(a,gt)
                mask4 = mask4_bool + 0

                # todo 计算每一层学习的样本数
                num = mask.sum()
                num0 = mask0.sum() if mask0.sum() > 0 else 1
                num1 = mask1.sum() if mask1.sum() > 0 else 1
                num3 = mask3.sum() if mask3.sum() > 0 else 1
                num4 = mask4.sum() if mask4.sum() > 0 else 1

                # todo 对于mask施加centerness权重
                cen_weight = stu_reg[5][i]

                w = mask * cen_weight
                w0 = mask0 * cen_weight
                w1 = mask1 * cen_weight
                w3 = mask3 * cen_weight
                w4 = mask4 * cen_weight

                # todo 对于cen_mask计算权重
                norm = w.sum() if w.sum() > 0 else 1
                norm0 = w0.sum() if w0.sum() > 0 else 1
                norm1 = w1.sum() if w1.sum() > 0 else 1
                norm3 = w3.sum() if w3.sum() > 0 else 1
                norm4 = w4.sum() if w4.sum() > 0 else 1

                # todo 新型的采用Giouloss-加上全1权重
                # if self.if_strict:
                #     loss_reg = self.loss_distill_reg(tea_reg[1][i], stu_reg[1][i], weight=mask3, avg_factor=1.0)
                #     loss_reg_norm = loss_reg / num3
                # elif self.if_ori:
                #     loss_reg = self.loss_distill_reg(tea_reg[1][i], stu_reg[1][i], weight=mask0, avg_factor=1.0)
                #     loss_reg_norm = loss_reg / num0
                # elif self.if_middle:
                #     loss_reg = self.loss_distill_reg(tea_reg[1][i], stu_reg[1][i], weight=mask1, avg_factor=1.0)
                #     loss_reg_norm = loss_reg / num1
                # else:
                #     loss_reg = self.loss_distill_reg(tea_reg[1][i], stu_reg[1][i], weight=mask, avg_factor=1.0)
                #     loss_reg_norm = loss_reg / num

                # todo 新型的采用Giouloss-加上cen权重
                if self.if_strict:
                    loss_reg = self.loss_distill_reg(tea_reg[1][i], stu_reg[1][i], weight=w3, avg_factor=1.0)
                    loss_reg_norm = loss_reg / norm3
                elif self.if_ori:
                    loss_reg = self.loss_distill_reg(tea_reg[1][i], stu_reg[1][i], weight=w0, avg_factor=1.0)
                    loss_reg_norm = loss_reg / norm0
                elif self.if_middle:
                    loss_reg = self.loss_distill_reg(tea_reg[1][i], stu_reg[1][i], weight=w1, avg_factor=1.0)
                    loss_reg_norm = loss_reg / norm1
                else:
                    loss_reg = self.loss_distill_reg(tea_reg[1][i], stu_reg[1][i], weight=w, avg_factor=1.0)
                    loss_reg_norm = loss_reg / norm

                # # todo 原来的计算的L1loss-加上全1权重
                # loss_reg = abs(tea_reg[2][i] - stu_reg[2][i])
                # loss_reg_sum = loss_reg.sum(dim=1)
                # if self.if_strict:
                #     loss_reg_norm = (loss_reg_sum * mask3).sum()/num3
                # elif self.if_ori:
                #     loss_reg_norm = (loss_reg_sum * mask0).sum()/num0
                # elif self.if_middle:
                #     loss_reg_norm = (loss_reg_sum * mask1).sum()/num1
                # else:
                #     loss_reg_norm = (loss_reg_sum * mask).sum()/num

                sup_loss_reg = sup_loss_reg + loss_reg_norm
            sup_loss_reg = sup_loss_reg * self.dis_reg_weight
            losses.update(dict(loss_sup_reg=sup_loss_reg))

        return losses

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

    # todo mask 生成函数*******************************************************
    # todo 为FPN之后的特征设计的mask
    # todo 为了每一个层级设计自己的阈值
    def get_masks(self, cls_scores, gt_bboxes, img_metas):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(featmap_sizes, img_metas, device=device)
        mask_batch = []
        A = self.bbox_head.num_anchors
        batch_size = len(gt_bboxes)
        levels = len(anchor_list[0])
        for i in range(batch_size):
            mask_per_im = []
            for j in range(levels):
                height, width = featmap_sizes[j][0], featmap_sizes[j][1]
                all_anchors = anchor_list[i][j]
                # 由于部分的切片里面存在着无标签的情况，跳过这个部分
                if gt_bboxes[i].shape[0] == 0:
                    continue
                IOU_map = bbox_overlaps(all_anchors, gt_bboxes[i]).view(height, width, A, gt_bboxes[i].shape[0])
                max_iou, _ = torch.max(IOU_map.view(height * width * A, gt_bboxes[i].shape[0]), dim=0)
                mask_per_level = torch.zeros([height, width], dtype=torch.int64).cuda()
                for k in range(gt_bboxes[i].shape[0]):
                    if torch.sum(gt_bboxes[i][k]) == 0.:
                        break
                    max_iou_per_gt = max_iou[k] * 0.5
                    mask_per_gt = torch.sum(IOU_map[:, :, :, k] > max_iou_per_gt, dim=2)
                    mask_per_level += mask_per_gt
                mask_per_im.append(mask_per_level)
            mask_batch.append(mask_per_im)

        return mask_batch
    # todo 所有的层级采用相同的阈值
    def get_masks_2(self, cls_scores, gt_bboxes, img_metas):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(featmap_sizes, img_metas, device=device)
        mask_batch = []
        A = self.bbox_head.num_anchors
        batch_size = len(gt_bboxes)
        levels = len(anchor_list[0])
        for i in range(batch_size):
            IOU_maps = []
            for j in range(levels):
                height, width = featmap_sizes[j][0], featmap_sizes[j][1]
                all_anchors = anchor_list[i][j]
                # Todo
                # 由于部分的切片里面存在着无标签的情况，则将这个iou_map全部置位0
                if gt_bboxes[i].shape[0] != 0:
                    IOU_map = bbox_overlaps(all_anchors, gt_bboxes[i]).view(height, width, A, gt_bboxes[i].shape[0])
                else:
                    IOU_map = torch.zeros([height, width, A, 1], dtype=torch.int64).cuda()

                # IOU_map = bbox_overlaps(all_anchors, gt_bboxes[i]).view(height, width, A, gt_bboxes[i].shape[0])

                IOU_maps.append(IOU_map)
                if gt_bboxes[i].shape[0] != 0:
                    max_iou, _ = torch.max(IOU_map.view(height * width * A, gt_bboxes[i].shape[0]), dim=0)
                else:
                    max_iou, _ = torch.max(IOU_map.view(height * width * A, 1), dim=0)
                # Todo

                # max_iou, _ = torch.max(IOU_map.view(height * width * A, gt_bboxes[i].shape[0]), dim=0)
                if j > 0:
                    max_iou = torch.cat((max_iou, pre_max), 0)
                    max_iou, _ = torch.max(max_iou.view(2, -1), 0)
                pre_max = max_iou

            mask_per_im = []
            for j in range(levels):

                # # 由于无标签的问题
                # if len(IOU_maps[j]):
                #     IOU_map = IOU_maps[j]
                # else:
                #     IOU_map = torch.zeros([featmap_sizes[j][0], featmap_sizes[j][1], A, 1], dtype=torch.int64).cuda()
                #

                IOU_map = IOU_maps[j]

                height, width = featmap_sizes[j][0], featmap_sizes[j][1]
                mask_per_level = torch.zeros([height, width], dtype=torch.int64).cuda()
                for k in range(gt_bboxes[i].shape[0]):
                    if torch.sum(gt_bboxes[i][k]) == 0.:
                        break
                    max_iou_per_gt = max_iou[k] * 0.5
                    mask_per_gt = torch.sum(IOU_map[:, :, :, k] > max_iou_per_gt, dim=2)
                    mask_per_level += mask_per_gt
                mask_per_im.append(mask_per_level)
            mask_batch.append(mask_per_im)

        return mask_batch

    def get_masks_3(self, cls_scores, gt_bboxes, img_metas, img):
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
            # todo 计算原始图片的尺寸
            area_image = torch.sqrt((img_ori[:, 0] * img_ori[:, 1]).to(torch.float32)).cuda()
            # todo area_image=[1],值=[1024]
            # todo 获得不同尺寸的gt应该分配给不同的层级,类似于fpn的操作
            target_lvls = torch.floor(4 + torch.log2(area / area_image + 1e-6)).cuda()
            # todo 将level限制5个fpn层级
            target_lvls = target_lvls.clamp(min=0, max=4).long()
            # todo target_lvls=[gt_num], 值=[0~4]
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
                # print(gt_mask.shape)
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
                    # todo 将mask 里面gt位置的值置为1
                    mask_per_gt[gt_trans[k][1].int():gt_trans[k][3].int(),
                    gt_trans[k][0].int():gt_trans[k][2].int()] = 1
                    # todo 将这个level里面所有的gt位置叠加起来，得到一个层级特征mask
                    mask_per_level += mask_per_gt
                # todo 得到一张图片的特征mask
                mask_per_im.append(mask_per_level)
            # todo 得到一个batch所有的mask
            mask_batch.append(mask_per_im)
        return mask_batch

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

    # todo 为cls设计的mask
    def get_masks_5(self, cls_scores, gt_bboxes, gt_label, img_metas, img):
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
            # todo gt_bboxes=[bachsize,[gt_num,4]]
            # todo gt_label=[batchsize,[gt_num]]
            # todo 根据gt_box计算其面积
            area = torch.sqrt((gt_bboxes[i][:, 2] - gt_bboxes[i][:, 0]) * (gt_bboxes[i][:, 3] - gt_bboxes[i][:, 1]))
            # todo arae=[gt_num], 值=[0~1024]
            # todo 计算原始图片的尺寸
            area_image = torch.sqrt((img_ori[:, 0] * img_ori[:, 1]).to(torch.float32)).cuda()
            # todo area_image=[1],值=[1024]
            # todo 获得不同尺寸的gt应该分配给不同的层级,类似于fpn的操作
            target_lvls = torch.floor(4 + torch.log2(area / area_image + 1e-6)).cuda()
            # todo 将level限制5个fpn层级
            target_lvls = target_lvls.clamp(min=0, max=4).long()
            # todo target_lvls=[gt_num], 值=[0~4]
            for j in range(levels):
                height, width = featmap_sizes[j][0], featmap_sizes[j][1]
                mask_per_level = torch.tensor([], dtype=torch.int64).cuda()
                # todo 获取每个特征图的尺寸
                fea_scale = featmap_sizes[j]
                # todo 将特征图尺寸转换为tensor
                fea_scale = torch.tensor([fea_scale])
                # todo fea_scale=[1,1] 值=[128,128],[64,64],[32,32],[16,16],[8,8]
                # todo 计算出每一层特征图相对于原始图片的缩放比率。
                # todo 对于dota刚好是长款等比，但是nwpu长宽比例不同，可能后面需要调整顺序
                percent = torch.div(fea_scale.float(), img_ori.float()).cuda()
                # todo percent=[1,1] 值=[0.125], [0.0625],[0.0312],[0.0156],[0.0078]
                # todo 构建按比例缩放后的gt坐标组
                gt_trans = torch.zeros(gt_bboxes[i].shape, dtype=torch.float32).cuda()
                gt_trans[:, :2] = gt_bboxes[i][:, :2] * percent
                gt_trans[:, 2:] = gt_bboxes[i][:, 2:] * percent
                # todo gt坐标放缩到对应的特征图上的位置
                # todo gt_trans=[gt_nums,4]
                # todo gt_mask尺寸为和gt_bboxes尺寸相同，gt_mask=[gt_num,4]
                gt_mask = torch.zeros(gt_bboxes[i].shape, dtype=torch.float32).cuda()
                # todo 根据当前图片里每所有的gt，生成对应当前层级的mask
                for k in range(gt_bboxes[i].shape[0]):
                    # todo 如果第k个gt分配的level层级等于当前的level层级，则将该层级的gt的gt_mask置为1
                    if target_lvls[k] == j:
                        gt_mask[k, :] = 1
                # todo 将mask和gt_trans相乘，得到对应的gt
                gt_trans = gt_trans * gt_mask

                # todo 实现多通道mask
                # todo cls_scores=[5,[8, channels,h,w,]
                # todo 为每个通道生成mask
                for c in range(len(cls_scores[0][2])):
                    mask_per_channel = torch.zeros(height, width, dtype=torch.int64).cuda()
                    for k in range(gt_bboxes[i].shape[0]):
                        # todo 如果 该层的gt数量为0直接退出，等效于某张图片里面没有任何真值
                        if torch.sum(gt_bboxes[i][k]) == 0.:
                            break
                        # todo 为每个Gt单独计算mask
                        mask_per_gt = torch.zeros(height, width, dtype=torch.int64).cuda()
                        # todo 只有当前的gt的类别和目前通道数能对应上
                        # todo gt_label=[batchsize,[gt_num]]
                        if gt_label[i][k] == c:
                            # todo 将mask 里面gt位置的值置为1
                            mask_per_gt[gt_trans[k][1].int():gt_trans[k][3].int(),
                            gt_trans[k][0].int():gt_trans[k][2].int()] = 1
                        mask_per_channel += mask_per_gt
                    # todo 得到每一个尺度的所有通道的mask
                    mask_per_channel_new = torch.unsqueeze(mask_per_channel, dim=0)
                    mask_per_level = torch.cat((mask_per_level, mask_per_channel_new), dim=0)
                # todo 得到一张图片的每个尺度的特征mask
                mask_per_im.append(mask_per_level)
            # todo 得到一个batch所有的mask
            mask_batch.append(mask_per_im)
        return mask_batch

    def get_masks_6(self, cls_scores, gt_bboxes, gt_label, img_metas, img):
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
            # todo gt_label=[gt_num]
            # todo 根据gt_box计算其面积
            area = torch.sqrt((gt_bboxes[i][:, 2] - gt_bboxes[i][:, 0]) * (gt_bboxes[i][:, 3] - gt_bboxes[i][:, 1]))
            # todo arae=[gt_num], 值=[0~1024]
            # todo 计算原始图片的尺寸
            area_image = torch.sqrt((img_ori[:, 0] * img_ori[:, 1]).to(torch.float32)).cuda()
            # todo area_image=[1],值=[1024]
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
                mask_per_level = torch.tensor([], dtype=torch.int64).cuda()
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
                # print(gt_mask.shape)
                # todo 根据当前图片里每所有的gt，生成对应当前层级的mask
                for k in range(gt_bboxes[i].shape[0]):
                    # todo 如果第k个gt分配的level层级等于当前的level层级，则将该层级的gt的gt_mask置为1
                    if target_lvls[k] == j:
                        gt_mask[k, :] = 1
                # todo 将mask和gt_trans相乘，得到对应的gt
                gt_trans = gt_trans * gt_mask

                # todo 实现多通道mask
                # todo cls_scores=[5,[8, channels,h,w,]
                # todo 为每个通道生成mask
                for c in range(len(cls_scores[0][2])):
                    mask_per_channel = torch.zeros(height, width, dtype=torch.int64).cuda()
                    for k in range(gt_bboxes[i].shape[0]):
                        # todo 如果 该层的gt数量为0直接退出，等效于某张图片里面没有任何真值
                        if torch.sum(gt_bboxes[i][k]) == 0.:
                            break
                        # todo 为每个Gt单独计算mask
                        mask_per_gt = torch.zeros(height, width, dtype=torch.int64).cuda()
                        # todo 只有当前的gt的类别和目前通道数能对应上
                        if gt_label[i][k] == c:
                            # todo 将mask 里面gt位置的值置为1
                            mask_per_gt[gt_trans[k][1].int():gt_trans[k][3].int(),
                            gt_trans[k][0].int():gt_trans[k][2].int()] = area_weight[k]
                        mask_per_channel += mask_per_gt
                    # todo 得到每一个尺度的所有通道的mask
                    mask_per_channel_new = torch.unsqueeze(mask_per_channel, dim=0)
                    mask_per_level = torch.cat((mask_per_level, mask_per_channel_new), dim=0)
                # todo 得到一张图片的每个尺度的特征mask
                mask_per_im.append(mask_per_level)
            # todo 得到一个batch所有的mask
            mask_batch.append(mask_per_im)
        return mask_batch

    # todo 消融实验*******************
    # todo 消融实验 fitnet
    def get_masks_fitnet(self, cls_scores, gt_bboxes, img_metas):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(featmap_sizes, img_metas, device=device)
        mask_batch = []
        batch_size = len(gt_bboxes)
        levels = len(anchor_list[0])
        # todo batchsize
        for i in range(batch_size):
            mask_per_im = []
            for j in range(levels):
                height, width = featmap_sizes[j][0], featmap_sizes[j][1]
                mask_per_level = torch.ones(height, width).cuda()
                mask_per_im.append(mask_per_level)
            mask_batch.append(mask_per_im)
        return mask_batch

    # todo 消融实验 KD
    def get_masks_KD(self, cls_scores, gt_bboxes, img_metas):
        mask_batch =[]
        return mask_batch

    # todo 消融实验 TAR 就是mask3加分类回归，

    # todo 消融实验 defeat
    def get_masks_defeat(self, cls_scores, gt_bboxes, img_metas, img):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        # todo 获取一个批次图片的尺寸
        img_ori = img.size()[-2:]
        img_ori = torch.tensor([img_ori])
        # todo img_ori=[1024,1024]
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(featmap_sizes, img_metas, device=device)
        mask_batch = []
        mask_batch_bg = []

        batch_size = len(gt_bboxes)
        levels = len(anchor_list[0])
        for i in range(batch_size):
            mask_per_im = []
            mask_per_im_bg = []
            # todo gt_bboxes=[gt_num,4]
            # todo 根据gt_box计算其面积
            area = torch.sqrt((gt_bboxes[i][:, 2] - gt_bboxes[i][:, 0]) * (gt_bboxes[i][:, 3] - gt_bboxes[i][:, 1]))
            # todo arae=[gt_num], 值=[0~1024]
            # todo 计算原始图片的尺寸
            area_image = torch.sqrt((img_ori[:, 0] * img_ori[:, 1]).to(torch.float32)).cuda()
            # todo area_image=[1],值=[1024]
            # todo 获得不同尺寸的gt应该分配给不同的层级,类似于fpn的操作
            target_lvls = torch.floor(4 + torch.log2(area / area_image + 1e-6)).cuda()
            # todo 将level限制5个fpn层级
            target_lvls = target_lvls.clamp(min=0, max=4).long()
            # todo target_lvls=[gt_num], 值=[0~4]
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
                # print(gt_mask.shape)
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
                    # todo 将mask 里面gt位置的值置为1
                    mask_per_gt[gt_trans[k][1].int():gt_trans[k][3].int(),
                    gt_trans[k][0].int():gt_trans[k][2].int()] = 1
                    # todo 将这个level里面所有的gt位置叠加起来，得到一个层级特征mask
                    mask_per_level += mask_per_gt
                # todo 获取背景的特征
                mask_per_level_bg = 1 - mask_per_level
                # exit()
                # todo 得到一张图片的特征mask
                mask_per_im.append(mask_per_level)
                mask_per_im_bg.append(mask_per_level_bg)
            # todo 得到一个batch所有的mask
            mask_batch.append(mask_per_im)
            mask_batch_bg.append(mask_per_im_bg)
        return mask_batch, mask_batch_bg

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

    # todo 绘图函数 ***********************************************************
    # todo 为mask4绘制的带有颜色的区分的可视化结果
    def show_mask4s(self, img_metas, mask_batch, path):
        for i, img_meta in enumerate(img_metas):
            filename = img_meta['filename']
            path_pic = path + filename.split("/")[-1].split(".")[0]
            self.mkdir(path_pic)
            flip = img_meta['flip']
            img = cv2.imread(filename)
            if flip:
                img = cv2.flip(img, 1)
            mask_per_im = mask_batch[i]
            plt.imshow(img)
            plt.xticks(())
            plt.yticks(())
            # plt.savefig(path_pic + "/img.png")
            cv2.imwrite(path_pic+"/img.png", img)
            fig = plt.figure(figsize=(22, 6))
            norm = matplotlib.colors.Normalize(vmin=0, vmax=3)
            for j, mask in enumerate(mask_per_im):
                plt.subplot(1, 5, j + 1)
                mask = mask.cpu().numpy()
                # photo = plt.contour(mask, cmap='hot', norm=norm)
                # plt.imshow(mask, cmap=plt.cm.gray)
                photo = plt.imshow(mask, interpolation='nearest', cmap='hot', origin='upper', norm=norm)
                plt.xticks(())
                plt.yticks(())
            fig.subplots_adjust(right=0.9)
            l = 0.92
            b = 0.25
            w = 0.015
            h = 1 - 2 * b - 0.01
            rect = [l, b, w, h]
            cbar_ax = fig.add_axes(rect)
            cb = plt.colorbar(photo, cax=cbar_ax)
            cb.ax.tick_params(labelsize=16)
            font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 16, }
            cb.set_label('Area Weighted Mask', fontdict=font)
            plt.savefig(path_pic + "/mask.png")
            plt.close("all")

    # todo 为m1.m2.m3的0-1可视化结果
    def show_masks(self, img_metas, mask_batch, path):
        for i, img_meta in enumerate(img_metas):
            filename = img_meta['filename']
            path_pic = path + filename.split("/")[-1].split(".")[0]
            self.mkdir(path_pic)
            flip = img_meta['flip']
            img = cv2.imread(filename)
            if flip:
                img = cv2.flip(img, 1)
            mask_per_im = mask_batch[i]
            plt.imshow(img)
            plt.xticks(())
            plt.yticks(())
            # plt.savefig(path_pic + "/img.png")
            cv2.imwrite(path_pic + "/img.png", img)
            fig = plt.figure(figsize=(22, 6))
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            for j, mask in enumerate(mask_per_im):
                plt.subplot(1, 5, j + 1)
                mask = mask.cpu().numpy() > 0
                # photo = plt.contour(mask, cmap='hot', norm=norm)
                # plt.imshow(mask, cmap=plt.cm.gray)
                photo = plt.imshow(mask, interpolation='nearest', cmap='gray', origin='upper', norm=norm)
                plt.xticks(())
                plt.yticks(())
            fig.subplots_adjust(right=0.9)
            l = 0.92
            b = 0.25
            w = 0.015
            h = 1 - 2 * b - 0.01
            rect = [l, b, w, h]
            cbar_ax = fig.add_axes(rect)
            cb = plt.colorbar(photo, cax=cbar_ax)
            cb.ax.tick_params(labelsize=16)
            font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 16, }
            cb.set_label('Binary Mask', fontdict=font)
            plt.savefig(path_pic + "/mask.png")
            plt.close("all")

            # plt.savefig("/workspace/a_yyr_code/yyr/masks/mask0/" + filename.split("/")[-1])

    # todo 展示所有的mask
    def show_masks_all(self, img_metas, mask_batch1, path1,
                       mask_batch2, path2,
                       mask_batch3, path3,
                       mask_batch4, path4):
        self.show_masks(img_metas, mask_batch1, path1)
        self.show_masks(img_metas, mask_batch2, path2)
        self.show_masks(img_metas, mask_batch3, path3)
        self.show_mask4s(img_metas, mask_batch4, path4)

    # todo &&&&&&&&&&&&&&&&&&&&&&
    # todo 为每张图片的mask生成一个文件夹
    def mask_level(self, img_metas, mask_batch, path):
        for i, img_meta in enumerate(img_metas):
            filename = img_meta['filename']
            # print(filename.split("/")[-1].split(".")[0])
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
            mask_per_im = mask_batch[i]
            for j, mask in enumerate(mask_per_im):
                mask = mask.cpu().numpy() > 0
                plt.imshow(mask, cmap=plt.cm.gray)
                plt.xticks(())
                plt.yticks(())
                # plt.colorbar()
                path_level = path_pic + "/level" + str(j + 1) + ".png"
                print(path_level)
                plt.savefig(path_level)
                plt.close("all")

        # todo 为mask4 单独绘制带有颜色bar的图片

    def mask_level_for_mask4(self, img_metas, mask_batch, path):
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
            mask_per_im = mask_batch[i]
            for j, mask in enumerate(mask_per_im):
                mask = mask.cpu().numpy()
                plt.imshow(mask, interpolation="nearest", cmap='hot', origin='upper')
                plt.xticks(())
                plt.yticks(())
                # plt.colorbar()
                if mask.sum() > 0:
                    cb = plt.colorbar()
                    cb.set_ticks(np.linspace(0, 2, 5))
                    cb.set_ticklabels(('0', '0.5', '1', '1.5', '2'))
                # cbar.set_ticklabels((0,0.5,1,1.0,1.5,2.0))
                path_level = path_pic + "/level" + str(j + 1) + ".png"
                print(path_level)
                plt.savefig(path_level)
                plt.close("all")

        # todo 同时为所有的mask同时生成

    def mask_level_all(self, img_metas, mask_batch1, path1,
                       mask_batch2, path2,
                       mask_batch3, path3,
                       mask_batch4, path4):
        self.mask_level(img_metas, mask_batch1, path1)
        self.mask_level(img_metas, mask_batch2, path2)
        self.mask_level(img_metas, mask_batch3, path3)
        self.mask_level_for_mask4(img_metas, mask_batch4, path4)

    # todo &&&&&&&&&&&&&&&&&&&&&&
    # todo 展示教师的网络的热力图
    def show_heatmap_t(self, img_meta, heatmaps, path):
        # todo heatmap 可以tea_x=[5,[n,c,h,w]], stu_x=[5,[n,c,h,w]]
        featmap_size = [heatmap.size()[-2:] for heatmap in heatmaps]
        levels = len(heatmaps)
        self.mkdir(path)
        for i, img_meta in enumerate(img_meta):
            filename = img_meta['filename']
            flip = img_meta['flip']
            img = cv2.imread(filename)
            if flip:
                img = cv2.flip(img, 1)
            plt.subplot(2, 3, 1)
            plt.imshow(img)
            plt.xticks(())
            plt.yticks(())
            for j in range(levels):
                plt.subplot(2, 3, j + 2)
                heat = torch.sum(heatmaps[j][i], dim=0).cpu().numpy()
                plt.imshow(heat, interpolation='bilinear')
                plt.xticks(())
                plt.yticks(())
                # cm 可选参数 cool hot, gray, cmap=plt.cm.cool。 没有的话默认是绿蓝的图
            plt.savefig(path + filename.split("/")[-1])

    # todo 展示学生网络的热力图
    def show_heatmap_s(self, img_meta, heatmaps, path):
        # todo heatmap 可以tea_x=[5,[n,c,h,w]], stu_x=[5,[n,c,h,w]]
        levels = len(heatmaps)
        self.mkdir(path)
        for i, img_meta in enumerate(img_meta):
            filename = img_meta['filename']
            flip = img_meta['flip']
            img = cv2.imread(filename)
            if flip:
                img = cv2.flip(img, 1)
            plt.subplot(2, 3, 1)
            plt.imshow(img)
            plt.xticks(())
            plt.yticks(())
            for j in range(levels):
                plt.subplot(2, 3, j + 2)
                heat = torch.sum(heatmaps[j][i], dim=0).cpu().clone().detach().numpy()
                plt.imshow(heat, interpolation='bilinear')
                plt.xticks(())
                plt.yticks(())
                # cm 可选参数 cool hot, gray, cmap=plt.cm.cool。 没有的话默认是绿蓝的图
            plt.savefig(path + filename.split("/")[-1])

    # todo 展示所有的热力图
    def show_heatmap_all(self, img_metas, stu_x, path1, tea_x, path2):
        self.show_heatmap_s(img_metas, stu_x, path1)
        self.show_heatmap_t(img_metas, tea_x, path2)

    # todo &&&&&&&&&&&&&&&&&&&&&&
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
                plt.imshow(heat, interpolation='bilinear')
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
                plt.imshow(heat, interpolation='bilinear')
                plt.xticks(())
                plt.yticks(())
                path_level = path_pic + "/level" + str(j + 1) + ".png"
                plt.savefig(path_level)
                plt.close("all")

    def heatmap_level_all(self, img_metas, stu_x, path1, tea_x, path2):
        self.heatmap_level_s(img_metas, stu_x, path1)
        self.heatmap_level_t(img_metas, tea_x, path2)

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


