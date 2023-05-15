import torch
from mmdet.models import build_detector
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmdet.core import bbox_overlaps
from ..builder import DETECTORS
from .retinanet import RetinaNet
import cv2
import matplotlib.pyplot as plt
from mmcv.cnn import ConvModule, xavier_init
import torch.nn as nn
import random
@DETECTORS.register_module()
class Distill_RetinaNet(RetinaNet):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 tea_model,
                 imitation_loss_weigth=0.1,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 tea_pretrained=None
                 ):
        super(Distill_RetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)
        self.tea_model = build_detector(tea_model, train_cfg=train_cfg, test_cfg=test_cfg)
        self.load_weights(pretrained=tea_pretrained)
        self.freeze_models()
        self.imitation_loss_weigth = imitation_loss_weigth
        # self.adapt = adapt()
        self.in_channels = 256
        self.out_channels = 64
        self.adapt_conv = nn.Conv2d(self.in_channels, self.out_channels, 1, padding=0)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        self.tea_model.eval()
        with torch.no_grad():
            tea_x = self.tea_model.extract_feat(img)
            tea_outs = self.tea_model.bbox_head(tea_x)##tea_outs[0]分类 tea_outs[1]回归

        stu_x = self.extract_feat(img)
        # 下面相当于直接调用了bbox_head(stu_x).forward
        stu_outs = self.bbox_head(stu_x)
        # print(stu_x[0].shape)
        # print(stu_x[0].shape)
        # print(stu_x[1].shape)
        # print(stu_x[2].shape)
        # print(stu_x[3].shape)
        # print(stu_x[4].shape)
        # print("****")
        # print(stu_outs[0][0].shape)
        # print(stu_outs[0][1].shape)
        # print(stu_outs[0][2].shape)
        # print(stu_outs[0][3].shape)
        # print(stu_outs[0][4].shape)
        # print("%%%%")
        # a = adapt
        # stu_adapt = self.adapt(stu_x[0])
        # print(stu_adapt.shape)
        # print(tea_x[0].shape)
        # 组成了一个元组传入参数。
        loss_inputs = stu_outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # mask_batch = self.get_masks_2(stu_outs[0],gt_bboxes,img_metas)
        mask_batch = self.get_masks_2(stu_x, gt_bboxes, img_metas)
        #self.show_masks(img_metas,mask_batch)

        sup_loss = 0.0
        norm = 0
        for mask_per_im in mask_batch:
            for i,mask_per_level in enumerate(mask_per_im):
                mask_per_level = (mask_per_level > 0).float().unsqueeze(0)
                norm += mask_per_level.sum() * 2
                # 师兄的代码，很明显蒸的不是特征图，而是分类的结果
                # sup_loss +=  (torch.pow(tea_outs[0][i] - stu_outs[0][i], 2) * mask_per_level).sum()
                # 我的代码，蒸的每个fpn层得到的特征图
                sup_loss += (torch.pow(stu_x[i] - tea_x[i], 2) * mask_per_level).sum()

                # print(stu_outs[0][i].shape)
                # print(tea_outs[0][i].shape)
                # print(stu_outs[0][1].shape)
                # print(stu_outs[0][2].shape)
                # print(stu_outs[0][3].shape)
                # print(stu_outs[0][4].shape)
                # print(mask_per_level.shape)
                # norms = mask_per_level.sum() * 2
                # if norms==0:
                #     continue
                # # sup_loss += (torch.pow(tea_outs[0][i] - stu_outs[0][i], 2) * mask_per_level).sum() / norms
                # sup_loss+=(torch.pow(tea_outs[0][i] - stu_outs[0][i], 2) * mask_per_level).sum() / norms /len(mask_batch) /len(mask_per_im)
        sup_loss = sup_loss * self.imitation_loss_weigth/(norm+1)

        losses.update(dict(loss_sup=sup_loss))

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

    def get_masks(self, cls_scores, gt_bboxes, img_metas):
        """Get anchors according to feature map sizes.

        Args:
            cls_scores (list[tensor]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            list: mask_batch of each image
        这个程序在获取的每一个层上面的最大iou
        """
        "此时的cls_scores还没有经过softmax,得到还是特征图的形式，长宽为特征图，通道数为num_anchors * num_classes"
        "cls = [batch, anchor*类别数量（9*10），h，w]"
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(featmap_sizes, img_metas, device=device)
        mask_batch = []
        # A=9
        A = self.bbox_head.num_anchors
        batch_size = len(gt_bboxes)
        levels = len(anchor_list[0])
        #为每一个batchsize循环
        for i in range(batch_size):
            mask_per_im = []
            for j in range(levels):
                height, width = featmap_sizes[j][0], featmap_sizes[j][1]
                all_anchors = anchor_list[i][j]
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

    def get_masks_2(self, cls_scores, gt_bboxes, img_metas):
        """Get anchors according to feature map sizes.
    
        Args:
            cls_scores (list[tensor]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
    
        Returns:
            list: mask_batch of each image
        获取所有层上面的mask
        """
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
                IOU_map = bbox_overlaps(all_anchors, gt_bboxes[i]).view(height, width, A, gt_bboxes[i].shape[0])
                IOU_maps.append(IOU_map)
                max_iou, _ = torch.max(IOU_map.view(height * width * A, gt_bboxes[i].shape[0]), dim=0)
                # print("1")
                # print(max_iou)
                # print("2")
                if j>0:
                    max_iou = torch.cat((max_iou,pre_max),0)
                    # print(max_iou)
                    # print("3")
                    # print(max_iou.view(2,-1))
                    # print("4")
                    max_iou, _ = torch.max(max_iou.view(2,-1),0)
                    # print(max_iou)
                pre_max = max_iou
    
            mask_per_im = []
            for j in range(levels):
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

    def show_masks(self, img_metas, mask_batch):
        for i,img_meta in enumerate(img_metas):
            filename=img_meta['filename']
            flip = img_meta['flip']
            img = cv2.imread(filename)
            if flip:
                img = cv2.flip(img,1)
            mask_per_im = mask_batch[i]
            plt.subplot(1, 6, 1)
            plt.imshow(img)
            for j,mask in enumerate(mask_per_im):
                plt.subplot(1, 6, j+2)
                mask = mask.cpu().numpy()>0
                plt.imshow(mask)
            #plt.show()
            plt.savefig("/yyr/masks/"+filename.split("/")[-1])

    def adapt(self, x):
        out = self.adapt_conv(x)
        return out

