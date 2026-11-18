import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init

from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
                        force_fp32, images_to_levels, multi_apply,
                        multiclass_nms, unmap)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead

EPS = 1e-12
try:
    import sklearn.mixture as skm
except ImportError:
    skm = None


def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


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


@HEADS.register_module()
class ATSSHeadDIL(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(ATSSHeadDIL, self).__init__(num_classes, in_channels, **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.loss_centerness = build_loss(loss_centerness)
        self.topk = 9
        self.covariance_type = 'diag'

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.atss_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.atss_centerness = nn.Conv2d(
            self.feat_channels, self.num_anchors * 1, 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.anchor_generator.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.atss_cls, std=0.01, bias=bias_cls)
        normal_init(self.atss_reg, std=0.01)
        normal_init(self.atss_centerness, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                centerness (Tensor): Centerness for a single scale level, the
                    channel number is (N, num_anchors * 1, H, W).
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        centerness = self.atss_centerness(reg_feat)
        return cls_score, bbox_pred, centerness

    def loss_single(self, anchors, cls_score, bbox_pred, centerness, labels,
                    label_weights, bbox_targets, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            num_total_samples (int): Number os positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]
            # todo LE demo
            # le_targets = self.LC_target()

            centerness_targets = self.centerness_target(
                pos_anchors, pos_bbox_targets)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)
            pos_decode_bbox_targets = self.bbox_coder.decode(
                pos_anchors, pos_bbox_targets)
            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=centerness_targets,
                avg_factor=1.0)

            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness,
                centerness_targets,
                avg_factor=num_total_samples)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = torch.tensor(0).cuda()

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             name=None,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos).cuda()).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, loss_centerness, \
        bbox_avg_factor = multi_apply(
            self.loss_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            centernesses,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            num_total_samples=num_total_samples)

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        # todo 根据名字返回不同的损失函数的名字
        if name == 'teacher':
            return dict(
                loss_cls_tea=losses_cls,
                loss_bbox_tea=losses_bbox,
                loss_centerness_tea=loss_centerness)
        else:
            return dict(
                loss_cls=losses_cls,
                loss_bbox=losses_bbox,
                loss_centerness=loss_centerness)

    def centerness_target(self, anchors, bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        gts = self.bbox_coder.decode(anchors, bbox_targets)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    def LC_target(self, anchors, bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        gts = self.bbox_coder.decode(anchors, bbox_targets)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        # todo 计算lc
        sigma = abs(t_-b_)/(2 * (t_ + b_)) + abs(l_-r_)/(2*(l_+r_))
        # print(sigma)
        # print(sigma.shape)
        # exit()
        # left_right = torch.stack([l_, r_], dim=1)
        # top_bottom = torch.stack([t_, b_], dim=1)
        # centerness = torch.sqrt(
        #     (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
        #     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        # todo 判断是否为空，为空的话，触发警报
        if torch.isnan(sigma).any():
            sigma = torch.ones_like(l_)
        return sigma

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                centerness_pred_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           centernesses,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                Has shape (num_anchors * 1, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, anchors in zip(
                cls_scores, bbox_preds, centernesses, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]

            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        return det_bboxes, det_labels

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single,
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of postive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,),
                                  self.background_label,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside

        # todo 此处为了实现面积加权所采用的新型损失函数

    # todo 将学生的输出引出来
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss_stu(self,
                 cls_scores,
                 bbox_preds,
                 centernesses,
                 gt_bboxes,
                 gt_labels,
                 img_metas,
                 gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_paa_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds, pos_gt_index) = cls_reg_targets

        # todo 将每个level上面的的分数转换为图片上
        cls_scores = levels_to_images(cls_scores)
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores
        ]
        test = cls_scores
        bbox_preds = levels_to_images(bbox_preds)
        bbox_preds = [item.reshape(-1, 4) for item in bbox_preds]

        # todo 计算得到过去的正样本
        labels_old = torch.cat(labels_list, 0).view(-1)
        pos_inds_flatten_old = ((labels_old >= 0) & (labels_old < self.num_classes)).nonzero().reshape(-1)

        # todo 计算所有正样本的损失， 利用cls和reg分别计算得分
        pos_losses_list, = multi_apply(self.get_pos_loss, anchor_list,
                                       cls_scores, bbox_preds, labels_list,
                                       label_weights_list, bbox_targets_list,
                                       bbox_weights_list, pos_inds)

        # todo dior: cls_scores [8,[13343, 20]] bbox_preds [8,[13343, 4]] 一张照片上所有尺度特征图的点的分类和回归结果。
        # todo labels_old, 所有图片所有尺度全部展开
        # todo 得到所需要的正样本的分类分数以及回归分数,从这里开始的所有的bath的样本的全部被拼接在一起！！！！！！！
        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        bbox_preds = torch.cat(bbox_preds, 0).view(-1, bbox_preds[0].size(-1))
        # todo labels_old, pos_inds_flatten_old
        # todo 获取所有铺平之后的anchor
        flatten_anchors = torch.cat(
            [torch.cat(item, 0) for item in anchor_list])
        bboxes_target = torch.cat(bbox_targets_list,
                                  0).view(-1, bbox_targets_list[0].size(-1))

        # todo 计算gt的数量
        gt_labels_flatten = torch.cat(gt_labels, 0).view(-1)
        # todo pos_gt_index是每个样本对应的真值的序号, 将每个样本的对应的标签获取出来
        # print(pos_gt_index)
        # print(gt_labels)
        trans_all = []
        for i in range(len(pos_gt_index)):
            trans = gt_labels[i][pos_gt_index[i]]
            trans_all.append(trans)
        trans_all_new = torch.cat(trans_all, 0).view(-1)
        # todo 为每一个样本的生成一个序号的，代表他是某一张图片的
        batch_index_all = []
        for j in range(len(test)):
            batch_index = torch.ones([test[j].size(0)], dtype=torch.int64) * (j + 1)
            batch_index_all.append(batch_index)
        batch_index_all = torch.cat(batch_index_all, 0).view(-1)
        # todo 获取所有的样本对应的真值序号
        batch_gt_all = []
        for i in range(len(test)):
            batch_gt_index = (torch.ones([test[j].size(0)], dtype=torch.int64) * 1000).cuda()
            batch_gt_index[pos_inds[i]] = pos_gt_index[i]
            batch_gt_all.append(batch_gt_index)
        batch_gt_all = torch.cat(batch_gt_all, 0).view(-1)

        # todo 获得所有的正样本的标签，正样本的分类的结果, 对应的图片，对应的真值的索引
        pos_labels = labels_old[pos_inds_flatten_old]
        pos_cls = cls_scores[pos_inds_flatten_old]
        pos_index = batch_index_all[pos_inds_flatten_old]
        pos_gt = batch_gt_all[pos_inds_flatten_old]
        # todo 只有正样本时才得到回归的样本的值
        if len(pos_labels):
            # todo 得到解码后的正样本
            pos_bbox_pred = self.bbox_coder.decode(
                flatten_anchors[pos_inds_flatten_old],
                bbox_preds[pos_inds_flatten_old])
            pos_bbox_target = self.bbox_coder.decode(
                flatten_anchors[pos_inds_flatten_old],
                bboxes_target[pos_inds_flatten_old]
            )

            # todo 计算定位校准函数, 计算预测后的偏差程度
            pos_anchors = flatten_anchors[pos_inds_flatten_old]
            pos_bp = bbox_preds[pos_inds_flatten_old]
            location_c = self.LC_target(pos_anchors, pos_bp)

            # todo 计算的所有的正样本与真值的之间的iou值
            iou_bbox = bbox_overlaps(
                pos_bbox_pred.detach(), pos_bbox_target, is_aligned=True)

            # todo 获取分离的 cls 和 reg 分数
            pos_cls_loss, pos_reg_loss = self.get_cr_loss(pos_cls, pos_bbox_pred, pos_labels,
                                                          pos_bbox_target, pos_inds_flatten_old)
        else:
            # todo 如果一个样本都没有就虚构一个，不参与计算
            pos_bbox_pred = torch.zeros([1, 4], dtype=torch.int64)
            pos_bbox_target = torch.ones([1, 4], dtype=torch.int64)
            iou_bbox = torch.zeros([1], dtype=torch.int64)
            location_c = torch.ones_like(pos_bbox_pred)
            # todo 获取分离的 cls 和 reg 分数
            pos_cls_loss = torch.tensor([0.0]).cuda()
            pos_reg_loss = torch.tensor([0.0]).cuda()

        return pos_cls, pos_bbox_pred, pos_labels, pos_index, pos_gt, \
               pos_cls_loss, pos_reg_loss, pos_inds_flatten_old, pos_bbox_target, gt_labels_flatten, location_c, \
               trans_all_new

    # todo 导出教师
    def loss_tea(self,
                 cls_scores,
                 bbox_preds,
                 centernesses,
                 gt_bboxes,
                 gt_labels,
                 img_metas,
                 pos_inds_flatten,
                 pos_labels,
                 pos_bbox_targets,
                 gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        # todo 获取教师网路的训练目标
        cls_reg_targets = self.get_paa_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
        )

        if cls_reg_targets is None:
            return None

        # todo pos_inds 和 pos_gt_index是对应的
        (labels, labels_weight, bboxes_target, bboxes_weight, pos_inds,
         pos_gt_index) = cls_reg_targets

        # todo 计算老师的的正样本
        labels_teacher = torch.cat(labels, 0).view(-1)
        pos_inds_flatten_teacher = ((labels_teacher >= 0) & (labels_teacher < self.num_classes)).nonzero().reshape(-1)

        # todo 将每个level上面的的分数转换为图片上, 此处拿到的还是按照每个batch的分批次的结果，此处仍然是一个的list
        cls_scores = levels_to_images(cls_scores)
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores
        ]
        bbox_preds = levels_to_images(bbox_preds)
        bbox_preds = [item.reshape(-1, 4) for item in bbox_preds]

        # todo 得到所需要的正样本的分类分数以及回归分数,从这里开始的所有的bath的样本的全部被拼接在一起！！！！！！！
        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        bbox_preds = torch.cat(bbox_preds, 0).view(-1, bbox_preds[0].size(-1))
        bboxes_target = torch.cat(bboxes_target, 0).view(-1, bboxes_target[0].size(-1))

        # todo 获取教师模型中的分类以及对应回归结果
        pos_cls_scores = cls_scores[pos_inds_flatten]
        pos_bbox_preds = bbox_preds[pos_inds_flatten]

        # todo 教师拿到的标签和目标都是来自学生网络生成的结果, 回归框为已经解码过
        pos_labels = pos_labels
        pos_bboxes_target = pos_bbox_targets

        # todo 所有的anchor
        flatten_anchors = torch.cat(
            [torch.cat(item, 0) for item in anchor_list])

        # todo 将正样本的回归结果进行编码
        if len(flatten_anchors):
            pos_bbox_pred = self.bbox_coder.decode(flatten_anchors[pos_inds_flatten],
                                                   pos_bbox_preds)
            pos_cls_loss, pos_reg_loss = self.get_cr_loss(pos_cls_scores, pos_bbox_pred, pos_labels,
                                                          pos_bboxes_target, pos_inds_flatten)
            # todo 计算定位校准函数, 计算预测后的偏差程度
            pos_anchors = flatten_anchors[pos_inds_flatten]
            pos_bp = bbox_preds[pos_inds_flatten]
            location_c = self.LC_target(pos_anchors, pos_bp)

        else:
            pos_bbox_pred = torch.zeros([1, 4], dtype=torch.int64)
            pos_cls_loss = torch.tensor([0.0]).cuda()
            pos_reg_loss = torch.tensor([0.0]).cuda()

        # todo 1\ 真样本的分类结果， 回归结果， 每个样本的label,
        return pos_cls_scores, pos_bbox_pred, pos_cls_loss, pos_reg_loss, location_c

    # todo 获取样本的损失******************************************************
    def get_pos_loss(self, anchors, cls_score, bbox_pred, label, label_weight,
                     bbox_target, bbox_weight, pos_inds):
        if not len(pos_inds):
            return cls_score.new([]),
        anchors_all_level = torch.cat(anchors, 0)
        # todo 获取所有的正样本的分数
        pos_scores = cls_score[pos_inds]
        pos_bbox_pred = bbox_pred[pos_inds]
        pos_label = label[pos_inds]
        pos_label_weight = label_weight[pos_inds]
        pos_bbox_target = bbox_target[pos_inds]
        pos_bbox_weight = bbox_weight[pos_inds]
        # todo 获取所有的正样本的anchor
        pos_anchors = anchors_all_level[pos_inds]
        pos_bbox_pred = self.bbox_coder.decode(pos_anchors, pos_bbox_pred)
        # todo 此处需要重新编码, 否则计算的 iou 是有问题的。
        pos_bbox_target = self.bbox_coder.decode(pos_anchors, pos_bbox_target)
        # to keep loss dimension
        # todo 计算所有的分类损失
        loss_cls = self.loss_cls(
            pos_scores,
            pos_label,
            pos_label_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')
        # todo 计算所有的回归的损失
        loss_bbox = self.loss_bbox(
            pos_bbox_pred,
            pos_bbox_target,
            pos_bbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')
        loss_cls = loss_cls.sum(-1)
        pos_loss = loss_bbox + loss_cls
        return pos_loss,

    def get_cr_loss(self, cls_score, bbox_pred, label,
                    bbox_target, pos_inds):
        if not len(pos_inds):
            return cls_score.new([]),
        # todo 获取所有的正样本的分数
        pos_scores = cls_score
        pos_bbox_pred = bbox_pred
        pos_label = label
        pos_label_weight = torch.ones((len(cls_score), 1)).cuda()
        pos_bbox_target = bbox_target
        pos_bbox_weight = torch.ones((len(bbox_pred), 4)).cuda()
        # todo 计算所有的分类损失
        loss_cls = self.loss_cls(
            pos_scores,
            pos_label,
            pos_label_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')
        # todo 计算所有的回归的损失
        loss_bbox = self.loss_bbox(
            pos_bbox_pred,
            pos_bbox_target,
            pos_bbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        loss_cls = loss_cls.sum(-1)
        return loss_cls, loss_bbox

    # todo 为样本设置训练目标***************************************************
    def get_paa_targets(self,
                        anchor_list,
                        valid_flag_list,
                        gt_bboxes_list,
                        img_metas,
                        gt_bboxes_ignore_list=None,
                        gt_labels_list=None,
                        label_channels=1,
                        unmap_outputs=True):

        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list, sampling_result) = multi_apply(
            self._get_paa_target_single,
            concat_anchor_list,
            concat_valid_flag_list,
            num_level_anchors_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None

        # todo 得到所有正样本的编号
        pos_inds = []
        for i, single_labels in enumerate(all_labels):
            pos_mask = (0 <= single_labels) & (
                    single_labels < self.num_classes)
            pos_inds.append(pos_mask.nonzero().view(-1))

        gt_inds = [item.pos_assigned_gt_inds for item in sampling_result]

        # todo 不映射回去
        return (all_labels, all_label_weights,
                all_bbox_targets, all_bbox_weights, pos_inds, gt_inds)

    def _get_paa_target_single(self,
                               flat_anchors,
                               valid_flags,
                               num_level_anchors,
                               gt_bboxes,
                               gt_bboxes_ignore,
                               gt_labels,
                               img_meta,
                               label_channels=1,
                               unmap_outputs=True):

        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,),
                                  self.background_label,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds, sampling_result)

    # todo paa reasign***********************************************
    # def paa_reassign(self, pos_losses, label, label_weight, bbox_weight,
    #                  pos_inds, pos_gt_inds, anchors):
    #     """Fit loss to GMM distribution and separate positive, ignore, negative
    #     samples again with GMM model.
    #
    #     Args:
    #         pos_losses (Tensor): Losses of all positive samples in
    #             single image.
    #         label (Tensor): classification target of each anchor with
    #             shape (num_anchors,)
    #         label_weight (Tensor): Classification loss weight of each
    #             anchor with shape (num_anchors).
    #         bbox_weight (Tensor): Bbox weight of each anchor with shape
    #             (num_anchors, 4).
    #         pos_inds (Tensor): Index of all positive samples got from
    #             first assign process.
    #         pos_gt_inds (Tensor): Gt_index of all positive samples got
    #             from first assign process.
    #         anchors (list[Tensor]): Anchors of each scale.
    #
    #     Returns:
    #         tuple: Usually returns a tuple containing learning targets.
    #
    #             - label (Tensor): classification target of each anchor after
    #               paa assign, with shape (num_anchors,)
    #             - label_weight (Tensor): Classification loss weight of each
    #               anchor after paa assign, with shape (num_anchors).
    #             - bbox_weight (Tensor): Bbox weight of each anchor with shape
    #               (num_anchors, 4).
    #             - num_pos (int): The number of positive samples after paa
    #               assign.
    #     """
    #     if not len(pos_inds):
    #         return label, label_weight, bbox_weight, 0
    #     # todo 前期数量统计
    #     label = label.clone()
    #     label_weight = label_weight.clone()
    #     bbox_weight = bbox_weight.clone()
    #     # todo 真值的数量
    #     num_gt = pos_gt_inds.max() + 1
    #     # todo 层级的数量
    #     num_level = len(anchors)
    #     num_anchors_each_level = [item.size(0) for item in anchors]
    #     num_anchors_each_level.insert(0, 0)
    #     # todo 不断叠加，返回元素的梯形累计和, 等效于不断返回anchor的数量叠加
    #     inds_level_interval = np.cumsum(num_anchors_each_level)
    #     # todo 准备每个层级的正样本mask, 根据上一笔划分的每个层级的anchor数量，利用mask将所有的正样本序号分配
    #     # todo
    #     pos_level_mask = []
    #     for i in range(num_level):
    #         mask = (pos_inds >= inds_level_interval[i]) & (
    #                 pos_inds < inds_level_interval[i + 1])
    #         pos_level_mask.append(mask)
    #     # todo paa之后的pos_inds
    #     pos_inds_after_paa = [label.new_tensor([])]
    #     # todo paa之后的忽略的样本
    #     ignore_inds_after_paa = [label.new_tensor([])]
    #     # todo 根据每个真值来计算样本的分配, 由于topk设置为9，所以说为每个gt在每个层级匹配9个样本
    #     for gt_ind in range(num_gt):
    #         pos_inds_gmm = []
    #         pos_loss_gmm = []
    #         # todo pos_gt_inds = 所有的样本中的为正样本的位置为对应真值的号码， gt_ind真值的编号
    #         # todo gt_mask = 从所有的真值中筛选出和每一个真值对应的所有正样本
    #         gt_mask = pos_gt_inds == gt_ind
    #         # print('******')
    #         for level in range(num_level):
    #             level_mask = pos_level_mask[level]
    #             # todo 提取每个层级中某个真值对应的所有
    #             #  真样本
    #             level_gt_mask = level_mask & gt_mask
    #             # todo 这个地方应该是每个层级挑选了前9个样本（得分最小的，损失最少，说明匹配最好）
    #             value, topk_inds = pos_losses[level_gt_mask].topk(
    #                 min(level_gt_mask.sum(), self.topk), largest=False)
    #             # todo 异常结果超出， 导致 top_k 出现特别大的数字。通过此方式规避掉
    #             # if len(topk_inds) > 0:
    #             #     print(topk_inds.max())
    #             #     print(topk_inds.min())
    #             if len(topk_inds) > 0 and topk_inds.max() > min(level_gt_mask.sum(), self.topk):
    #                 continue
    #             # todo 为每个层级添加参与gmm的正样本的编号
    #             pos_inds_gmm.append(pos_inds[level_gt_mask][topk_inds])
    #             # todo 为每个层级添加参与gmm的正样本的值
    #             pos_loss_gmm.append(value)
    #         # todo 把一个list拼接起来
    #         pos_inds_gmm = torch.cat(pos_inds_gmm)
    #         pos_loss_gmm = torch.cat(pos_loss_gmm)
    #         # fix gmm need at least two sample
    #         # todo gmm至少需要两个样本
    #         if len(pos_inds_gmm) < 2:
    #             continue
    #         device = pos_inds_gmm.device
    #         # todo 此处获得了 pos_loss_gmm 重新按照顺序进行排序， 默认是按照升序排列，从小到大，
    #         # todo 原本是分数越高，约为正样本，在原始论文中的是按照损失来计算，损失越小约为正样本,
    #         pos_loss_gmm, sort_inds = pos_loss_gmm.sort()
    #         # todo 所以按照重新排序后来说，排在前面的为正样本
    #         pos_inds_gmm = pos_inds_gmm[sort_inds]
    #         # todo pos_loss_gmm的数量不会超过45, level 5 * top 9  = 45
    #         pos_loss_gmm = pos_loss_gmm.view(-1, 1).cpu().numpy()
    #         # todo 获取其中损失的最小值以及最大值
    #         min_loss, max_loss = pos_loss_gmm.min(), pos_loss_gmm.max()
    #         # todo 初始化高斯混合估计的参数
    #         means_init = np.array([min_loss, max_loss]).reshape(2, 1)
    #         weights_init = np.array([0.5, 0.5])
    #         precisions_init = np.array([1.0, 1.0]).reshape(2, 1, 1)  # full
    #         # todo 设置斜方差矩阵 默认为diag
    #         if self.covariance_type == 'spherical':
    #             precisions_init = precisions_init.reshape(2)
    #         elif self.covariance_type == 'diag':
    #             precisions_init = precisions_init.reshape(2, 1)
    #         elif self.covariance_type == 'tied':
    #             precisions_init = np.array([[1.0]])
    #         if skm is None:
    #             raise ImportError('Please run "pip install sklearn" '
    #                               'to install sklearn first.')
    #         # todo 设置高斯混合估计模型, 分为两类，正负样本
    #         # todo 收敛阈值为0.001， 最大迭代次数为100
    #         gmm = skm.GaussianMixture(
    #             2,
    #             weights_init=weights_init,
    #             means_init=means_init,
    #             precisions_init=precisions_init,
    #             covariance_type=self.covariance_type,
    #             reg_covar=1e-5
    #         )
    #         # todo 规避出现nan的情况
    #         if np.isnan(pos_loss_gmm).any():
    #             print(pos_loss_gmm)
    #             continue
    #         # todo 使用EM算法估算模型参数
    #         gmm.fit(pos_loss_gmm)
    #         # todo 使用估计好的模型来预测 数据样本的标签, 0和1分别代表两个类别
    #         gmm_assignment = gmm.predict(pos_loss_gmm)
    #         # todo 计算每个样本的加权对数的概率
    #         scores = gmm.score_samples(pos_loss_gmm)
    #         # # todo 构造BGM函数
    #         # bgm = skm.BayesianGaussianMixture(
    #         #     n_components=2,
    #         #     init_params='random',
    #         #     covariance_type=self.covariance_type,
    #         #     # random_state=42,
    #         # )
    #         # # todo 使用EM算法估算模型参数
    #         # bgm.fit(pos_loss_gmm)
    #         # # todo 使用估计好的模型来预测 数据样本的标签, 0和1分别代表两个类别
    #         # gmm_assignment = bgm.predict(pos_loss_gmm)
    #         # # todo 计算每个样本的加权对数的概率
    #         # scores = bgm.score_samples(pos_loss_gmm)
    #
    #         # todo 将numpy转换为tensor
    #         gmm_assignment = torch.from_numpy(gmm_assignment).to(device)
    #         scores = torch.from_numpy(scores).to(device)
    #         # todo 根据高斯分布将数据分为两个个部分，正样本和忽略的赝本
    #         pos_inds_temp, ignore_inds_temp = self.gmm_separation_scheme(
    #             gmm_assignment, scores, pos_inds_gmm)
    #         # todo 添加新的已经分配完成的样本
    #         pos_inds_after_paa.append(pos_inds_temp)
    #         ignore_inds_after_paa.append(ignore_inds_temp)
    #     # todo 获取一个图片上面所有的正样本，以及不参与训练的样本
    #     pos_inds_after_paa = torch.cat(pos_inds_after_paa)
    #     ignore_inds_after_paa = torch.cat(ignore_inds_after_paa)
    #     # todo 对比原本正样本的序号和重新分配之后的序号，如果不相同，则得到剔除的正样本
    #     reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_paa).all(1)
    #     # todo 获取重新分配的样本的id
    #     reassign_ids = pos_inds[reassign_mask]
    #     # todo 将所有的剔除的正样本的标签设置为背景
    #     label[reassign_ids] = self.num_classes
    #     # todo 被忽略的样本的权重设为0， 即不参与训练
    #     label_weight[ignore_inds_after_paa] = 0
    #     # todo 将重新分配后的与之前的进行对比，凡是没有被分配的pos，这次权重全部重新置0
    #     bbox_weight[reassign_ids] = 0
    #     num_pos = len(pos_inds_after_paa)
    #     return label, label_weight, bbox_weight, num_pos
    #
    # def gmm_separation_scheme(self, gmm_assignment, scores, pos_inds_gmm):
    #     """A general separation scheme for gmm model.
    #
    #     It separates a GMM distribution of candidate samples into three
    #     parts, 0 1 and uncertain areas, and you can implement other
    #     separation schemes by rewriting this function.
    #
    #     Args:
    #         gmm_assignment (Tensor): The prediction of GMM which is of shape
    #             (num_samples,). The 0/1 value indicates the distribution
    #             that each sample comes from.
    #         scores (Tensor): The probability of sample coming from the
    #             fit GMM distribution. The tensor is of shape (num_samples,).
    #         pos_inds_gmm (Tensor): All the indexes of samples which are used
    #             to fit GMM model. The tensor is of shape (num_samples,)
    #
    #     Returns:
    #         tuple[Tensor]: The indices of positive and ignored samples.
    #
    #             - pos_inds_temp (Tensor): Indices of positive samples.
    #             - ignore_inds_temp (Tensor): Indices of ignore samples.
    #     """
    #     # The implementation is (c) in Fig.3 in origin paper instead of (b).
    #     # todo 这个地方说明，没有被忽略的样本， 样本被分为正样本和负样本
    #     # You can refer to issues such as
    #     # https://github.com/kkhoot/PAA/issues/8 and
    #     # https://github.com/kkhoot/PAA/issues/9.
    #     # todo score 是一个根据高斯模型预测出来的每个anchor的概率
    #     # todo gmm_ass = [num] scores = [num]
    #     # todo 首先把assignment=0 的找出来 fgs应该是前景, 最多是45个
    #     fgs = gmm_assignment == 0
    #     # todo 设置等待分配的正样本
    #     pos_inds_temp = fgs.new_tensor([], dtype=torch.long)
    #     ignore_inds_temp = fgs.new_tensor([], dtype=torch.long)
    #     # todo numel() 返回一个矩阵中元素的个数 nonzero()返回的是数组中非零元素的位置
    #     # todo 当有正样本的时候进行分配
    #     if fgs.nonzero().numel():
    #         # todo 从前景中选择得分最高的一个作为阈值,并返回它的索引。得分最高的
    #         _, pos_thr_ind = scores[fgs].topk(1)
    #         # todo 将前景的列表设置 ，pos_inds_gmm:正样本的编号， 先从正样本的中筛选出前景， 再从所有的前景中筛选出从零到阈值的样本
    #         # todo 由于是按照损失的排序，损失小的都在前面，即为正样本。
    #         pos_inds_temp = pos_inds_gmm[fgs][:pos_thr_ind + 1]
    #         # todo 将忽略的样本的列表设置为空
    #         ignore_inds_temp = pos_inds_gmm.new_tensor([])
    #     return pos_inds_temp, ignore_inds_temp

