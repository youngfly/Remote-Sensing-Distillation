import torch
from mmdet.models import build_detector
from mmdet.models import build_loss
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmdet.core import bbox_overlaps
from ..builder import DETECTORS, LOSSES
from .atss import ATSS
from .single_stage import SingleStageDetector
import cv2
import matplotlib.pyplot as plt
import matplotlib
from mmcv.cnn import normal_init
import torch.nn as nn
import torch.nn.functional as F
import mmcv
import numpy as np
import os
from mmdet.core import bbox2result

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
class DIL_ATSS(ATSS):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 tea_model,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 tea_pretrained=None,
                 student2=None,
                 ):
        super(DIL_ATSS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                       test_cfg, pretrained)
        # todo 构建教师模型
        self.tea_model = build_detector(tea_model, train_cfg=train_cfg, test_cfg=test_cfg)
        # todo 加载教师模型的权重---全部权重
        self.load_weights(pretrained=tea_pretrained, student2=student2)
        # todo 教师模型仅仅加载backbone
        # self.freeze_models()

        # todo 单独测试或者全部测试
        self.only_use_student = True

        # todo 采用giou作为回归的损失函数
        loss_dis_reg_cfg = dict(type='NIoULoss', loss_weight=2.0)
        self.loss_distill_reg = build_loss(loss_dis_reg_cfg)
        # todo 针对学生的fpn的特征层的尺度变为原来的一半，此处从configs获取学生fpn通道数
        self.in_channels_adapt = bbox_head.in_channels
        self.out_channels = 256
        self.adapt_conv = nn.Conv2d(self.in_channels_adapt, self.out_channels, 1, padding=0)
        # todo 初始化适应层的权重
        normal_init(self.adapt_conv, std=0.01)

    # todo 冻结教师模型
    def freeze_models(self):
        self.tea_model.eval()
        for param in self.tea_model.parameters():
            param.requires_grad = False

    # todo 载入模型
    def load_weights(self, pretrained=None, student2=None):
        # todo 同时加载两个未训练学生模型
        if student2:
            print("begin load the second student model")
            self.tea_model.backbone.init_weights(pretrained=pretrained)
            print("load the second student model success")
        # todo 加载教师模型
        else:
            if isinstance(pretrained, str):
                print("begin load the teacher model")
                logger = get_root_logger()
                load_checkpoint(self.tea_model, pretrained, strict=False, logger=logger)
                print("load teacher model success")
            else:
                raise TypeError('pretrained must be a str')

    # todo 前向训练
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        # todo 获取教师网络的输出
        # self.tea_model.eval()
        # with torch.no_grad():
        tea_x = self.tea_model.extract_feat(img)
        tea_outs = self.tea_model.bbox_head(tea_x)
        tea_loss_inputs = tea_outs + (gt_bboxes, gt_labels, img_metas)
        tea_losses = self.tea_model.bbox_head.loss(*tea_loss_inputs, name='teacher', gt_bboxes_ignore=gt_bboxes_ignore)

        # todo 获取学生的网络的输出
        stu_x = self.extract_feat(img)
        stu_outs = self.bbox_head(stu_x)
        loss_inputs = stu_outs + (gt_bboxes, gt_labels, img_metas,)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(tea_losses)

        # todo stu_x=[5,[n,c,h,w]]
        # todo stu_out=[3,[5,[n,c,h,w]]]
        # todo stu_out[0]分类， stu_outs[1]回归, stu_outs[2]centness
        # todo tupe (3, 5, n, c, h ,w), 3代表三个输出的分支头， 5代表5个尺度，n代表batch_size,
        # todo 分类头的c=15, 代表类别， 检测头c=4, 代表4个偏移量， 点头c=1代表， 代表一个中心点
        # todo [h,w] 128, 64, 32 ,16, 8

        # todo 学生网络的各种计算结果
        pos_cls_stu, pos_reg_stu, pos_labels, \
        pos_index, pos_gt, \
        cls_loss_stu, reg_loss_stu, \
        pos_inds_flatten_old, \
        pos_bbox_targets, gt_labels_faltten, lc_stu, trans_all \
            = self.bbox_head.loss_stu(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # todo 教师网络的计算结果
        pos_cls_tea, pos_reg_tea, \
        cls_loss_tea, reg_loss_tea, lc_tea\
            = self.bbox_head.loss_tea(*tea_loss_inputs, pos_inds_flatten_old,
                                                             pos_labels, pos_bbox_targets,
                                                             gt_bboxes_ignore=gt_bboxes_ignore)

        # todo code检测，保证教师和学生的预测可以对齐, 以及有结果
        assert len(pos_reg_tea) == len(pos_reg_stu)
        assert len(pos_cls_tea) == len(pos_cls_stu)
        assert len(pos_reg_stu) > 0
        assert len(pos_cls_stu) > 0

        # todo IIL module *********************
        # todo 实例损失比较
        loss_stu = cls_loss_stu + reg_loss_stu
        loss_tea = reg_loss_tea + reg_loss_tea
        # todo 学生需要从教师网络学习实例
        stu_mask = loss_tea < loss_stu
        # todo 教师需要从学生网络学习实例
        tea_mask = loss_stu < loss_tea

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

        # todo 获取展平的正样本特征
        ft_all = ft_flatten[pos_inds_flatten_old]
        fs_all = fs_flatten[pos_inds_flatten_old]
        fs_ad_all = fs_adapt_fla[pos_inds_flatten_old]

        # todo IPP module ******************************************
        # todo feature enhancement 特征增强   ***********
        # todo sum
        ft_attention = torch.sum(ft_all, dim=1)
        fs_attention = torch.sum(fs_ad_all,dim=1)
        # todo max
        # ft_attention, _ = torch.max(ft_all, dim=1)
        # fs_attention, _ = torch.max(fs_ad_all, dim=1)
        # todo get the weight
        FE_attention = torch.abs(torch.pow(ft_attention - fs_attention, 1))
        FE = FE_attention * stu_mask

        # todo category balance   类别平衡   **************
        # todo 维护一个各个元素出现的次数的字典
        fre = {}
        for i in range(len(gt_labels_faltten)):
             fre["{}".format(gt_labels_faltten[i])] = 0
        for i in range(len(gt_labels_faltten)):
             fre["{}".format(gt_labels_faltten[i])] +=1
        trans_all_cpu = trans_all.cpu().numpy()
        # todo 统计每个样本属于哪个真值
        for i in range(len(trans_all_cpu)):
            trans_all_cpu[i] = fre[str(trans_all_cpu[i])]
        trans_all_gpu = torch.from_numpy(trans_all_cpu).cuda()
        # todo 计算atte
        cls_max, _ = torch.max(pos_cls_stu.sigmoid(), 1)
        CB_attention = 1 - trans_all_gpu/len(gt_labels_faltten)*cls_max
        # todo get the weight
        CB = CB_attention * stu_mask

        # todo location calibration 定位校准   **********
        LC_attention = torch.abs(lc_stu-lc_tea)
        LC = LC_attention * stu_mask

        # todo 知识蒸馏**************************************************
        # todo 特征学习******
        # todo stu
        # sup_stu_fea = self.dis_fea_ada(ft_all, fs_ad_all, stu_mask)
        # sup_stu_fea = self.dis_fea_ada(ft_all, fs_ad_all, FE)
        # losses.update(dict(loss_stu_fea=sup_stu_fea))

        # todo tea
        # sup_tea_fea = self.dis_fea_ada(ft_all, fs_ad_all, tea_mask)
        # losses.update(dict(loss_tea_fea=sup_tea_fea))

        # todo 类别学习******
        # todo stu
        # sup_stu_cls = self.dis_cls(stu_mask, pos_cls_stu, pos_cls_tea)
        sup_stu_cls = self.dis_cls(CB, pos_cls_stu, pos_cls_tea)
        losses.update(dict(loss_stu_cls= sup_stu_cls))

        # todo tea
        # sup_tea_cls = self.dis_cls(tea_mask, pos_cls_tea, pos_cls_stu)
        # losses.update(dict(loss_tea_cls= sup_tea_cls))
        # print(sup_stu_cls)

        # # todo 定位学习******
        # todo stu
        # sup_stu_reg = self.dis_reg(stu_mask, pos_reg_stu, pos_reg_tea)
        # sup_stu_reg = self.dis_reg(LC, pos_reg_stu, pos_reg_tea)
        # losses.update(dict(loss_stu_reg=sup_stu_reg))

        # todo tea
        # sup_tea_reg = self.dis_reg(tea_mask, pos_reg_stu, pos_reg_tea)
        # losses.update(dict(loss_tea_reg=sup_tea_reg))
        # todo 知识蒸馏 **************************************************

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

    # todo 选择测试教师还是学生网络
    def simple_test(self, img, img_metas, rescale=False):
        # todo use student
        if self.only_use_student:
            x = self.extract_feat(img)
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                    *outs, img_metas, rescale=rescale)
            bbox_results = [
                    bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                    for det_bboxes, det_labels in bbox_list]
        else:
            x = self.tea_model.extract_feat(img)
            outs = self.tea_model.bbox_head(x)
            bbox_list = self.tea_model.bbox_head.get_bboxes(
                    *outs, img_metas, rescale=rescale)
            bbox_results = [
                    bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                    for det_bboxes, det_labels in bbox_list]
        return bbox_results[0]

    # todo 蒸馏学习函数************************************************************
    def dis_cls(self, mask_A, pos_cls_stu, pos_cls_tea):
        loss_cls = torch.sqrt((torch.pow(pos_cls_stu/0.25 - pos_cls_tea/0.25, 2) * mask_A.unsqueeze(1)).sum())
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

    def dis_fea_ada(self, ft, fs, mask):
        loss_fea = torch.sqrt((torch.pow(ft - fs, 2) * mask.unsqueeze(1)).sum())
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
    # todo 生成文件夹
    def mkdir(self, path):
        isexists = os.path.exists(path)
        if not isexists:
            os.makedirs(path)
            print("file makes successfully")
        else:
            print("file is exist")

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
