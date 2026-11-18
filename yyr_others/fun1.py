# from mmdet.models import ResNet
# from mmdet.models import FPN
# import torch
import numpy as np


# def giou_loss(pred, target, eps=1e-7):
#     r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
#     Box Regression <https://arxiv.org/abs/1902.09630>`_.
#
#     Args:
#         pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
#             shape (n, 4).
#         target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
#         eps (float): Eps to avoid log(0).
#
#     Return:
#         Tensor: Loss tensor.
#     """
#     # overlap
#     lt = torch.max(pred[:, :2], target[:, :2])
#     rb = torch.min(pred[:, 2:], target[:, 2:])
#     wh = (rb - lt).clamp(min=0)
#     overlap = wh[:, 0] * wh[:, 1]
#
#     # union
#     ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
#     ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
#     union = ap + ag - overlap + eps
#
#     # IoU
#     ious = overlap / union
#
#     # enclose area
#     enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
#     enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
#     enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
#     enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + eps
#
#     # GIoU
#     gious = ious - (enclose_area - union) / enclose_area
#     print(gious)
#     loss = 1 - gious
#     return loss
#
#
# pred = torch.tensor(((447, 136, 833, 515), (234, 571, 657, 1014), (563, 388, 890, 429)))
# gt = torch.tensor(((395, 8, 882, 515),(183, 453, 718, 949), (674, 334, 802, 461)))
#
# loss = giou_loss(pred, gt)
# print(loss)

# self=ResNet(depth=50)
# self.eval()
# inputs = torch.rand(1, 3, 224, 224)
# level_outputs = self.forward(inputs)
# for level_output in level_outputs:
#      print(tuple(level_output.shape))
# print("@@@@@@@@@@@@@@@@@@@@@")
# in_channels = [256, 512, 1024,2048]
# ha = FPN(in_channels,11,len(in_channels)).eval()
# outputs = ha.forward(level_outputs)
# for output in outputs:
#      print(tuple(output.shape))

# import numpy as np
# # overlap = torch.rand(4,3)
# # print(overlap.size(-1))
# # print(overlap)
# # a,b =overlap.max(dim=0)
# # print("%%%%%%%%%")
# # print(a,b)
# # print("%%%%%%%%%")
# # assign = overlap.new_full((3,),-1,dtype=torch.long)
# # print(assign)
# # print("%%%%%%%%%")
# #
# # pos = a >= 0.7
# # print("pos", pos)
# # assign[pos] = b[pos] + 1
# # print(assign)
# # print(b[pos] + 1)
#
# # import torch
# #
# # from mmdet.utils import util_mixins
# # from mmdet.core.bbox.assigners import AssignResult
# # num_gts = 9
# # max_overlaps = torch.LongTensor([0, .5, .9, 0])
# # gt_inds = torch.LongTensor([-1, 1, 2, 0])
# # labels = torch.LongTensor([0, 3, 4, 0])
# # self = AssignResult(num_gts, gt_inds, max_overlaps, labels)
# # print(self)  # xdoctest: +IGNORE_WANT
#
# import torch
# # iou=torch.rand(2,2,3,4)
# # B=torch.sum(iou,dim=2)
# # print(iou)
# # print('***')
# # print(B)
# # iou1= torch.rand(2,4,3,2,2)
# # max_iou = torch.tensor([0.5, 0.6])
# # print(iou1)
# # print("^^^")
# # print(iou1[0])
# # #print(max_iou.shape[1])
# #
# # for k in range(4):
# #      mask_per_gt = torch.sum(iou1[0][:, :, :, k] > max_iou, dim=2)
# #
# #      print(mask_per_gt)
#
# # b= 2**9
# # print(b)
#
# import torch
#
# # a=torch.rand(4,3)
# # c,d= torch.max(a,dim=1)
# # print(a)
# # print("8**")
# # print(c)
# # print("&&&&&&")
# # print(d)
# #
# # targets = torch.ones(4,3) * -1
# # targets[torch.lt(c, 0.8), :] = 0
# # print(targets)
# # pos = torch.ge(c,0.5)
# # print(pos)
# # h=pos.sum()
# # print(h)
#
# import torch
# a=torch.tensor([[0.23,0.56],[0.56,0.79]])
# b=torch.tensor([[0.96,0.13],[0.89,0.36]])
# c = torch.cat((a,b),0)
# print(c)
#
# A=[[1,2,3],[1,2,2]]
# # print(len(A))
# import torch
# A = torch.ones
# import cv2
# import matplotlib.pyplot as plt
#
# plt.subplot(1, 6, 1)
# plt.imshow()

# import torch
# import torch.nn.functional as F
# A= torch.rand(3,4)
# B = A.reshape(-1, 1)
# C = F.softmax(A)
# print(A)
# print(B)
# print(C)

# A = 0
# B = 0
# C = 0
#
# if A:
#     print("a")
# elif B:
#     print("b")
# else:
#     print("C")
# import numpy as np
# mask0 = np.ones((3,3))
# mask1 = np.zeros((3,3))
# mask2 = np.eye(3)
# # print(mask2)
# num = 0
# mask = mask0
# print(mask)


# A = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
# print(A)
# print(A.shape)
#
#
# B = A.unsqueeze(0)
# # B = torch.cat(A, 0)
# print(B)
# print(B.shape)
# # D = B.transpose(0,1)
# # print(D.shape)
# C = B.repeat((len(A), 1, 1), 0)
# print(C)
# print(C.shape)
# print(A.unsqueeze(1)-C)
A = [0.4711964549483013,
0.48850881881346875,
0.5089309517462011,
0.4876847290640394,
0.5196629213483146,
0.4830324909747292,
0.5120614035087719,
0.489593657086224,
0.5004405286343613,
0.462738301559792,
0.5,
0.4934210526315789,
0.4485049833887043,
0.4631268436578171,
0.4964705882352941,
0.49733333333333335,
0.5376344086021505,
0.5050784856879039,
0.49302915082382764,
0.5,
0.4609250398724083,
0.5269823788546255,
0.4629418472063854,
0.5051724137931034,
0.5253878702397743,
0.513177159590044,
0.5152526215443279,
0.44587628865979384,
0.484984984984985,
0.45650301464254955,
0.47388059701492535,
0.4904109589041096,
0.4821615949632739,
0.5237209302325582,
0.5180180180180181,
0.46017699115044247,
0.517426273458445,
0.5526189332051898,
0.5391498881431768,
0.5128205128205128,
0.5013368983957219,
0.48128342245989303,
0.5102803738317757,
0.5,
0.48328267477203646,
0.5016949152542373,
0.4896449704142012,
0.5430232558139535,
0.552,0.5218068535825545]
average = np.mean(A)
print(average)