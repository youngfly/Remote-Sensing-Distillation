# a_config 配置目录说明

本目录是遥感蒸馏研究的全部自定义训练/测试配置，共 **237 个 .py**，基于 mmdetection v2.3.0rc0（见 `mmdet/VERSION`）。
配套自定义代码位于 `mmdet/`（datasets / models / yyr_creation），根目录三份说明 txt 记录了服务器上的原始用法。

> 注意：本快照中的配置不能全部直接运行，存在缺失基座文件、checkpoint 路径失效、代码开关硬编码等问题，详见文末「已知问题」。

## 目录结构

```
a_config/
├── base_configs/    101 个  三数据集基线模型（教师/学生候选池）
│   ├── dior/         40 个  baseline_resnet / regnet / regnet_compose / resnet_mini / self_statistics
│   ├── dota/         29 个  baseline_resnet / regnet / resnet
│   └── nwpu/         32 个  baseline_resnet / regnet / resnet
├── paper1_ARSD/      80 个  ARSD 蒸馏（paper1）
│   ├── dior/         10 个  Tea_Stu（师生单独训练）+ distill
│   ├── dota/         16 个  teacher / student / distill
│   ├── nwpu/          9 个  Tea&Stu + distil
│   └── other/        45 个  消融与历史版本（见下文）
├── paper2_S3MKM/     36 个  S3MKM 蒸馏（paper2），按学生 backbone 分目录
│   ├── dior/         10 个  T_S_D_Res / T_S_D_Mob / T_S_D_Shu
│   ├── dota/         17 个  T_S_D_Res / T_S_D_Reg / T_S_D_Mob / T_S_D_Shu
│   └── nwpu/          9 个  T_S_D_Res / T_S_D_Mob / T_S_D_Shu
└── paper4_DIL/       20 个  DIL 蒸馏（paper4）
    ├── dior/         13 个  T_S_D_Res / T_S_D_Reg / T_S_D_Mob / T_S_D_Shu
    ├── dota/          2 个  仅基座 + R18 蒸馏（实验未完成）
    └── nwpu/          5 个  含 student2 双学生互学配置
```

## 命名约定

| 命名片段                              | 含义                                                                                                                         |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `atss18` / `atss50` / `atss101` | ATSS 检测器 + ResNet-18/50/101 backbone                                                                                      |
| `f0.5` / `fpn0.5`                 | FPN 输出通道 256 → 128（`f0.25` → 64），head 通道同步缩减                                                                |
| `bk0.5` / `bk0.25`                | backbone 通道减半 / 减至 1/4（`resnet_half.py` / `resnet_quarter.py`）                                                   |
| `_area`                             | 使用面积加权 head `ATSSHeadnew`（权重 `exp(-sqrt(area)/512)+1`）；无此后缀为普通 `ATSSHead`                            |
| `D_` 前缀                           | 蒸馏配置（paper2/paper4）                                                                                                    |
| `T_S_D_*`                           | Teacher / Student / Distill 三类配置按学生 backbone 分目录：Res=ResNet18、Mob=MobileNetV2、Shu=ShuffleNet、Reg=RegNetX-800MF |
| `1dsn.py` / `2dsn.py`             | 数据集 + 调度 + runtime 基座文件，供同目录配置 `_base_` 继承                                                               |
| `800MF` / `1.6GF` / `3.2GF`     | RegNetX 算力档位                                                                                                             |

## 数据集与训练超参

| 数据集      | 格式      | img_scale  | batch | epochs | lr (SGD) | 自定义数据集类                                                                        |
| ----------- | --------- | ---------- | ----- | ------ | -------- | ------------------------------------------------------------------------------------- |
| DIOR        | VOC/XML   | 800×800   | 8     | 24     | 0.005    | `mmdet/datasets/dior.py`（年份须为 2020，否则输出 VOC 原类别）                      |
| DOTA        | COCO json | 1024×1024 | 8     | 24–30 | 0.005    | `mmdet/datasets/dota.py`（大图裁剪/拼接流程见根目录 `说明dota.txt`）              |
| NWPU VHR-10 | COCO json | 1000×600  | 4     | 40     | 0.001    | `mmdet/datasets/nwpu.py`（逐类别 mAP 依赖 `mmdet/yyr_creation/pycocotools_yyr/`） |

特殊基线超参：YOLOF lr=0.12；Deformable-DETR AdamW 2e-4 / 50ep；Sparse R-CNN AdamW 2.5e-5 / 60ep；YOLOv3 273ep；CornerNet 210ep / 511×511；resnet_mini 100ep / bs4。

## 各块说明

### base_configs（基线池）

检测器覆盖：ATSS（含 DCN/CARAFE 变体）、RetinaNet（含 GHM）、FCOS、Faster/Cascade R-CNN、DetectoRS、PAA、GFL、AutoAssign、YOLOF、YOLOv3、Sparse R-CNN、Deformable-DETR、CornerNet、SSD512。

特殊目录（均在 dior 下）：

- `baseline_resnet_mini`：R18 + FPN128 小模型基线（ATSS/PAA），100ep。
- `baseline_regnet_compose`：PAA + RegNet3.2GF 的损失组合消融（CIoU/DIoU/QFL/贝叶斯头）。**依赖的自定义类未实现，不可运行**。
- `baseline_self_statistics`：自研 SS/SSHead 检测器（PAA 改 MaxIoU 分配）。**类未实现，不可运行**。

### paper1_ARSD（Distill_Atss）

- 结构：teacher（ATSS+R101，NWPU 用 R50）内嵌于 `tea_model` 字段，加载 `tea_pretrained` 后冻结；student 为 R18 + f0.5/f0.25（另有 bk 变体）。
- 蒸馏损失：FPN 特征 MSE×目标 mask（×0.01）、cls 蒸馏（×0.01）、reg 蒸馏 NIoU（×0.1，默认关闭）。
- 面积加权通过 head 类型切换（`ATSSHeadnew` vs `ATSSHead`），不在配置参数中。
- `other/` 消融矩阵：
  - `dota_model_area_no_gn`：去 GN 消融（head norm_cfg=None）。
  - `dota_new_model`：无 area 加权的新 schedule 基线。
  - `dota_old`：旧 schedule（24ep / lr0.001 / bs4），含 101→50 蒸馏。
  - `nwpu_new_model`：非蒸馏单模型训练。
  - `test_model`：蒸馏配置的测试副本。

### paper2_S3MKM（Dis_ATSS / Distill_PAA + ATSSHead2）

- 机制：师生 cls/reg 损失比较生成 mask 选样本（多知识项：cate/geo/sup_fea/cls/reg）+ 通道适配 conv + 延迟蒸馏（`if_delay=True`，`ss_epoch=12`）。
- 配置只含 `tea_model` / `tea_pretrained`，**蒸馏损失开关与权重全部硬编码在 `mmdet/models/detectors/dis_atss.py`**。
- 实验矩阵不完整：多数 backbone×数据集格子只有学生基线，没有蒸馏配置。

### paper4_DIL（DIL_ATSS + ATSSHeadDIL）

- 机制：IIL 实例损失比较 + 三种注意力（FE/CB/LC），当前代码仅启用 CB 加权 cls 损失，FE/LC 被注释。
- `student2=True`（nwpu 配置）：加载未训练的第二学生互学，无教师 checkpoint。
- dota 仅 2 个配置，无师生基线，实验未完成。

## 配置 type 与代码对照

| 配置 type                                | 实现文件                                                           |
| ---------------------------------------- | ------------------------------------------------------------------ |
| `Distill_Atss` / `Distill_RetinaNet` | `mmdet/models/detectors/distill_atss.py` / `distill_retina.py` |
| `ATSSHeadnew`                          | `mmdet/models/dense_heads/atss_head_new.py`                      |
| `Dis_ATSS` / `Distill_PAA`           | `mmdet/models/detectors/dis_atss.py` / `dis_paa.py`            |
| `ATSSHead2`                            | `mmdet/models/dense_heads/atss_head2.py`                         |
| `DIL_ATSS`                             | `mmdet/models/detectors/DIL_atss.py`                             |
| `ATSSHeadDIL`                          | `mmdet/models/dense_heads/atss_head_DIL.py`                      |
| `ResNet_half` 系列                     | `mmdet/models/backbones/resnet_half.py` / `resnet_quarter.py`  |

## 使用方法

训练（单卡，指定 GPU）：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py a_config/paper1_ARSD/dota/distill/distill_atss_101_18_f0.5_dota_area.py
```

测试 + 评测（COCO 格式数据集）：

```bash
python tools/test.py <config.py> <checkpoint.pth> --eval bbox
```

DOTA 大图评测需走裁剪→测试→拼接→devkit 评测四步流程，见根目录 `说明dota.txt`。
参数量/FLOPs 分析：

```bash
python tools/get_flops.py <config.py>
```

运行前提：

1. 先 `pip install -e .` 生成 `mmdet/version.py`（环境为 Python 3.6 + mmcv-full 1.0 系 + PyTorch 1.5/1.6 老栈）。
2. 修改配置中所有 `data_root`、`pretrained`、`tea_pretrained` 为本机实际路径（原值均为 Linux 服务器路径）。
3. 蒸馏实验需先训好教师并指定其 checkpoint。

## 已知问题

分为已修复与待解决两部分（核查 2026-11-18，同日多次复查更新）。

### 已修复 / 部分修复（复查确认）

- ✅ **已修复**：`dis_atss.py` 蒸馏开关 `if_cat/if_geo/if_fea/if_cls/if_reg` 已全部改为 True（原全 False 导致 Dis_ATSS 退化为普通 ATSS 的问题已解决）。注意：开关仍为代码内硬编码，无法从配置切换。
- ✅ **已修复**：`DIL_atss.py:168` 教师损失已改为 `loss_tea = cls_loss_tea + reg_loss_tea`（原漏加 cls 项）。
- ✅ **已修复**：`paper2_S3MKM/dota/T_S_D_Res/D_paa_101_18_f0.5_dota.py` 外层 type 已改为 `Distill_PAA`（tea_model 内的 `type='PAA'` 为教师模型，属正确写法）。
- ✅ **已修复**：3 个含空格文件名已重命名（`cascade_rcnn_50_dior.py`、`sparse_rcnn50_dior.py`、`sparse_rcnn_r50_nwpu.py`）；2 个拼写错误已更正（`autossign_r50_8x2_nwpu.py` → `autoassign_r50_8x2_nwpu.py`、`defor_detr.py` → `deformable_detr.py`）。
- ✅ **已修复**：`retinanet_ghm_r50_fpn_1x_dota.py` 已重命名为 `retinanet_ghm_r101_fpn_1x_dota.py`，文件名与实际 depth=101 一致。
- ✅ **已修复**：`paper1_ARSD/dota/student/atss18_dota_area.py` 的 `_base_` 已补上 `.py` 后缀，基座文件存在，可正常加载。
- ✅ **已修复**：`2dsn.py` 缺失问题全部解决。基座文件已补充到 8 个目录（原仅 `dior/baseline_resnet_mini` 一处），paper2/paper4 及 paper1 dior 的 9 处 `./2dsn.py` 引用可正常解析；其余 31 个配置的 `_base_` 已改指向同目录既有的 `2_dataset_schedule_runtime.py`（dior/baseline_resnet 9、dior/baseline_regnet 9、baseline_regnet_compose 4、dota/baseline_regnet 2、paper1 dota/student 2、student_on_gn 2，及 3 个多变体目录的 distill 配置各 1，均选定 2_ 版本）。全量核查（237 个配置）：语法错误 0、`_base_` 目标缺失 0。

### 待解决问题

- **手动修订**：`distill_atss.py:39` `imitation_loss_weight` 被硬编码 0.01 覆盖，配置值无效。
- **手动修订**：`dis_atss.py:101-111` 已定义 dior/voc/nwpu/dota 四个数据集图片数，但 `self.images` 仍固定取 `num_images_dota`（15749），代码 todo 注明"根据实验自动选择"尚未实现，DIOR/NWPU 上延迟蒸馏轮数计算错误。
- **加载即失败**：未注册类 `SS`、`SSHead`、`SS_BAYHead`、`PAAQFLHead`、`CIoULoss`、`DIoULoss`（self_statistics / regnet_compose 目录，代码未实现）。
- **路径失效**：204 个配置含 `/home/airstudio/...`、`/workspace/...` 服务器路径（pretrained / tea_pretrained / data_root），本机不存在；部分路径含空格及 `pretained` 拼写错误。
- **卫生问题**：`paper1_ARSD/other/.../student_on_gn` 目录内 head 实际 norm_cfg=None，名实相反。
- **卫生问题**：个别 DOTA 基线训练 Resize 800×800 而测试 1024×1024（如 `dota/baseline_resnet/autoassign_r50_dota.py`）。
- **卫生问题**：paper4 dior 各 backbone lr 不一致（Res 0.01 / Shu 0.001）；paper2 dota Res 的 1dsn=30ep 与 2dsn=24ep 不一致。
