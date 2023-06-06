#ARSD-S3MKMDIL
The official code for ARSD-S3MKM-DIL. We foucus more on the detection distillation in remote sensing  field.
We release the configs (a_configs) and some designed modules (mmdet)

## Install
You should install as the instruction of mmdetection official setting
https://github.com/open-mmlab/mmdetection

You also need to install below library (if you use the DOTA dataset)
https://github.com/CAPTAIN-WHU/DOTA_devkit

## Train 
### Teacher model
```bash
python3 tools/train.py a_config/ARSD/Tea_Stu/atss101_dior.py
```

### Student model
```bash
python3 tools/train.py a_config/ARSD/Tea_Stu/atss18_fpn0.5_dior.py
```

### Distillation model
```bash
python3 tools/train.py a_config/ARSD/dota/distill/distill_atss_101_18_f0.5_dior.py
```

## Test
```bash
python3 tools/test.py a_config/ARSD/dota/distill/distill_atss_101_18_f0.5_dior.py [model.pth] --eval bbox
```

## Paper
1 Adaptive Knowledge Distillation for Lightweight Remote Sensing Object Detectors Optimizing

Link : https://ieeexplore.ieee.org/document/9775159/

2 Statistical Sample Selection and Multivariate Knowledge Mining for Lightweight Detectors in Remote Sensing Imagery

Link : https://ieeexplore.ieee.org/document/9832637/

3 Dynamic Interactive Learning for Lightweight Detectors in Remote Sensing Imagery

Link : https://ieeexplore.ieee.org/document/9921272/

## Citation
If Our work is useful or relevant to your research, please kindly recognize our contributions by citing our paper:

```bibtex
@article{yang2022adaptive,
	title={Adaptive Knowledge Distillation for Lightweight Remote Sensing Object Detectors Optimizing},
	author={Yang, Yiran and Sun, Xian and Diao, Wenhui and Li, Hao and Wu, Youming and Li, Xinming and Fu, Kun},
	journal={IEEE Transactions on Geoscience and Remote Sensing},
	year={2022},
	volume={60},
	number={},
	pages={1-15},
	publisher={IEEE}
}

@article{yang2022statistical,
	title={Statistical Sample Selection and Multivariate Knowledge Mining for Lightweight Detectors in Remote Sensing Imagery},
	author={Yang, Yiran and Sun, Xian and Diao, Wenhui and Yin, Dongshuo and Yang, Zhujun and Li, Xinming},
	journal={IEEE Transactions on Geoscience and Remote Sensing},
	volume={60},
	pages={1--14},
	year={2022},
	publisher={IEEE}
}

@article{yang2022dynamic,
	title={Dynamic Interactive Learning for Lightweight Detectors in Remote Sensing Imagery},
	author={Yang, Yiran and Diao, Wenhui and Rong, Xuee and Li, Xinming and Sun, Xian},
	journal={IEEE Transactions on Geoscience and Remote Sensing},
	volume={60},
	pages={1--14},
	year={2022},
	publisher={IEEE}
}
```
