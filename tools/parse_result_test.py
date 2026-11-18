from __future__ import division

import argparse
from pycocotools.coco import COCO
import sys
from DOTA_devkit.ResultMerge_multi_process import *
import mmcv
# import pdb; pdb.set_trace()
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--json', default='/home?airstudio/dota_data/val1024/DOTA_val1024.json')#groundtruth
    parser.add_argument('--type', default=r'HBB',
                        help='parse type of detector')
    args = parser.parse_args()

    return args

def HBBDet2Comp4(filenames,classes,results):
    results_dict = {}
    for idx in range(len(filenames)):
        # print('idx: ', idx, 'total: ', len(dataset))
        filename = filenames[idx]
        result = results[idx]
        for label in range(len(result)):
            bboxes = result[label]
            # try:
            cls_name =classes[label]
            # except:
            #     import pdb
            #     pdb.set_trace()
            if cls_name not in results_dict:
                results_dict[cls_name] = []
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                score = float(bboxes[i][-1])
                outline = filename + ' ' + str(score) + ' ' + ' '.join(map(str, bbox))
                results_dict[cls_name].append(outline)
    return results_dict

def load_annotations( ann_file):
        coco = COCO(ann_file)
        cat_ids = coco.getCatIds()
        cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(cat_ids)
        }
        img_ids = coco.getImgIds()
        img_infos = []
        for i in img_ids:
            info = coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos


def parse_results(resultfile, dstpath, filenames,classes,type):
    outputs = mmcv.load(resultfile)
    hbb_results_dict = HBBDet2Comp4(filenames,classes,outputs)
    if 'hbb_results_dict' in vars():
        if not os.path.exists(os.path.join(dstpath, 'Task2_results')):
            os.makedirs(os.path.join(dstpath, 'Task2_results'))
        for cls in hbb_results_dict:
            with open(os.path.join(dstpath, 'Task2_results', 'Task2_'+cls + '.txt'), 'w') as f_out:
                for index, outline in enumerate(hbb_results_dict[cls]):
                    if index != (len(hbb_results_dict[cls]) - 1):
                        f_out.write(outline + '\n')
                    else:
                        f_out.write(outline)

        if not os.path.exists(os.path.join(dstpath, 'Task2_results_nms')):
            os.makedirs(os.path.join(dstpath, 'Task2_results_nms'))
        mergebyrec(os.path.join(dstpath, 'Task2_results'),os.path.join(dstpath, 'Task2_results_nms'))

if __name__ == '__main__':
    args = parse_args()
    pkl_file = './result.pkl'
    output_path = "./"
    classes = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
    img_infos=load_annotations('/home/airstudio/data/dota_data/test1024/DOTA_test1024.json')
    filenames=[p['file_name'][:-4] for p in img_infos]
    type = args.type
    parse_results(pkl_file, output_path, filenames,classes,type)
    print("test result is ok")