# Designed by Mao Wingkeung
import pandas as pd
import json
import matplotlib.pyplot as plt
# import tushare as tf

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (10.0, 10.0)

ann_json = 'DOTA_train1024.json'
with open(ann_json) as f:
    ann = json.load(f)

category_dic = dict([(i['id'], i['name']) for i in ann['categories']])
counts_label = dict([(i['name'], 0) for i in ann['categories']])
for i in ann['annotations']:
    counts_label[category_dic[i['category_id']]] += 1
num = 0
box_w = []
box_h = []
box_wh = []
categorys_wh = [[] for j in range(10)]
for a in ann['annotations']:
    if a['category_id'] != 0:
        box_w.append(round(a['bbox'][2], 2))
        box_h.append(round(a['bbox'][3], 2))
        if a['bbox'][3] == 0:
            wh = 1
            num = num + 1
        else:
            wh = round(a['bbox'][2] / a['bbox'][3], 0)
        # wh = round(a['bbox'][2]/a['bbox'][3], 0)

        if a['bbox'][2] == 0:
            wh = 1
            num = num + 1

        if wh < 1:
            wh = round(a['bbox'][3]/a['bbox'][2], 0)
        box_wh.append(wh)
        # print(a, box_wh)
        # categorys_wh[a['category_id']-1].append(wh)
        # print(categorys_wh)

box_wh_unique = list(set(box_wh))
box_wh_unique = list(range(50))
# print(box_wh_unique)
box_wh_count = [box_wh.count(i) for i in box_wh_unique]

wh_df = pd.DataFrame(box_wh_count, index=box_wh_unique, columns=['宽高比数量'])
wh_df.plot(kind='bar', color="#55aacc")
plt.show()
print(num)