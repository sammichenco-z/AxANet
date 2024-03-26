import os
import pandas as pd
import pickle
import json
import numpy as np
from tqdm import tqdm


human_choose = os.listdir("data/human_machine_data")


def bbox_xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] where xy=top-left
    # to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    if len(x.shape) == 2:
        y = np.copy(x)
        y[:, 0] = x[:, 0]  # top left x
        y[:, 1] = x[:, 1]  # top left y
        y[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
    else:
        y = np.copy(x)
        y[0] = x[0]  # top left x
        y[1] = x[1]  # top left y
        y[2] = x[0] + x[2]  # bottom right x
        y[3] = x[1] + x[3]  # bottom right y
    return y



with open("anomaly_dataset_gt/anomaly_dataset_gt_test.json") as f:
    gt = json.load(f)


# to_eval_file = "anomaly_output/base_pretrained/checkpoint-1-126/pred.test.tsv"
to_eval_file = "anomaly_output/base_pretrained/checkpoint-100-12600/pred.test.tsv"

df = pd.read_csv(to_eval_file, sep='\t', header=None)
result_dict = {}
for index, row in df.iterrows():
    dict_list = json.loads(row[1])
    result_dict[row[0]] = dict_list




acc = 0.
count = 0.

choose_acc = 0.
choose_iou = 0.
choose_count = 0.

for filename in gt.keys():

    
    pred_bbox = np.array(json.loads(result_dict["test_"+filename][0]['box']))
    box_acc = np.array(json.loads(result_dict["test_"+filename][0]['box_acc'])) # x1, y1, x2, y2
    box_iou = np.array(json.loads(result_dict["test_"+filename][0]['box_iou']))


    gt_bbox = gt[filename]['x1y1x2y2_bbox']
    if (pred_bbox[0]+pred_bbox[2])/2 > gt_bbox[0] and (pred_bbox[0]+pred_bbox[2])/2 < gt_bbox[2] and \
       (pred_bbox[1]+pred_bbox[3])/2 > gt_bbox[1] and (pred_bbox[1]+pred_bbox[3])/2 < gt_bbox[3]:
        assert box_acc == 1
    acc += box_acc

    if filename.split("seq")[1].split("/")[0]+"_"+filename.split("testset/")[1].split("/")[0]+"_"+filename.split("/")[-1] in human_choose:
        choose_acc += box_acc
        choose_iou += box_iou
        choose_count += 1


    count += 1
    print(f"tmp_acc: {acc/count}")

print(f"Acc: {acc/count}")
print(f"30Acc: {choose_acc/30}")
print(f"30IoU: {choose_iou/30}")
print(choose_count)