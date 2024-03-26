import os
import pandas as pd
import pickle
import json
import numpy as np
from tqdm import tqdm

import torch
import torchvision


with open("gaze_visualize_utils/human_machine_gaze_data/v3_gt/drama_bbox_gt.json", "r") as f:
    human_machine_choose = json.load(f).keys()


def get_iou(pred, targ):
    pred = torch.tensor(pred)
    targ = torch.tensor(targ)
    return torchvision.ops.box_iou(pred.unsqueeze(0), targ.unsqueeze(0)).squeeze()



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




split = "test"
root_data_path = f"data/datasets/{split}"

to_eval_file = "drama_output_eds2/base_pretrained/checkpoint-1-92/pred.test.tsv"
# to_eval_file = "drama_output_eds2/base_pretrained/checkpoint-100-9200/pred.test.tsv"


# to_eval_file = "drama_output/transformer_baseline_pred_class_6layer_lr0002_detr_backbone_pretrained/checkpoint-1-92/pred.test.tsv"
# to_eval_file = "drama_output/transformer_baseline_pred_class_6layer_lr0002_detr_backbone_pretrained/checkpoint-117-10764/pred.test.tsv"

df = pd.read_csv(to_eval_file, sep='\t', header=None)
result_dict = {}
for index, row in df.iterrows():
    dict_list = json.loads(row[1])
    result_dict[row[0]] = dict_list



data_list = os.listdir(root_data_path)
data_list.sort()

acc = 0.
count = 0.

choose_acc = 0.
choose_iou = 0.
choose_count = 0.


for filename in tqdm(data_list):
    if not filename.endswith(".pkl"):
        continue

    json_data = result_dict[split+"_"+filename][0]
    pred_bbox = np.array(json.loads(json_data['box'])) # x1, y1, x2, y2

    test_data_path = os.path.join(root_data_path, filename)
    with open(test_data_path, "rb") as f:
        test_data = pickle.load(f)
        gt_bbox = np.array(test_data['bbox']) # x1,y1,w,h
        tmp_img_name = test_data['image_path'][-1].replace("/", "_").replace(".png", "_raw.png")
        gt_bbox = bbox_xywh2xyxy(gt_bbox)
    if (pred_bbox[0]+pred_bbox[2])/2 > gt_bbox[0] and (pred_bbox[0]+pred_bbox[2])/2 < gt_bbox[2] and \
       (pred_bbox[1]+pred_bbox[3])/2 > gt_bbox[1] and (pred_bbox[1]+pred_bbox[3])/2 < gt_bbox[3]:
       acc += 1
       if tmp_img_name in human_machine_choose:
           choose_acc += 1
           choose_iou += get_iou(pred_bbox, gt_bbox)
    #    if 
    count += 1
    if tmp_img_name in human_machine_choose:
       choose_count += 1
    print(f"tmp_acc: {acc/count}")

print(f"Acc: {acc/count}")
print(f"30Acc: {choose_acc/30}")
print(f"30IoU: {choose_iou/30}")
print(choose_count)
print(to_eval_file)
