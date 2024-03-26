import pandas as pd
import cv2
import numpy as np
import os
from gazeheat import draw_heatmap
from tqdm import tqdm
import json



with open("human_machine_gaze_data/v3_gt/drama_bbox_gt.json", "r") as f:
    gts = json.load(f)


with open("drama_new_outputs.json", "r") as f:
    preds = json.load(f)

counts = 0.
acc = 0.
for key in gts.keys():
    pred = preds[key]
    gt = gts[key]
    
    if (pred[0]+pred[2])/2 > gt[0] and (pred[0]+pred[2])/2 < gt[0]+gt[2] and (pred[1]+pred[3])/2 > gt[1] and (pred[1]+pred[3])/2 < gt[1] + gt[3]:
        acc += 1
    
    counts += 1

print(acc/counts)
print(counts)