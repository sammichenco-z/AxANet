import pandas as pd
import cv2
import numpy as np
import os
from gazeheat import draw_heatmap
from tqdm import tqdm
import json

split = "new"

dir_name = f"all_new_old_driver/{split}"

rm_rate = 0.3
rm_b_rate = 0.1

filelist = os.listdir(dir_name)
filelist.sort()

acc_all = []

for gaze_file in filelist:
    
    gaze_path = os.path.join(dir_name, gaze_file)

    if gaze_path.find("marker") == -1:
        continue


    data = pd.read_csv(gaze_path, sep=',', header='infer')
    display_size = 512
    materials = data['name']

    found_image = False
    current_material = None
    gaze_point_list_per_img = []
    count = 0
    gaze_acc_dict = {}
    for idx, material in tqdm(enumerate(materials)):
        if found_image and material != current_material:
            #if change image, save heatmap
            found_image = False

            gaze_acc_dict[os.path.basename(image_path)] = response

        if not isinstance(material, str):
            continue
        elif not found_image:
            found_image = True
            current_material = material
            gaze_point_list_per_img = []

            response = data['respond'][idx]
            if str(response) == 'nan':
                found_image = False
            # NewMaterial\drama_random pic_test30\itan_clip_62_000474_frame_000474_raw.png
            image_path = os.path.join("human_machine_gaze_data/all_raw_image_v2_png", material.split('\\')[-1])
            image_path = os.path.splitext(image_path)[0] + '.png'
            image_path = image_path.replace("itan_", "titan_")
            # image = cv2.imread(image_path)

        # if data['Gaze Point X[px]'][idx] != -1 and data['Gaze Point Y[px]'][idx] != -1 and not np.isnan(data['Gaze Point X[px]'][idx]) and not np.isnan(data['Gaze Point Y[px]'][idx]):
        #     gaze_point_list_per_img.append((int(data['Fixation Point X[px]'][idx]/1920*display_size), int(data['Fixation Point Y[px]'][idx]/1080*display_size), 1))

    # with open("turn_acc.json", "w") as f:
    #     json.dump(gaze_acc_dict, f)

    gt_data = pd.read_csv('selected night.csv', sep=',', header='infer', usecols=[0,1])
    gt_acc_dict = {}
    for idx, data in enumerate(gt_data['name']):
        gt_acc_dict[os.path.splitext(data)[0]+'.png'] = str(gt_data['Ground truth, Left = 1'][idx])

    # with open("gt_acc.json", "w") as f:
    #     json.dump(gt_acc_dict, f)

    MAP = {
        '{SPACE}': '0',
        '{ENTER}': '1'

    }

    acc = []
    for raw_key in gt_acc_dict.keys():
        key = raw_key.replace("_viz.png", "_raw.png")
        if MAP[gaze_acc_dict[key]] == gt_acc_dict[raw_key]:
            acc.append(1)
        else:
            acc.append(0)
    print(gaze_path)
    print(len(acc))
    print(len(gaze_acc_dict))
    print(np.mean(acc))
    acc_all.append(np.mean(acc))

print(np.mean(acc_all))