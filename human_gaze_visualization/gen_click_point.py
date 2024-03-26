import pandas as pd
import cv2
import numpy as np
import os
from gazeheat import draw_heatmap
from tqdm import tqdm
import json

def get_gt(task, img_path):
    if task == "Anomaly":
        gt_json_path = "human_machine_gaze_data/v3_gt/anomaly_bbox_gt.json"
        data = json.load(open(gt_json_path, "r"))
        key = os.path.basename(img_path).split(".")[0]
        if key in data:
            gt = data[key]['bbox']
        else:
            gt = None
    if task == "Hazard":
        gt_json_path = "human_machine_gaze_data/v3_gt/drama_bbox_gt.json"
        data = json.load(open(gt_json_path, "r"))
        key = os.path.basename(img_path)
        if key in data:
            gt = data[key]

            def bbox_xywh2xyxy(x):
                y = x.copy()
                y[0] = x[0]  # top left x
                y[1] = x[1]  # top left y
                y[2] = x[0] + x[2]  # bottom right x
                y[3] = x[1] + x[3]  # bottom right y
                return y

            gt = bbox_xywh2xyxy(gt)
        else:
            gt = None

    if task == "Turn":
        gt_json_path = "human_machine_gaze_data/v3_gt/TurnGroundTruth.xlsx"
        def xlsx_to_dict(file_path):
            df = pd.read_excel(file_path)
            data_dict = df.set_index(df.columns[0]).to_dict(orient='index')
            return data_dict

        data = xlsx_to_dict(gt_json_path)
        key = 'NewMaterial\\selected items_nighttime\\'+os.path.basename(img_path).replace(".png", ".jpg")
        if key in data:
            gt = data[key]
        else:
            gt = None


    return gt

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--root_time", type=str, default="10_13")
parser.add_argument("--split", type=str, default="old")
parser.add_argument("--gaussian", type=int, default=35)
parser.add_argument("--choose_driver", type=int, default=0)
parser.add_argument("--data_path", type=str, default="all_new_old_driver_v3")
args = parser.parse_args()


root_time = args.root_time
split = args.split

if True:
    dir_name = f"{args.data_path}/{split}"



    filelist = os.listdir(dir_name)
    filelist.sort()


    # for begin_rate, end_rate in [(0.1,0.3), (0.3,0.5), (0.5,0.7), (0.7,0.9), (0.1,0.9)]:
    for begin_rate, end_rate in [(0.1,0.9)]:

        average_result = {}

        for gaze_file in filelist:
            gaze_path = os.path.join(dir_name, gaze_file)

            if gaze_path.find("time") != -1:
                continue

            time = gaze_file.split("_")[1]
            data = pd.read_csv(gaze_path, sep=',', header='infer')
            display_size = 512
            materials = data['name']

            found_image = False
            current_material = None
            gaze_point_list_per_img = []
            count = 0

            gaussianwh = 200
            gaussiansd = args.gaussian


            task = None

            root = f"gaze_compare_{root_time}/"
            for idx, material in tqdm(enumerate(materials)):
                if found_image and material != current_material:
                    #if change image, save heatmap
                    found_image = False

                    if gaze_path.find("fixation_only") != -1:
                        prefex = "fixation_only"
                        break
                    elif gaze_path.find("removefix") != -1:
                        prefex = "removefix"
                        break
                    elif gaze_path.find("marker") != -1:
                        prefex = "raw"

                    print(gaze_path)

                    save_root_path = f"{root}/{split}/viz_gaze_{time}_{prefex}_{gaussianwh}_{gaussiansd}_{begin_rate}_{end_rate}/{task}"
                    count += 1

                    os.makedirs(save_root_path, exist_ok=True)
                    
                    img_save_path = os.path.join(save_root_path, os.path.basename(image_path))
                    
                    json_save_path = img_save_path.replace(".png", "response.json")
                    
                    if not (str(response_x) == 'nan' or str(response_y) == 'nan'):
                        click_point = ((response_x-1920)/1920, response_y/1080)

                    if not str(response) == 'nan':
                        click_point = response
                    
                    
                    to_save = {
                        "image_path": image_path,
                        "click_point": click_point,
                        "user_name": gaze_path,
                    }
                    with open(json_save_path, "w") as f:
                        json.dump(to_save, f)

                if not isinstance(material, str):
                    continue
                elif not found_image:
                    found_image = True
                    current_material = material
                    gaze_point_list_per_img = []

                    response_x = data['x'][idx]
                    response_y = data['y'][idx]
                    response = data['respond'][idx]
                    if (str(response_x) == 'nan' or str(response_y) == 'nan') and str(response) == 'nan' :
                        found_image = False

                    # NewMaterial\drama_random pic_test30\itan_clip_62_000474_frame_000474_raw.png
                    image_path = os.path.join("human_machine_gaze_data/all_raw_image_v3_png", material.split('\\')[-1])
                    image_path = os.path.splitext(image_path)[0] + '.png'
                    image_path = image_path.replace("itan_", "titan_")
                    # image = cv2.imread(image_path)

                if data['Gaze Point X[px]'][idx] != -1 and data['Gaze Point Y[px]'][idx] != -1:
                    gaze_point_list_per_img.append((int(data['Gaze Point X[px]'][idx]/1920*display_size), int(data['Gaze Point Y[px]'][idx]/1080*display_size), 1))
                    task = data['task'][idx]
            
