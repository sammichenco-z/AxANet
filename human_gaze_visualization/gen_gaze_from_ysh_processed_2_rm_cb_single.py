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
parser.add_argument("--root_time", type=str, default="0")
parser.add_argument("--split", type=str, default="new")
parser.add_argument("--gaussian", type=int, default=30)
parser.add_argument("--choose_driver", type=int, default=0)
parser.add_argument("--data_path", type=str, default="all_new_old_driver_v3")
args = parser.parse_args()


root_time = args.root_time
split = args.split

if args.choose_driver != 0:
    with open("driver_choose_list.json", "r") as f:
        driver_choose_list = json.load(f)

if True:
    dir_name = f"{args.data_path}/{split}"



    filelist = os.listdir(dir_name)
    filelist.sort()


    for begin_rate, end_rate in [(0.1,0.3), (0.3,0.5), (0.5,0.7), (0.7,0.9), (0.1,0.9)]:
    # for begin_rate, end_rate in [(0.1,0.9)]:

        average_result = {}

        for gaze_file in filelist:
            gaze_path = os.path.join(dir_name, gaze_file)

            if gaze_path.find("time") != -1:
                continue
            if gaze_path.find("fixation_only") != -1:
                prefex = "fixation_only"
                continue
            elif gaze_path.find("removefix") != -1:
                prefex = "removefix"
                continue
            elif gaze_path.find("marker") != -1:
                prefex = "raw"


            user_id = gaze_file.split("_")[1]
            
            # if user_id != "User65":
            #     continue
            
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

                    print(gaze_path)

                    if args.choose_driver != 0 and task is not None:
                        if gaze_file.split("_23")[0] not in driver_choose_list[split][task]:
                            continue

                    save_root_path = f"{root}/{split}/viz_gaze_{user_id}_{prefex}_{gaussianwh}_{gaussiansd}_{begin_rate}_{end_rate}/{task}"
                    count += 1

                    os.makedirs(save_root_path, exist_ok=True)
                    
                    img_save_path = os.path.join(save_root_path, os.path.basename(image_path))
                    gaze_point_number = len(gaze_point_list_per_img)

                    first_num = max(0, int(begin_rate*gaze_point_number))
                    last_num = min(gaze_point_number, int(end_rate*gaze_point_number))
                    gaze_point_list_per_img = gaze_point_list_per_img[first_num:last_num]

                    # print(len(gaze_point_list_per_img), gaze_point_number)

                    if task is None:
                        continue

                    if image_path+"**"+task not in average_result:
                        average_result[image_path+"**"+task] = gaze_point_list_per_img
                        average_result[image_path+"**"+task+"save"] = img_save_path
                    else:
                        average_result[image_path+"**"+task] = average_result[image_path+"**"+task] + gaze_point_list_per_img
                    

                    heatmap = draw_heatmap(gaze_point_list_per_img, (display_size, display_size), alpha=0.5, savefilename=img_save_path, imagefile=image_path, gaussianwh=gaussianwh, gaussiansd=gaussiansd)
                    # np.save(os.path.splitext(img_save_path)[0]+'.npy', heatmap)

                    heat_image = cv2.imread(img_save_path)
                    h, w = heat_image.shape[:2]
                    if not (str(response_x) == 'nan' or str(response_y) == 'nan'):
                        cv2.circle(heat_image, (int((response_x-1920)/1920*w), int(response_y/1080*h)), 10, (255, 255, 255), 10)
                        gt = get_gt(task, img_save_path)
                        if gt is not None:
                            cv2.rectangle(heat_image, (int(gt[0]*w), int(gt[1]*h)), (int(gt[2]*w), int(gt[3]*h)), (0, 255, 255), 4)

                    if not str(response) == 'nan':
                        cv2.putText(heat_image, response, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                        gt = get_gt(task, img_save_path)
                        if gt is not None:
                            cv2.putText(heat_image, str(gt), (100, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

                    cv2.imwrite(img_save_path.replace(".png", "response.png"), heat_image)



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
                # print(task)
        # if len(average_result) != 0:
        #     break

    # for image_path_task in average_result.keys():
    #     if image_path_task.endswith("save"):
    #         continue
    #     os.makedirs(os.path.dirname(average_result[image_path_task+"save"]), exist_ok=True)
    #     # img_save_path = os.path.join(save_root_path, os.path.basename(image_path_task))
    #     os.makedirs(os.path.join(root, f"{split}_mean"), exist_ok=True)
    #     heatmap = draw_heatmap(average_result[image_path_task], (display_size, display_size), alpha=0.5, savefilename=average_result[image_path_task+"save"].replace(split, f"{split}_mean"), imagefile=image_path_task.split("**")[0], gaussianwh=gaussianwh, gaussiansd=gaussiansd)
    
    