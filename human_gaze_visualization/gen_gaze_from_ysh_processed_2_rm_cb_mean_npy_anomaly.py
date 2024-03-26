import pandas as pd
import cv2
import numpy as np
import os
from gazeheat import draw_heatmap
from tqdm import tqdm
import json


import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--root_time", type=str, default="1_15_for_anomaly_v2_t0_t1")
parser.add_argument("--split", type=str, default="new")
parser.add_argument("--gaussian", type=int, default=35)
parser.add_argument("--choose_driver", type=int, default=0)
parser.add_argument("--data_path", type=str, default="all_new_old_driver_other/for_anomaly_new/t0_t1")
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


    for begin_rate, end_rate in [(0.0,1.0)]:
    # for begin_rate, end_rate in [(0.1,0.9)]:

        average_result = {}

        for gaze_file in filelist:
            
            # if gaze_file.find("User96")==-1:
            #     continue

            gaze_path = os.path.join(dir_name, gaze_file)

            if gaze_path.find("time") != -1:
                continue

            user_id = gaze_file.split("_")[1]
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
            material_record = {}
            for idx, material in tqdm(enumerate(materials)):
                material_record[material] = 1
                if found_image and material != current_material:
                    #if change image, save heatmap
                    found_image = False

                    if gaze_path.find("fixation_only") != -1:
                        prefex = "fixation_only"
                    elif gaze_path.find("removefix") != -1:
                        prefex = "removefix"
                        break
                    elif gaze_path.find("marker") != -1:
                        prefex = "raw"
                        break
                    else:
                        break

                    # print(gaze_path)

                    # if args.choose_driver != 0 and task is not None:
                    #     if gaze_file.split("_23")[0] not in driver_choose_list[split][task]:
                    #         continue


                    save_root_path = f"{root}/{split}/viz_gaze_{prefex}_{gaussianwh}_{gaussiansd}_{begin_rate}_{end_rate}/{task}"
                    count += 1

                    os.makedirs(save_root_path, exist_ok=True)
                    
                    image_path = image_path.replace("training", "train")
                    seq_id = image_path.split("seq")[-1].split("/")[0]
                    train_test_split = image_path.split("v1_downsample_1000_")[-1].split("/")[0]
                    if train_test_split == "train":
                        train_test_split = "trainset"
                    if train_test_split == "test":
                        train_test_split = "testset"
                    img_save_path = os.path.join(save_root_path, train_test_split, "seq"+seq_id, os.path.basename(image_path))
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


                if not isinstance(material, str):
                    continue
                elif not found_image:
                    found_image = True
                    current_material = material
                    gaze_point_list_per_img = []

                    # response_x = data['x'][idx]
                    # response_y = data['y'][idx]
                    # response = data['respond'][idx]
                    # if (str(response_x) == 'nan' or str(response_y) == 'nan') and str(response) == 'nan' :
                    #     found_image = False

                    # NewMaterial\drama_random pic_test30\itan_clip_62_000474_frame_000474_raw.png
                    image_path = os.path.join("human_machine_gaze_data/all_raw_image_v4_anomaly", material.replace('\\', '/').replace("aoi_all_", "v1_downsample_1000_"))
                    image_path = os.path.splitext(image_path)[0] + '.jpeg'
                    image_path = image_path.replace("itan_", "titan_")
                    # image = cv2.imread(image_path)

                if data['Gaze Point X[px]'][idx] != -1 and data['Gaze Point Y[px]'][idx] != -1:
                    gaze_point_list_per_img.append((int(data['Gaze Point X[px]'][idx]/1920*display_size), int(data['Gaze Point Y[px]'][idx]/1080*display_size), 1))
                task = data['task'][idx]
            # if len(average_result) != 0:
            #     break
            print(len(material_record))
            # break

        # x = 1

        for image_path_task in tqdm(average_result.keys()):
            if image_path_task.endswith("save"):
                continue
            os.makedirs(os.path.dirname(average_result[image_path_task+"save"].replace(split, f"{split}_mean")), exist_ok=True)
            
            # img_save_path = os.path.join(save_root_path, os.path.basename(image_path_task))
            os.makedirs(os.path.join(root, f"{split}_mean"), exist_ok=True)
            heatmap = draw_heatmap(average_result[image_path_task], (display_size, display_size), alpha=0.5, savefilename=average_result[image_path_task+"save"].replace(split, f"{split}_mean"), imagefile=image_path_task.split("**")[0], gaussianwh=gaussianwh, gaussiansd=gaussiansd)
            np.save(average_result[image_path_task+"save"].replace(split, f"{split}_mean")+'.npy', heatmap)
            # print(average_result[image_path_task+"save"].replace(split, f"{split}_mean")+'.npy')
        