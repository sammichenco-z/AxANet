import os
import cv2
import numpy as np
from tqdm import tqdm
import json

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--root_time", type=str, default="0")
parser.add_argument("--split", type=str, default="new")
parser.add_argument("--gaussian", type=int, default=30)
parser.add_argument("--choose_driver", type=int, default=0)
parser.add_argument("--data_path", type=str, default="all_new_old_driver_v3")
args = parser.parse_args()

split = args.split
root_time = args.root_time
gaussian = args.gaussian

if args.choose_driver != 0:
    with open("driver_choose_list.json", "r") as f:
        driver_choose_list = json.load(f)

if True:

    dir_name = f"{args.data_path}/{split}"

    filelist = os.listdir(dir_name)
    filelist.sort()

    gaze_names = {}
    for gaze_file in filelist:
        gaze_path = os.path.join(dir_name, gaze_file)

        if gaze_path.find("time") != -1:
            continue

        gaze_names[gaze_file.split("_")[1]] = ''

    for choice in ['Anomaly', 'Turn', 'Hazard']:
        tmp_game_name = gaze_file.split("_")[1]
        tmp_path = f"gaze_compare_{root_time}/{split}/viz_gaze_{tmp_game_name}_raw_200_{gaussian}_compare/{choice}"
        names = os.listdir(tmp_path)
        names.sort()
        for name in tqdm(names):
            if not name.endswith(".png"):
                continue
            all_in_one = []
            all_in_one_save_path = f"gaze_compare_{root_time}/new_old_compare/{split}/{choice}"
            os.makedirs(all_in_one_save_path, exist_ok=True)

            for gaze_name in gaze_names.keys():

                if args.choose_driver != 0 and choice is not None:
                    if "raw_"+gaze_name not in driver_choose_list[split][choice]:
                        continue

                path1 = f"gaze_compare_{root_time}/{split}/viz_gaze_{gaze_name}_raw_200_{gaussian}_compare/{choice}"


                file_path1 = os.path.join(path1, name)
                # file_path2 = os.path.join(path2, name)
                # file_path3 = os.path.join(path3, name)
                # file_path4 = os.path.join(path4, name)
                # file_path5 = os.path.join(path5, name)

                img1 = cv2.imread(file_path1)
                # img2 = cv2.imread(file_path2)
                # img3 = cv2.imread(file_path3)
                # img4 = cv2.imread(file_path4)
                # img5 = cv2.imread(file_path5)

                # image = np.hstack((img1, img2, img3, img4, img5))
                image = img1

                # cv2.imwrite(os.path.join(save_path, name), image)
                all_in_one.append(image.copy())
            all_in_one = np.hstack(all_in_one)
            cv2.imwrite(os.path.join(all_in_one_save_path, f"compare_all_{name}"), all_in_one)