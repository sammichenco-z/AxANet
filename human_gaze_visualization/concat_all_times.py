import os
import cv2
import numpy as np
from tqdm import tqdm


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

dir_name = f"{args.data_path}/{split}"

filelist = os.listdir(dir_name)
filelist.sort()

gaze_names = []
for gaze_file in filelist:
    gaze_path = os.path.join(dir_name, gaze_file)

    if gaze_path.find("time") != -1:
        continue
    
    gaze_names.append(gaze_file.split("_")[1])

for choice in ['Anomaly', 'Turn', 'Hazard']:
    all_in_one = []
    # all_in_one_save_path = f"gaze_compare_{root_time}/{split}"

    for gaze_name in gaze_names:
        save_path = f"gaze_compare_{root_time}/{split}/viz_gaze_{gaze_name}_raw_200_{gaussian}_compare/{choice}"
        os.makedirs(save_path, exist_ok=True)
        path1 = f"gaze_compare_{root_time}/{split}/viz_gaze_{gaze_name}_raw_200_{gaussian}_0.1_0.3/{choice}"
        path2 = f"gaze_compare_{root_time}/{split}/viz_gaze_{gaze_name}_raw_200_{gaussian}_0.3_0.5/{choice}"
        path3 = f"gaze_compare_{root_time}/{split}/viz_gaze_{gaze_name}_raw_200_{gaussian}_0.5_0.7/{choice}"
        path4 = f"gaze_compare_{root_time}/{split}/viz_gaze_{gaze_name}_raw_200_{gaussian}_0.7_0.9/{choice}"
        path5 = f"gaze_compare_{root_time}/{split}/viz_gaze_{gaze_name}_raw_200_{gaussian}_0.1_0.9/{choice}"

        names = os.listdir(path1)
        names.sort()
        for name in tqdm(names):
            if not name.endswith(".png"):
                continue
            file_path1 = os.path.join(path1, name)
            file_path2 = os.path.join(path2, name)
            file_path3 = os.path.join(path3, name)
            file_path4 = os.path.join(path4, name)
            file_path5 = os.path.join(path5, name)

            img1 = cv2.imread(file_path1)
            img2 = cv2.imread(file_path2)
            img3 = cv2.imread(file_path3)
            img4 = cv2.imread(file_path4)
            img5 = cv2.imread(file_path5)      

            image = np.vstack((img1, img2, img3, img4, img5))

            cv2.imwrite(os.path.join(save_path, name), image)
            all_in_one.append(image.copy())

        # break
    # all_in_one = np.vstack(all_in_one)
    # cv2.imwrite(os.path.join(all_in_one_save_path, "compare_all.jpg"), all_in_one)