import os
import cv2
import numpy as np
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--root_time", type=str, default="0")
# parser.add_argument("--split", type=str, default="new")
parser.add_argument("--gaussian", type=int, default=30)
args = parser.parse_args()

# split = args.split
root_time = args.root_time
gaussian = args.gaussian


if True:


    for choice in ['Anomaly', 'Turn', 'Hazard']:
        save_path = f"gaze_compare_{root_time}/new_old_mean/viz_gaze_raw_200_{gaussian}_compare/{choice}"
        os.makedirs(save_path, exist_ok=True)
        path1 = f"gaze_compare_{root_time}/new_mean/viz_gaze_raw_200_{gaussian}_compare/{choice}"
        path2 = f"gaze_compare_{root_time}/old_mean/viz_gaze_raw_200_{gaussian}_compare/{choice}"


        names = os.listdir(path1)
        names.sort()
        for name in tqdm(names):
            if not name.endswith(".png"):
                continue
            file_path1 = os.path.join(path1, name)
            file_path2 = os.path.join(path2, name)

            img1 = cv2.imread(file_path1)
            img2 = cv2.imread(file_path2)

            image = np.vstack((img1, img2))

            cv2.imwrite(os.path.join(save_path, name), image)