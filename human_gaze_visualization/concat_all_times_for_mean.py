import os
import cv2
import numpy as np
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--root_time", type=str, default="0")
parser.add_argument("--split", type=str, default="new")
parser.add_argument("--gaussian", type=int, default=30)
args = parser.parse_args()

split = args.split
root_time = args.root_time
gaussian = args.gaussian

prefexes = [f"{root_time}/{split}_mean/viz_gaze_raw_200_{gaussian}"]

for prefex in prefexes:

    for choice in ['Anomaly', 'Turn', 'Hazard']:
        save_path = f"gaze_compare_{prefex}_compare/{choice}"
        os.makedirs(save_path, exist_ok=True)
        path1 = f"gaze_compare_{prefex}_0.1_0.3/{choice}"
        path2 = f"gaze_compare_{prefex}_0.3_0.5/{choice}"
        path3 = f"gaze_compare_{prefex}_0.5_0.7/{choice}"
        path4 = f"gaze_compare_{prefex}_0.7_0.9/{choice}"
        path5 = f"gaze_compare_{prefex}_0.1_0.9/{choice}"

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