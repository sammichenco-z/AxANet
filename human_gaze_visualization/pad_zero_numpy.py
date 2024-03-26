import numpy as np
import os
from tqdm import tqdm



import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--root_time", type=str, default="0")
args = parser.parse_args()

root_time = args.root_time
root_dir = f"gaze_compare_{root_time}/"

filenames = os.listdir("human_machine_gaze_data/all_raw_image_v3_png")
filenames.sort()


for split in ['new_mean', 'old_mean']:
    split_dir = os.path.join(root_dir, split)
    for task in ['Anomaly', 'Hazard', 'Turn']:
        gaze_names = os.listdir(split_dir)
        for gaze_name in gaze_names:

            image_dir = os.path.join(split_dir, gaze_name, task)
            for filename in tqdm(filenames):
                npy_path = os.path.join(image_dir, filename)+".npy"
                if not os.path.exists(npy_path):
                    zero_pad = np.zeros((512, 512), dtype=np.float64)
                    np.save(npy_path, zero_pad)
                    print(npy_path)