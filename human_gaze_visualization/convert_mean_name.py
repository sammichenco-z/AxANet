import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil


import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--root_time", type=str, default="0")
parser.add_argument("--split", type=str, default="new")
parser.add_argument("--gaussian", type=int, default=30)
args = parser.parse_args()

split = args.split
root_time = args.root_time
gaussian = args.gaussian

path = f"gaze_compare_{root_time}/{split}_mean/"


filenames = os.listdir(path)
filenames.sort()
for filename in filenames:
    to_remove = filename.split("_gaze_")[-1].split("_raw_")[0]+"_"
    new_filename = filename.replace(to_remove, "")

    old_path = os.path.join(path, filename)
    new_path = os.path.join(path, new_filename)
    
    shutil.copytree(old_path, new_path, dirs_exist_ok=True)
