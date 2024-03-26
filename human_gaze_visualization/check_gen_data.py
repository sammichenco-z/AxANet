import os
import sys

from IPython import embed
import cv2
import random
import numpy as np
import time
from tqdm.autonotebook import tqdm

import pandas as pd
from openpyxl import load_workbook



output_path = '/DATA/DISCOVER/lpf/gaze/dataset/drama_gaze_collect/gaze/images'
drama_path = '/DATA/aidrive/zyp/jinbu/combined'

video_count = 0

dir_1 = sorted([f for f in os.listdir(output_path)])
for dir_1_item in dir_1:
    dir_1_item_path = os.path.join(output_path, dir_1_item)

    dir_2 = sorted([f for f in os.listdir(dir_1_item_path)])

    for dir_2_item in dir_2:
        dir_2_item_path = os.path.join(dir_1_item_path, dir_2_item, 'fixation')

        files = sorted([f for f in os.listdir(dir_2_item_path)])
        
        ori_data_path = os.path.join(drama_path, dir_1_item, dir_2_item)
        files_ori = sorted([f for f in os.listdir(ori_data_path) if f.startswith('frame')])

        assert files == files_ori

        print(dir_2_item_path)


        video_count += 1

print(video_count)



