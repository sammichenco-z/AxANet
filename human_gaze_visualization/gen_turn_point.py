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

def read_info_from_excecl(raw_data_excel_path):
    print('read excel...')
    df = pd.read_excel(raw_data_excel_path)
    print('read excel Done')

    
    raw_img_path = df['TurnTestImage']
    labels = df['TurnTestDisplay.RESP']

    CLASS_NAMES = {
        '{SPACE}': 0,
        '{ENTER}': 1,
    }

    raw_base_names = []
    new_labels = []
    for id_, i in enumerate(raw_img_path):
        if isinstance(i, str):
            raw_base_names.append(i.split("\\")[-1].replace("itan_", "titan_"))
            new_labels.append(CLASS_NAMES[labels[id_]])

    return raw_base_names, new_labels



# 组间取平均

def main():
    root_img_path = "human_machine_gaze_data/all_raw_image_v3"
    raw_data_excels = ['collect_data_2023_7_25/072501_2_Turn.xlsx']
    save_root_path = "viz_click_info"
    

    for raw_data_excel in raw_data_excels:
        time1 = time.time()
        print(raw_data_excel, 'begin........................')

        #### get keys
        raw_base_names, new_labels = \
            read_info_from_excecl(raw_data_excel)

        for base_name, new_label in tqdm(zip(raw_base_names, new_labels)):
            raw_img = cv2.imread(os.path.join(root_img_path, base_name))
            
            if raw_img is None:
                continue

            raw_img = cv2.resize(raw_img, (1920, 1080))

            cv2.putText(raw_img, str(new_label), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            save_path = os.path.join(save_root_path, os.path.splitext(os.path.basename(raw_data_excel))[0])
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, os.path.splitext(base_name)[0]+".jpg"), raw_img)

        # break
        time2 = time.time()
        print('time: ', time2-time1)

        # break


if __name__ == '__main__':
    main()
