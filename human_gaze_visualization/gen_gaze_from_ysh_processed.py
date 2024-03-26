import pandas as pd
import cv2
import numpy as np
import os
from gazeheat import draw_heatmap
from tqdm import tqdm

data = pd.read_csv('collect_data_2023_7_26/User1_230726195723_marker.csv', sep=',', header='infer', usecols=[0,1,2,4])
display_size = 512
materials = data['name']

found_image = False
current_material = None
gaze_point_list_per_img = []
count = 0

gaussianwh = 200
gaussiansd = 20

for idx, material in tqdm(enumerate(materials)):
    if found_image and material != current_material:
        #if change image, save heatmap
        found_image = False
        
        save_root_path = f"viz_gaze_7_26_{gaussianwh}_{gaussiansd}/{count//90}"
        count += 1

        os.makedirs(save_root_path, exist_ok=True)
        
        img_save_path = os.path.join(save_root_path, os.path.basename(image_path))
        heatmap = draw_heatmap(gaze_point_list_per_img, (display_size, display_size), alpha=0.5, savefilename=img_save_path, imagefile=image_path, gaussianwh=gaussianwh, gaussiansd=gaussiansd)
        np.save(os.path.splitext(img_save_path)[0]+'.npy', heatmap)

    if not isinstance(material, str):
        continue
    elif not found_image:
        found_image = True
        current_material = material
        gaze_point_list_per_img = []

        # NewMaterial\drama_random pic_test30\itan_clip_62_000474_frame_000474_raw.png
        image_path = os.path.join("human_machine_gaze_data/all_raw_image_v2_png", material.split('\\')[-1])
        image_path = os.path.splitext(image_path)[0] + '.png'
        image_path = image_path.replace("itan_", "titan_")
        # image = cv2.imread(image_path)

    gaze_point_list_per_img.append((int(data['gazePoint.point.x'][idx]*display_size), int(data['gazePoint.point.y'][idx]*display_size), 1))

with open(os.path.join(os.path.dirname(save_root_path), "order.txt"), "r") as f:
    f.writelines("25: Hazard, Turn, Anomaly \n26: Anomaly, Turn, Hazard")