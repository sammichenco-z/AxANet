import pandas as pd
import cv2
import numpy as np
import os
from gazeheat import draw_heatmap
from tqdm import tqdm

data = pd.read_csv('collect_data_2023_8_7/raw_User30_230807182954_0808175603_marker.csv', sep=',', header='infer')
display_size = 512
materials = data['name']

found_image = False
current_material = None
gaze_point_list_per_img = []
count = 0

gaussianwh = 200
gaussiansd = 20

task = None

for idx, material in tqdm(enumerate(materials)):
    if found_image and material != current_material:
        #if change image, save heatmap
        found_image = False
        
        save_root_path = f"viz_gaze_8_7_response_{gaussianwh}_{gaussiansd}/{task}"
        count += 1

        os.makedirs(save_root_path, exist_ok=True)
        
        img_save_path = os.path.join(save_root_path, os.path.basename(image_path))
        heatmap = draw_heatmap(gaze_point_list_per_img, (display_size, display_size), alpha=0.5, savefilename=img_save_path, imagefile=image_path, gaussianwh=gaussianwh, gaussiansd=gaussiansd)
        np.save(os.path.splitext(img_save_path)[0]+'.npy', heatmap)

        heat_image = cv2.imread(img_save_path)
        h, w = heat_image.shape[:2]
        cv2.circle(heat_image, (int((response_x-1920)/1920*w), int(response_y/1080*h)), 10, (255, 255, 255), 10)
        cv2.imwrite(img_save_path.replace(".png", "response.png"), heat_image)

    if not isinstance(material, str):
        continue
    elif not found_image:
        found_image = True
        current_material = material
        gaze_point_list_per_img = []

        response_x = data['x'][idx]
        response_y = data['y'][idx]
        if str(response_x) == 'nan' or str(response_y) == 'nan':
            found_image = False


        # NewMaterial\drama_random pic_test30\itan_clip_62_000474_frame_000474_raw.png
        image_path = os.path.join("human_machine_gaze_data/all_raw_image_v2_png", material.split('\\')[-1])
        image_path = os.path.splitext(image_path)[0] + '.png'
        image_path = image_path.replace("itan_", "titan_")
        # image = cv2.imread(image_path)

    if data['Gaze Point X[px]'][idx] != -1 and data['Gaze Point Y[px]'][idx] != -1:
        gaze_point_list_per_img.append((int(data['Gaze Point X[px]'][idx]/1920*display_size), int(data['Gaze Point Y[px]'][idx]/1080*display_size), 1))
        task = data['task'][idx]
