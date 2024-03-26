import pandas as pd
import cv2
import numpy as np
import os
from gazeheat import draw_heatmap
from tqdm import tqdm

split = "new"

dir_name = f"all_new_old_driver/{split}"



filelist = os.listdir(dir_name)
filelist.sort()


for begin_rate, end_rate in [(0.1,0.3), (0.3,0.5), (0.5,0.7), (0.7,0.9), (0.1,0.9)]:
# for begin_rate, end_rate in [(0.1,0.9)]:

    average_result = {}

    for gaze_file in filelist:
        gaze_path = os.path.join(dir_name, gaze_file)

        if gaze_path.find("time") != -1:
            continue

        time = gaze_file.split("_")[1]
        data = pd.read_csv(gaze_path, sep=',', header='infer')
        display_size = 512
        materials = data['name']

        found_image = False
        current_material = None
        gaze_point_list_per_img = []
        count = 0

        gaussianwh = 200
        gaussiansd = 30


        task = None

        for idx, material in tqdm(enumerate(materials)):
            if found_image and material != current_material:
                #if change image, save heatmap
                found_image = False

                if gaze_path.find("fixation_only") != -1:
                    prefex = "fixation_only"
                    break
                elif gaze_path.find("removefix") != -1:
                    prefex = "removefix"
                    break
                elif gaze_path.find("marker") != -1:
                    prefex = "raw"



                save_root_path = f"/data/jinbu/gaze_compare_8_25/{split}/viz_gaze_{gaussianwh}_{gaussiansd}_{begin_rate}_{end_rate}/{task}"
                count += 1

                # os.makedirs(save_root_path, exist_ok=True)
                
                img_save_path = os.path.join(save_root_path, os.path.basename(image_path))
                gaze_point_number = len(gaze_point_list_per_img)

                first_num = min(gaze_point_number-1, int(begin_rate*gaze_point_number))

                rm_rate = 1-end_rate
                last_num = max(-int(rm_rate*gaze_point_number), 1-gaze_point_number)
                gaze_point_list_per_img = gaze_point_list_per_img[first_num:last_num]

                # print(len(gaze_point_list_per_img), gaze_point_number)

                if image_path+"**"+task not in average_result:
                    average_result[image_path+"**"+task] = gaze_point_list_per_img
                    average_result[image_path+"**"+task+"save"] = img_save_path
                else:
                    average_result[image_path+"**"+task] = average_result[image_path+"**"+task] + gaze_point_list_per_img
                

                heatmap = draw_heatmap(gaze_point_list_per_img, (display_size, display_size), alpha=0.5, savefilename=img_save_path, imagefile=image_path, gaussianwh=gaussianwh, gaussiansd=gaussiansd)
                # np.save(os.path.splitext(img_save_path)[0]+'.npy', heatmap)

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

    for image_path_task in average_result.keys():
        if image_path_task.endswith("save"):
            continue
        os.makedirs(os.path.dirname(average_result[image_path_task+"save"]), exist_ok=True)
        # img_save_path = os.path.join(save_root_path, os.path.basename(image_path_task))
        heatmap = draw_heatmap(average_result[image_path_task], (display_size, display_size), alpha=0.5, savefilename=average_result[image_path_task+"save"], imagefile=image_path_task.split("**")[0], gaussianwh=gaussianwh, gaussiansd=gaussiansd)
    
    