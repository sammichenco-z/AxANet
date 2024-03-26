import os
import cv2
import numpy as np
from tqdm import tqdm

for choice in ['Anomaly', 'Turn', 'Hazard']:
    save_path = f"gaze_compare_8_25/new_old_comp/{choice}"
    os.makedirs(save_path, exist_ok=True)
    path1 = f"gaze_compare_8_25/new/viz_gaze_230805151723_raw_200_20_compare/{choice}"
    path2 = f"gaze_compare_8_25/old/viz_gaze_230806181942_raw_200_20_compare/{choice}"
    # path3 = f"viz_gaze_{time}_removefix_200_20/{choice}"

    names = os.listdir(path1)
    names.sort()
    for name in tqdm(names):
        if not name.endswith(".png"):
            continue
        file_path1 = os.path.join(path1, name)
        file_path2 = os.path.join(path2, name)
        # file_path3 = os.path.join(path3, name)

        img1 = cv2.imread(file_path1)
        img2 = cv2.imread(file_path2)
        # img3 = cv2.imread(file_path3)
        
        image = np.vstack((img1, img2))

        cv2.imwrite(os.path.join(save_path, name), image)