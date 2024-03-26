import os
import cv2
import numpy as np
from tqdm import tqdm


time = "8_7"
for choice in ['Anomaly', 'Turn', 'Hazard']:
    save_path = f"viz_gaze_{time}_compare/{choice}"
    os.makedirs(save_path, exist_ok=True)
    path1 = f"viz_gaze_{time}_raw_200_20/{choice}"
    path2 = f"viz_gaze_{time}_fixation_only_200_20/{choice}"
    path3 = f"viz_gaze_{time}_removefix_200_20/{choice}"

    names = os.listdir(path1)
    names.sort()
    for name in tqdm(names):
        if not name.endswith(".png"):
            continue
        file_path1 = os.path.join(path1, name)
        file_path2 = os.path.join(path2, name)
        file_path3 = os.path.join(path3, name)

        img1 = cv2.imread(file_path1)
        img2 = cv2.imread(file_path2)
        img3 = cv2.imread(file_path3)
        
        image = np.hstack((img1, img2, img3))

        cv2.imwrite(os.path.join(save_path, name), image)