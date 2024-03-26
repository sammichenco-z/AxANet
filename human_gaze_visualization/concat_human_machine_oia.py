import os
import cv2
import numpy as np
from tqdm import tqdm

save_path = "machine_output/oia_output_concat"
os.makedirs(save_path, exist_ok=True)
path1 = "machine_output/oia_output"
path2 = "gaze_compare_10_30/new_old_compare/new_old+mean/Turn"

names = os.listdir(path2)
names.sort()
for name in tqdm(names):
    file_path1 = os.path.join(path1, "head_cams_"+name.replace("compare_all_", "").replace("response", ""))
    file_path2 = os.path.join(path2, name)

    print(file_path1)
    print(file_path2)
    print()

    try:
        img1 = cv2.imread(file_path1)
        img2 = cv2.imread(file_path2)
    except:
        continue

    height1, width1, _ = img1.shape
    height2, width2, _ = img2.shape

    if height1 < height2:
        new_height = height2
        new_width = int(width1 * (height2 / height1))
        img1 = cv2.resize(img1, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    else:
        new_height = height1
        new_width = int(width2 * (height1 / height2))
        img2 = cv2.resize(img2, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # result[:height1, :width1] = img1
    # result[:height2, width1:] = img2

    result = np.hstack((img1, img2))

    cv2.imwrite(os.path.join(save_path, name), result)