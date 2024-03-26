import os
import cv2
import numpy as np
from tqdm import tqdm

save_path = "raw_attention_human_machine_v3_new_old_10_7"
os.makedirs(save_path, exist_ok=True)
path1 = "raw_attention_human_machine_v3_new_10_7"
path2 = "raw_attention_human_machine_v3_old_10_7"

names = os.listdir(path2)
names.sort()
for name in tqdm(names):
    file_path1 = os.path.join(path1, name)
    file_path2 = os.path.join(path2, name)

    img1 = cv2.imread(file_path1)
    img2 = cv2.imread(file_path2)

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

    result = np.vstack((img1, img2))

    cv2.imwrite(os.path.join(save_path, name), result)


