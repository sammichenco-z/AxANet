import json
import os
import numpy as np
import cv2


def bbox_xywh2cxcywh(x):
    # Convert nx4 boxes from [cx, cy, w, h] where cxcy=middle
    # to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    if len(x.shape) == 2:
        y = np.copy(x)
        y[:, 0] = x[:, 0] + x[:, 2]/2  # top left x
        y[:, 1] = x[:, 1] + x[:, 3]/2  # top left y
        y[:, 2] = x[:, 2]   # bottom right x
        y[:, 3] = x[:, 3]   # bottom right y
    else:
        y = np.copy(x)
        y[0] = x[0] + x[2]/2  # top left x
        y[1] = x[1] + x[3]/2  # top left y
        y[2] = x[2]  # bottom right x
        y[3] = x[3]  # bottom right y
    return y


human_path = "gaze_compare_10_11_35_hazard_anomaly/old_mean/viz_gaze_230825115919_raw_200_35_0.1_0.9/Hazard"
# human_path = "gaze_compare_10_11_35_hazard_anomaly/new_mean/viz_gaze_230827103946_raw_200_35_0.1_0.9/Hazard"

fake_gaze_path = "ada_dada2000_gaze"

gaze_paths = os.listdir(human_path)
gaze_paths.sort()

gt_path = "all_gts/drama_bbox_gt.json"
with open(gt_path, "r") as f:
    gts = json.load(f)

all_array = []
all_array_std = []
for filename in gaze_paths:
    if not filename.endswith(".png.npy"):
        continue

    file_key = filename.replace(".png.npy", "")
    if file_key not in gts:
        continue

    gaze_path = os.path.join("ada_dada2000_gaze", file_key)
    print("gaze_path:", gaze_path)
    gaze_data = cv2.resize(cv2.imread(gaze_path), (512, 512))[:, :, 0]
    
    # gaze_path = file_key.split("_")[1]

    # gaze_path = os.path.join(human_path, filename)
    # gaze_data = np.load(gaze_path)
    gaze_data = gaze_data/gaze_data.max()
    gaze_mean = gaze_data.mean()
    h, w = gaze_data.shape
    

    
    gt = gts[file_key]
    bbox_xywh = np.array([gt[0]*w, gt[1]*h, gt[2]*w, gt[3]*h])
    bbox_cxcywh = bbox_xywh2cxcywh(bbox_xywh)
    
    cx = bbox_cxcywh[0]
    cy = bbox_cxcywh[1]
    radius = np.sqrt(bbox_cxcywh[-2] ** 2 + bbox_cxcywh[-1] ** 2)/2
    
    new_array = np.zeros_like(gaze_data)
    std_array = np.zeros_like(gaze_data)
    for j in range(h):
        for i in range(w):
            distance = np.sqrt((j-cy) ** 2 + (i-cx) ** 2)/2

            # new_array[j][i] = gaze_data[j][i]
            # std_array[j][i] = (gaze_data[j][i]-gaze_mean)**2

            if distance < radius:
                new_array[j][i] = gaze_data[j][i]
                std_array[j][i] = (gaze_data[j][i]-gaze_mean)**2
            else:
                new_array[j][i] = gaze_data[j][i]*radius / distance
                std_array[j][i] = (gaze_data[j][i]-gaze_mean)**2 * radius / distance
    
    mean_value = new_array.mean()
    std_value = std_array.mean()
    
    print(filename, ": ", mean_value)
    print(filename, ": ", std_value)
    
    all_array.append(mean_value)
    all_array_std.append(std_value)

print("mean:", np.array(all_array).mean())
print("std:", np.array(all_array_std).mean())
