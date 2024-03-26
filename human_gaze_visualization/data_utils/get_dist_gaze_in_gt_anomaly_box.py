import json
import os
import numpy as np

def bbox_xyxy2cxcywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # to [cx, cy, w, h] where cxcy=middle
    if len(x.shape) == 2:
        y = np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2])/2  # top left x
        y[:, 1] = (x[:, 1] + x[:, 3])/2  # top left y
        y[:, 2] = x[:, 2] - x[:, 0]   # bottom right x
        y[:, 3] = x[:, 3] - x[:, 1]   # bottom right y
    else:
        y = np.copy(x)
        y[0] = (x[0] + x[2])/2  # top left x
        y[1] = (x[1] + x[3])/2  # top left y
        y[2] = x[2] - x[0]  # bottom right x
        y[3] = x[3] - x[1]  # bottom right y
    return y


# human_path = "gaze_compare_10_18/old_mean/viz_gaze_230918115438_raw_200_20_0.1_0.9/Anomaly"
# human_path = "gaze_compare_10_18/new_mean/viz_gaze_230827103946_raw_200_20_0.1_0.9/Anomaly"

###
# human_path = "gaze_compare_10_18/old_mean/viz_gaze_230918115438_raw_200_20_0.1_0.9/Turn"
# human_path = "gaze_compare_10_18/old_mean/viz_gaze_230918115438_raw_200_20_0.1_0.9/Hazard"

# human_path = "gaze_compare_10_18/new_mean/viz_gaze_230827103946_raw_200_20_0.1_0.9/Turn"
human_path = "gaze_compare_10_18/new_mean/viz_gaze_230827103946_raw_200_20_0.1_0.9/Hazard"
###



# human_path = "gaze_compare_10_11_35_hazard_anomaly/old_mean/viz_gaze_230825115919_raw_200_35_0.1_0.9/Anomaly"
# human_path = "gaze_compare_10_11_35_hazard_anomaly/new_mean/viz_gaze_230827103946_raw_200_35_0.1_0.9/Anomaly"

# human_path = "gaze_compare_10_20/old_mean/viz_gaze_230918115438_raw_200_40_0.1_0.9/Anomaly"
# human_path = "gaze_compare_10_20/new_mean/viz_gaze_230827103946_raw_200_40_0.1_0.9/Anomaly"

# human_path = "gaze_compare_10_20/old_mean/viz_gaze_230918115438_raw_200_40_0.1_0.9/Turn"
# human_path = "gaze_compare_10_20/old_mean/viz_gaze_230918115438_raw_200_40_0.1_0.9/Hazard"

# human_path = "gaze_compare_10_20/new_mean/viz_gaze_230827103946_raw_200_40_0.1_0.9/Turn"
# human_path = "gaze_compare_10_20/new_mean/viz_gaze_230827103946_raw_200_40_0.1_0.9/Hazard"



# human_path = "gaze_compare_10_19/old_mean/viz_gaze_230918115438_raw_200_50_0.1_0.9/Anomaly"
# human_path = "gaze_compare_10_19/new_mean/viz_gaze_230827103946_raw_200_50_0.1_0.9/Anomaly"

gaze_paths = os.listdir(human_path)
gaze_paths.sort()

gt_path = "all_gts/anomaly_bbox_gt.json"
with open(gt_path, "r") as f:
    gts = json.load(f)

all_array = []
all_array_std = []
for filename in gaze_paths:
    if not filename.endswith(".png.npy"):
        continue
    
    gaze_path = os.path.join(human_path, filename)
    gaze_data = np.load(gaze_path)
    gaze_data = gaze_data/gaze_data.max()
    gaze_mean = gaze_data.mean()
    h, w = gaze_data.shape
    
    file_key = filename.replace(".png.npy", "")
    if file_key not in gts:
        continue
    
    gt = gts[file_key]
    bbox_xyxy = np.array([gt[0]*w, gt[1]*h, gt[2]*w, gt[3]*h])
    bbox_cxcywh = bbox_xyxy2cxcywh(bbox_xyxy)
    
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
