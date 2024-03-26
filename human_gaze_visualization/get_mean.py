import os
import json
from gazeheat import draw_heatmap
import json

for duration in ["t0_t1", "t1_t2", "t2_end"]:
    for split in ["new", "old"]:
        for task in ["Anomaly", "Hazard", "Turn"]:
            for dataset in ["Anomaly", "Hazard", "Turn"]:
                dir_path = f"gaze_compare_personalised_{duration}_35_hazard_anomaly/{split}_mean/viz_gaze_fixation_only_200_35_0.0_1.0/{task}"

                if dataset == "Anomaly":
                    file_name_list = "all_gts/anomaly_bbox_gt.json"
                if dataset == "Hazard":
                    file_name_list = "all_gts/drama_bbox_gt.json"
                if dataset == "Turn":
                    file_name_list = "all_gts/oia_gt.json"

                with open(file_name_list, "r") as f:
                    filenames = json.load(f).keys()

                average_result = []
                for filename in filenames:
                    gaze_json_path = os.path.join(dir_path, filename+".png"+".json")
                    if not os.path.exists(gaze_json_path):
                        continue

                    with open(gaze_json_path, "r") as f:
                        gaze_data = json.load(f)['gaze_data']
                    average_result += gaze_data

                save_path = f"mean_gaze/{split}_{dataset}_{task}_{duration}.jpg"

                heatmap = draw_heatmap(average_result, (512, 512), alpha=0.5, savefilename=save_path, imagefile="black.jpg", gaussianwh=200, gaussiansd=35)
