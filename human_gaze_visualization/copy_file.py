import os
from os.path import join
import pickle
import shutil
from tqdm import tqdm
from joblib import Parallel, delayed

raw_video_path = "/DATA_EDS/zyp/jinbu/datasets/drama/combined/"
data_path = "/DATA_EDS/zyp/jinbu/datasets/drama/my_processed/test/"
save_path = "/DATA_EDS/zyp/jinbu/datasets/drama_raw_images/test/"

viz_path = '/DATA_EDS/zyp/jinbu/datasets/drama/viz_output/'
raw_video_path = '/DATA_EDS/zyp/jinbu/datasets/drama/combined/'
gaze_path = '/DATA_EDS/lpf/gaze/dataset/drama_gaze_pred/'

filenames = os.listdir(data_path)
filenames.sort()

# for filename in tqdm(filenames):
def move_dir(filename):
    with open(join(data_path, filename), "rb") as f:
        pickle_file = pickle.load(f)

        # find the video path in our machine
        img_paths = [pickle_file['image_path'][-1]]
        for img_path in img_paths:
            old_path = join(raw_video_path, img_path)
            new_path = join(save_path, img_path.replace("/", "_").replace(".png", "_raw.png"))
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.copyfile(old_path, new_path)
            print("*************** "+new_path+" ***************")

            old_gaze_path = join(gaze_path, os.path.dirname(img_path), "gaze_heatmap", os.path.basename(img_path))
            new_gaze_path = join(save_path, img_path.replace("/", "_").replace(".png", "_gaze.png"))
            shutil.copyfile(old_gaze_path, new_gaze_path)

            if img_path.startswith("titan/"):
                old_viz_path = join(viz_path, img_path.replace("titan/", "").replace("/", "_"))
            else:
                tmp = img_path.split("/")
                old_viz_path = join(viz_path, tmp[0]+"_"+tmp[-1])

            new_viz_path = join(save_path, img_path.replace("/", "_").replace(".png", "_viz.png"))
            shutil.copyfile(old_viz_path, new_viz_path)

# move_dir(filenames[0])
Parallel(n_jobs=20)(delayed(move_dir)(filename) for filename in tqdm(filenames))