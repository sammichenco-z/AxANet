import os
from os.path import join
import pickle
import shutil
from tqdm import tqdm
from joblib import Parallel, delayed

raw_video_path = "/DATA_EDS/lpf/bddoia_project/bddoia_data/video/raw_images/test_frames"
save_path = "/DATA_EDS/zyp/jinbu/datasets/oia_test_raw_images/"

filenames = os.listdir(raw_video_path)
filenames.sort()

# for filename in tqdm(filenames):
def move_dir(filename):

    # find the video path in our machine
    if filename.endswith("0064.jpg"):
        old_path = join(raw_video_path, filename)
        new_path = join(save_path, filename)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.copyfile(old_path, new_path)
        print("*************** "+new_path+" ***************")


Parallel(n_jobs=20)(delayed(move_dir)(filename) for filename in tqdm(filenames))