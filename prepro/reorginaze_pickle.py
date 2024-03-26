import base64
import code
import json
import math
import os
import os.path as op
import pickle
import random
import time

import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from numpy.random import randint
from PIL import Image
from tqdm import tqdm


class DRAMA_DATASET(object):
    def __init__(self, args, root_dir, tokenizer, tensorizer, is_train=True, mode='train'):

        self.args = args
        self.tokenizer = tokenizer
        self.tensorizer = tensorizer
        self.mode = mode

        self.root_dir = root_dir
        self.raw_video_path = "raw_videos/"


        assert op.exists(self.root_dir)
        assert op.exists(self.raw_video_path)

        file_list = os.listdir(self.root_dir)
        file_list.sort()
        self.filenames = file_list
        self.image_keys = [self.mode + '_' + filename for filename in self.filenames]

        with open(f"datasets/{self.mode}.caption_for_dataloader.json", "r") as caption_file:
            self.captions = json.load(caption_file)



    def bbox_xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] where xy=top-left
        # to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        if len(x.shape) == 2:
            y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
            y[:, 0] = x[:, 0]  # top left x
            y[:, 1] = x[:, 1]  # top left y
            y[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
            y[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
        else:
            y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
            y[0] = x[0]  # top left x
            y[1] = x[1]  # top left y
            y[2] = x[0] + x[2]  # bottom right x
            y[3] = x[1] + x[3]  # bottom right y
        return y

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        image_key = self.mode + '_' + filename
        filename = op.join(self.root_dir, filename)

        caption = self.captions[image_key]

        # these pickles are original pickles in drama, and we use the pickle to locate the raw images and inpainted images
        with open(filename, "rb") as f:
            pickle_file = pickle.load(f)

            # img = pickle_file['img']

            gt_bbox      = pickle_file['bbox']

            # find the video path in our machine
            img_path = pickle_file['img_path'].replace('/data/02/kdi/internal_datasets/drama/combined/', '')
            assert (not img_path.startswith('/')) and (img_path.endswith('.png') or img_path.endswith('.jpg'))
            video_path = op.dirname(op.join(self.raw_video_path, img_path))

            rgb_and_flow_frames = os.listdir(video_path)
            rgb_and_flow_frames.sort()
            rgb_frames = []
            for frame in rgb_and_flow_frames:
                if frame.startswith('frame_'):
                    rgb_frame_path = op.join(os.path.dirname(img_path), frame)
                    rgb_frames.append(rgb_frame_path)

            is_video = False
            caption_sample = pickle_file['caption'].replace('<startseq>', '').replace('<endseq>', '')
            assert caption_sample == caption, f"caption_sample: {caption_sample}, caption: {caption}" 
            tag = ''

            class_label = 0
            str_class_label = pickle.load(open(filename.replace("processed", "processed/processed_label"), "rb"))['class_label']
            if   str_class_label == 'Vehicle':
                class_label = 0
            elif str_class_label == 'Pedestrian':
                class_label = 1
            elif str_class_label == 'Infrastructure':
                class_label = 2
            elif str_class_label == 'Cyclist':
                class_label = 3


            data = dict()
            data['image_path'] = rgb_frames
            data['caption'] = caption_sample
            data['bbox'] = gt_bbox
            data['label'] = class_label
            data['label_name'] = str_class_label

        return filename, data

save_dir = 'datasets'

for mode in ['train', 'val', 'test']:
    mode_root = os.path.join(save_dir, mode)
    os.makedirs(mode_root, exist_ok=True)
    my_dataset = DRAMA_DATASET({}, f"/DATA_EDS/zyp/jinbu/processed/{mode}/", None, None, True, mode)
    for i in tqdm(range(len(my_dataset))):
        filename, data = my_dataset.__getitem__(i)
        file_name = str(i).zfill(6)+'.pkl'
        assert os.path.basename(filename) == file_name
        with open(os.path.join(mode_root, file_name), "wb") as f:
            pickle.dump(data, f)
