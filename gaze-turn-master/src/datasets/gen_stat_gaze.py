"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""
import torch
import torchvision.transforms as transforms
import cv2
import math
import json
from PIL import Image
import os.path as op
import numpy as np
from numpy.random import randint
import random

import base64
import json

import code, time
from IPython import embed
from tqdm import tqdm


class BDDOIA_dataset(object):
    def __init__(self, args, root, tokenizer, tensorizer=None, is_train=True, on_memory=False,test=True):

        self.args = args
        self.tokenizer = tokenizer
        self.tensorizer = tensorizer

        self.root = root
        
        self.raw_data = json.load(open(root))
        
        self.img_keys = list(self.raw_data.keys())
        #self.img_keys= ['005c4fd3-cb4d6287','0060b445-5acc00ed','007aeb45-c601742b',"00d79c0a-a2b85ca4","01041028-187a2d1f"]
        self.image_keys = self.img_keys

        self.is_train = is_train

        self.img_res = 224

        self.decoder_target_fps = 3
        self.decoder_num_frames = 2
        self.decoder_multi_thread_decode = False

        self.decoder_safeguard_duration = False
        self.is_composite = False

        self.gaze_pred = True
        self.detr_aug = False

    def get_gaze_path(self, path):
        path = path.replace('data/', 'gaze/')
        dirname = op.dirname(path)
        basename = op.basename(path)
        gaze_path = op.join(dirname, 'saliency', basename)
        return gaze_path

    def clip_read(self, key):
        root_dirname = op.dirname(self.root)

        imgs_list = []
        
        img_path = op.join(root_dirname, "data", key+'.jpg')
        # print(img_path)
        assert op.exists(img_path)
        
        if self.gaze_pred:
            gazes_list = []
            gaze_path = self.get_gaze_path(img_path)
            assert op.exists(gaze_path)
            gaze = cv2.imread(gaze_path)
            gaze = np.transpose(gaze, (2, 0, 1))
        else:
            gaze = None
        
        return gaze

    def __len__(self):
        return len(self.img_keys)

    def __getitem__(self, idx):
        img_key = self.img_keys[idx]

        gaze_frames = self.clip_read(img_key)
        
        return gaze_frames


if __name__ == '__main__':
    my_dataset = BDDOIA_dataset(None, 'datasets/train.json', None, None, True, 'train')

    frames = np.zeros_like(my_dataset.__getitem__(0)[0]).astype(np.int)
    for i in tqdm(range(len(my_dataset))):
        gaze_frames = my_dataset.__getitem__(i)
        frames = frames + gaze_frames

    mean_frame = frames/len(my_dataset)
    
    np.save('bddoia_gaze_mean_float.npy', np.transpose(mean_frame, (1,2,0)))
    cv2.imwrite("bddoia_gaze_mean.jpg", np.transpose(np.array(mean_frame, dtype=int), (1,2,0)))



