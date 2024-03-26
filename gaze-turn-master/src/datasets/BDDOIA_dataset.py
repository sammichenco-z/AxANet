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
from src.utils.tsv_file import TSVFile, CompositeTSVFile
from src.utils.tsv_file_ops import tsv_reader
from src.utils.load_files import load_linelist_file, load_from_yaml_file, find_file_path_in_yaml, load_box_linelist_file
from .data_utils.image_ops import img_from_base64
from .data_utils.video_ops import extract_frames_from_video_binary, extract_frames_from_video_path
from src.utils.logger import LOGGER
import base64
import json

# video_transforms & volume_transforms from https://github.com/hassony2/torch_videovision
from .data_utils.video_transforms import Compose, Resize, RandomCrop, ColorJitter, Normalize, CenterCrop, RandomHorizontalFlip, RandomResizedCrop
from .data_utils.volume_transforms import ClipToTensor
from .data_utils.video_bbox_transforms import video_bbox_prcoess
import code, time

from IPython import embed

class BDDOIA_dataset(object):
    def __init__(self, args, root, tokenizer, tensorizer=None, is_train=True, on_memory=False,test=True):

        self.args = args
        self.tokenizer = tokenizer
        self.tensorizer = tensorizer

        self.root = root
        
        self.raw_data = json.load(open(root))
        
        self.img_keys = list(self.raw_data.keys())
        #self.img_keys= ['005c4fd3-cb4d6287','0060b445-5acc00ed','007aeb45-c601742b',"00d79c0a-a2b85ca4","01041028-187a2d1f"]
        

        if is_train:
            assert tokenizer is not None

        self.is_train = is_train

        self.img_res = getattr(args, 'img_res', 224)
        self.patch_size = getattr(args, 'patch_size', 16)

        self.img_feature_dim = args.img_feature_dim
        self.decoder_target_fps = 3
        self.decoder_num_frames = getattr(args, 'max_num_frames', 2)
        self.decoder_multi_thread_decode = False

        self.decoder_safeguard_duration = False

        self.gaze_pred = getattr(args, 'gaze_pred', False)
        self.detr_aug = getattr(args, 'detr_aug', False)
        print('detr_aug:', self.detr_aug)

        # use uniform sampling as default for now
        self.decoder_sampling_strategy = getattr(args, 'decoder_sampling_strategy', 'uniform')
        LOGGER.info(f'isTrainData: {self.is_train}\n[PyAV video parameters] '
                    f'Num of Frame: {self.decoder_num_frames}, '
                    f'FPS: {self.decoder_target_fps}, '
                    f'Sampling: {self.decoder_sampling_strategy}')
        # Initialize video transforms
        # adapt from https://github.com/hassony2/torch_videovision

        if is_train==True:
            self.raw_video_crop_list = [
                Resize(self.img_res),
                Resize((self.img_res,self.img_res)),
                ClipToTensor(channel_nb=3),
                Normalize(mean=[0.485,0.456,0.406],std=[0.229, 0.224, 0.225])
            ]
        else:
            self.raw_video_crop_list = [
                Resize(self.img_res),
                Resize((self.img_res,self.img_res)),
                ClipToTensor(channel_nb=3),
                Normalize(mean=[0.485,0.456,0.406],std=[0.229, 0.224, 0.225])
            ]            
        self.raw_video_prcoess = Compose(self.raw_video_crop_list)
        self.video_bbox_process = video_bbox_prcoess(is_train, self.img_res, self.gaze_pred)

    def apply_augmentations(self, frames):
        # if failed to decode video, generate fake frames (should be corner case)
        if frames is None:
            frames = np.zeros((self.decoder_num_frames,self.img_res,self.img_res,3)).astype(np.uint8)
        # (T, C, H, W) -> (T, H, W, C), channel is RGB
        elif 'torch' in str(frames.dtype):
            frames = frames.numpy()
            frames = np.transpose(frames, (0, 2, 3, 1))
        else:
            frames = frames.astype(np.uint8)
            frames = np.transpose(frames, (0, 2, 3, 1))
        num_of_frames, height, width, channels = frames.shape

        frame_list = []
        for i in range(self.decoder_num_frames):
            if num_of_frames==1: 
                # if it is from image-caption dataset, we duplicate the image
                # convert numpy to PIL format, compatible to augmentation operations
                frame_list.append(Image.fromarray(frames[0]))
            else:
                # if it is from video-caption dataset, we add each frame to the list
                # convert numpy to PIL format, compatible to augmentation operations
                frame_list.append(Image.fromarray(frames[i]))
        
        # adapt from torch_videovision: https://github.com/hassony2/torch_videovision
        # after augmentation, output tensor (C x T x H x W) in the range [0, 1.0]
        crop_frames = self.raw_video_prcoess(frame_list)
        # (C x T x H x W) --> (T x C x H x W)
        crop_frames = crop_frames.permute(1, 0, 2, 3)
        return crop_frames 

    def apply_bbox_augmentations(self, frames, seed=None, is_gaze=False):

        if seed is None:
            seed = randint(0, 100000)
        random.seed(seed)
        np.random.seed(seed)

        # print(random.randint(0, 100000))
        # print(randint(0, 100000))
        # if failed to decode video, generate fake frames (should be corner case)
        if frames is None:
            frames = np.zeros((self.decoder_num_frames,self.img_res,self.img_res,3)).astype(np.uint8)
        # (T, C, H, W) -> (T, H, W, C), channel is RGB
        elif 'torch' in str(frames.dtype):
            frames = frames.numpy()
            frames = np.transpose(frames, (0, 2, 3, 1))
        else:
            frames = frames.astype(np.uint8)
            frames = np.transpose(frames, (0, 2, 3, 1))
        num_of_frames, height, width, channels = frames.shape

        frame_list = []
        for i in range(self.decoder_num_frames):
            if num_of_frames==1: 
                # if it is from image-caption dataset, we duplicate the image
                # convert numpy to PIL format, compatible to augmentation operations
                frame_list.append(Image.fromarray(frames[0]))
            else:
                # if it is from video-caption dataset, we add each frame to the list
                # convert numpy to PIL format, compatible to augmentation operations
                frame_list.append(Image.fromarray(frames[i]))
        
        # adapt from torch_videovision: https://github.com/hassony2/torch_videovision
        # after augmentation, output tensor (C x T x H x W) in the range [0, 1.0]
        bbox = torch.rand(4) # unused
        crop_frames, _ = self.video_bbox_process(frame_list, bbox=bbox, is_gaze=is_gaze)
        # (C x T x H x W) --> (T x C x H x W)
        crop_frames = crop_frames.permute(1, 0, 2, 3)
        return crop_frames

    def get_gaze_path(self, path, mode):
        dirname = path.split('/')[0]
        basename = op.basename(path)
        gaze_path = op.join(dirname, 'gaze_video/BDDoia/BDDA', mode, 'saliency', basename)
        return gaze_path

    def clip_read(self, key):

        root_dirname = op.dirname(self.root)
        if 'train' in op.basename(self.root):
            root_basename = 'train_frames'
            mode = 'train'
        elif 'val' in op.basename(self.root):
            root_basename = 'val_frames'
            mode = 'val'
        elif 'test' in op.basename(self.root):
            root_basename = 'test_frames'
            mode = 'test'

        def sampling(start,end,n):
            if n == 1:
                return [int(round((start+end)/2.))]
            if n < 1:
                raise Exception("behaviour not defined for n<2")
            step = (end-start)/float(n-1)
            return [int(round(start+x*step)) for x in range(n)]
        
        sample_choices = sampling(0, 63, self.decoder_num_frames)
        
        imgs_list = []
        # TODO: FIX DATASETS BUG
        img = np.zeros(((720, 1280, 3)), dtype=np.int8)
        N=0
        for i in sample_choices:
            img_path = op.join(root_dirname, "raw_images", root_basename, key+f'_frame{str(i+1).zfill(4)}.jpg')
            if op.exists(img_path):
                img = cv2.imread(img_path)
                img = img[:,:,::-1]
                img = np.transpose(img[np.newaxis, ...], (0, 3, 1, 2))
                imgs_list.append(img)
                N=i
            else:
                # imgs_list.append(img)
                assert False
        imgs_np = np.vstack(imgs_list)
        
        if self.gaze_pred:
            gazes_list = []
            
            for i in sample_choices:
                img_path = op.join(root_dirname, "raw_images", root_basename, key+f'_frame{str(i+1).zfill(4)}.jpg')
                gaze_path = self.get_gaze_path(img_path, mode)
                if op.exists(gaze_path):
                    img = cv2.imread(gaze_path)
                    img = img[:,:,::-1]
                    img = np.transpose(img[np.newaxis, ...], (0, 3, 1, 2))
                    gazes_list.append(img)
                else:
                    # imgs_list.append(img)
                    assert False
            gazes_np = np.vstack(gazes_list)
        else:
            gazes_np = None
        
        return imgs_np, gazes_np

    def __len__(self):
        return len(self.img_keys)

    def dot_image(self, raw_frames, gaze_frames):
        gaze_raw_frames = raw_frames.copy()
        gaze_raw_frames = gaze_raw_frames.astype(np.float64)
        gaze_raw_frames = gaze_raw_frames * (gaze_frames.astype(np.float64) / 255)
        gaze_raw_frames = gaze_raw_frames.astype(np.uint8)
        return gaze_raw_frames
        
    def __getitem__(self, idx):
        if self.args.debug_speed:
            idx = idx % self.args.effective_batch_size
        img_key = self.img_keys[idx]

        raw_frames, gaze_frames = self.clip_read(img_key)
        
        gaze_frames = self.dot_image(raw_frames, gaze_frames)
        
        labels = self.raw_data[img_key]
        assert len(labels['actions']) == 4
        label_action = torch.as_tensor(labels['actions'])
        label_reason = torch.as_tensor(labels['reason'])

        # apply augmentation. frozen-in-time if the input is an image
        # preproc_frames: (T, C, H, W), C = 3, H = W = self.img_res, channel is RGB   
        if self.detr_aug:
            my_seed_for_this_iter = randint(0, 100000000)
            preproc_frames = self.apply_bbox_augmentations(raw_frames, seed=my_seed_for_this_iter, is_gaze=False)
            preproc_gaze_frames = self.apply_bbox_augmentations(gaze_frames, seed=my_seed_for_this_iter, is_gaze=True)
        else:
            preproc_frames = self.apply_augmentations(raw_frames)
            preproc_gaze_frames = self.apply_augmentations(gaze_frames)

        # preparing outputs
        meta_data = {}
        meta_data['img_key'] = img_key
        meta_data['raw_frame'] = raw_frames[-1]
        example =  (preproc_frames, label_action, label_reason, preproc_gaze_frames)

        return img_key, example, meta_data
