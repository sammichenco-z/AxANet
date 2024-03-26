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
import os

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
        self.image_keys = self.img_keys

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
        self.is_composite = False

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
        
        
        if is_train==True:
            self.raw_video_gaze_crop_list = [
                Resize(self.img_res),
                Resize((self.img_res,self.img_res)),
                ClipToTensor(channel_nb=3),
                Normalize(mean=[0.485,0.456,0.406],std=[0.229, 0.224, 0.225])
            ]
        else:
            self.raw_video_gaze_crop_list = [
                Resize(self.img_res),
                Resize((self.img_res,self.img_res)),
                ClipToTensor(channel_nb=3),
                Normalize(mean=[0.485,0.456,0.406],std=[0.229, 0.224, 0.225])
            ]            
        self.raw_video_gaze_prcoess = Compose(self.raw_video_gaze_crop_list)
        
        
        
        self.video_bbox_process = video_bbox_prcoess(is_train, self.img_res, self.gaze_pred)

        if self.gaze_pred:
            self.mean_gaze_type = 'div'
            
            self.mean_gaze_float = np.load('datasets/bddoia_gaze_mean_float.npy')
            self.mean_gaze_float = np.transpose(self.mean_gaze_float[np.newaxis, ...], (0, 3, 1, 2))
            
            self.mean_gaze = cv2.imread('datasets/bddoia_gaze_mean.jpg')
            self.mean_gaze = np.transpose(self.mean_gaze[np.newaxis, ...], (0, 3, 1, 2))
            
            ## normalize
            self.mean_gaze = self.mean_gaze.astype(np.float)
            self.mean_gaze = self.mean_gaze / self.mean_gaze.max() * 255
            self.mean_gaze = self.mean_gaze.astype(np.uint8)

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

    def apply_augmentations_gaze(self, frames):
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
        crop_frames = self.raw_video_gaze_prcoess(frame_list)
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

    def get_gaze_path(self, path):
        path = path.replace('data/', 'gaze/')
        dirname = op.dirname(path)
        basename = op.basename(path)
        gaze_path = op.join(dirname, 'saliency', basename)
        return gaze_path

    def clip_read(self, key):
        root_dirname = op.dirname(self.root)

        imgs_list = []
        img = np.zeros(((720, 1280, 3)), dtype=np.int8)
        
        img_path = op.join(root_dirname, "data", key+'.jpg')
        # print(img_path)
        assert op.exists(img_path)
        img = cv2.imread(img_path)
        img = img[:,:,::-1]
        img = np.transpose(img[np.newaxis, ...], (0, 3, 1, 2))
        imgs_list.append(img)
        imgs_list.append(img)
        imgs_np = np.vstack(imgs_list)
        
        if self.gaze_pred:
            gazes_list = []
            gaze_path = self.get_gaze_path(img_path)
            assert op.exists(gaze_path)
            ori_gaze = cv2.imread(gaze_path)
            ori_gaze = np.transpose(ori_gaze[np.newaxis, ...], (0, 3, 1, 2))
            if self.mean_gaze_type == 'sub':
                gaze = ori_gaze.astype(np.int) - self.mean_gaze.astype(np.int)
                gaze = np.clip(gaze, 0,255).astype(np.uint8)
                gaze = gaze / gaze.max() * 255
                gaze = gaze.astype(np.uint8)
            elif self.mean_gaze_type == 'div':
                gaze = ori_gaze.astype(np.float) * (1-self.mean_gaze.astype(np.float)/255)
                gaze = np.clip(gaze, 0,255).astype(np.uint8)
                gaze = gaze / gaze.max() * 255
                gaze = gaze.astype(np.uint8)
            else:
                gaze = ori_gaze

            ###### visualize ########
            gaze_img = np.transpose(np.concatenate([ori_gaze[0], self.mean_gaze[0], gaze[0]], axis=1), (1,2,0))
            gaze_img = cv2.applyColorMap(gaze_img, cv2.COLORMAP_JET)
            rgb_img = np.transpose(np.concatenate([img[0], img[0], img[0]], axis=1), (1,2,0))
            gaze_img = cv2.addWeighted(rgb_img,0.5,gaze_img,0.5,0)
            cv2.imwrite(os.path.join('./div_gaze_vis', key+'.jpg'), gaze_img)
            # embed()
            ###### visualize ########

            gazes_list.append(gaze)
            gazes_list.append(gaze)
            gazes_np = np.vstack(gazes_list)
        else:
            gazes_np = None
        
        return imgs_np, gazes_np

    def __len__(self):
        return len(self.img_keys)

    def __getitem__(self, idx):
        if self.args.debug_speed:
            idx = idx % self.args.effective_batch_size
        img_key = self.img_keys[idx]

        raw_frames, gaze_frames = self.clip_read(img_key)
        
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
            preproc_gaze_frames = self.apply_augmentations_gaze(gaze_frames)

        # preparing outputs
        meta_data = {}
        meta_data['img_key'] = img_key
        meta_data['raw_frame'] = raw_frames
        example =  (preproc_frames, label_action, label_reason, preproc_gaze_frames)

        return img_key, example, meta_data
