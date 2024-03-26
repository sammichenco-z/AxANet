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
import os
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
import h5py
# video_transforms & volume_transforms from https://github.com/hassony2/torch_videovision
from .data_utils.video_transforms import Compose, Resize, RandomCrop, ColorJitter, Normalize, CenterCrop, RandomHorizontalFlip, RandomResizedCrop
from .data_utils.volume_transforms import ClipToTensor
from .data_utils.video_bbox_transforms import video_bbox_prcoess
import code, time
import pickle
import pandas as pd


class DRAMA_DATASET(object):
    def __init__(self, args, root_dir, tokenizer, tensorizer, is_train=True, mode='train'):

        self.args = args
        self.tokenizer = tokenizer
        self.tensorizer = tensorizer
        self.mode = mode

        self.root_dir = root_dir
        self.raw_video_path = "data/raw_videos/"


        assert op.exists(self.root_dir)
        assert op.exists(self.raw_video_path)

        if not args.openset:
            file_list = os.listdir(self.root_dir)
            file_list.sort()

            self.filenames = []
            self.filenames += pd.read_csv(op.join(self.root_dir, "Vehicle.csv"),sep=',',header=None,usecols=[0])[0].tolist()
            self.filenames += pd.read_csv(op.join(self.root_dir, "Cyclist.csv"),sep=',',header=None,usecols=[0])[0].tolist()
            self.filenames += pd.read_csv(op.join(self.root_dir, "Pedestrian.csv"),sep=',',header=None,usecols=[0])[0].tolist()
            self.filenames += pd.read_csv(op.join(self.root_dir, "Infrastructure.csv"),sep=',',header=None,usecols=[0])[0].tolist()
        else:
            if is_train:
                self.filenames = []
                self.filenames += pd.read_csv(op.join(self.root_dir, "Vehicle.csv"),sep=',',header=None,usecols=[0])[0].tolist()
                self.filenames += pd.read_csv(op.join(self.root_dir, "Cyclist.csv"),sep=',',header=None,usecols=[0])[0].tolist()
                # self.filenames += pd.read_csv(op.join(self.root_dir, "Pedestrian.csv"),sep=',',header=None,usecols=[0])[0].tolist()
                self.filenames += pd.read_csv(op.join(self.root_dir, "Infrastructure.csv"),sep=',',header=None,usecols=[0])[0].tolist()
            else:
                self.filenames = []
                self.filenames += pd.read_csv(op.join(self.root_dir, "Vehicle.csv"),sep=',',header=None,usecols=[0])[0].tolist()
                self.filenames += pd.read_csv(op.join(self.root_dir, "Cyclist.csv"),sep=',',header=None,usecols=[0])[0].tolist()
                self.filenames += pd.read_csv(op.join(self.root_dir, "Pedestrian.csv"),sep=',',header=None,usecols=[0])[0].tolist()
                self.filenames += pd.read_csv(op.join(self.root_dir, "Infrastructure.csv"),sep=',',header=None,usecols=[0])[0].tolist()

        self.filenames.sort()
        self.image_keys = [self.mode + '_' + filename for filename in self.filenames]

        with open(f"{args.data_dir}/{self.mode}.caption_for_dataloader.json", "r") as caption_file:
            self.captions = json.load(caption_file)

        self.detr_aug = getattr(args, 'detr_aug', False)

        self.is_composite = False

        if is_train:
            assert tokenizer is not None

        self.is_train = is_train

        self.is_train = is_train
        self.img_res = getattr(args, 'img_res', 224)
        self.patch_size = getattr(args, 'patch_size', 16)

        self.img_feature_dim = args.img_feature_dim
        self.decoder_num_frames = getattr(args, 'max_num_frames', 2)

        LOGGER.info(f'isTrainData: {self.is_train}\n[PyAV video parameters] '
                    f'Num of Frame: {self.decoder_num_frames}, ')

        if is_train==True:
            self.raw_video_crop_list = [
                Resize(self.img_res),
                Resize((self.img_res,self.img_res)),
                ClipToTensor(channel_nb=3),
                Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]
        else:
            self.raw_video_crop_list = [
                Resize(self.img_res),
                Resize((self.img_res,self.img_res)),
                ClipToTensor(channel_nb=3),
                Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]
        
        self.raw_video_prcoess = Compose(self.raw_video_crop_list)
        self.video_bbox_process = video_bbox_prcoess(is_train, self.img_res)


    def apply_augmentations(self, frames):

        # TODO: this should be changed when the levels are changed
        frames_num_to_return = self.decoder_num_frames

        # if failed to decode video, generate fake frames (should be corner case)
        if frames is None:
            frames = np.zeros((frames_num_to_return,self.img_res,self.img_res,3)).astype(np.uint8)
        # (T, C, H, W) -> (T, H, W, C), channel is RGB
        elif 'torch' in str(frames.dtype):
            frames = frames.numpy()
            frames = np.transpose(frames, (0, 2, 3, 1))
        else:
            frames = frames.astype(np.uint8)
            frames = np.transpose(frames, (0, 2, 3, 1))
        num_of_frames, height, width, channels = frames.shape

        frame_list = []
        for i in range(frames_num_to_return):
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
    
    def apply_bbox_augmentations(self, frames, bbox):

        # TODO: this should be changed when the levels are changed
        frames_num_to_return = self.decoder_num_frames

        # if failed to decode video, generate fake frames (should be corner case)
        if frames is None:
            frames = np.zeros((frames_num_to_return,self.img_res,self.img_res,3)).astype(np.uint8)
        # (T, C, H, W) -> (T, H, W, C), channel is RGB
        elif 'torch' in str(frames.dtype):
            frames = frames.numpy()
            frames = np.transpose(frames, (0, 2, 3, 1))
        else:
            frames = frames.astype(np.uint8)
            frames = np.transpose(frames, (0, 2, 3, 1))
        num_of_frames, height, width, channels = frames.shape

        frame_list = []
        for i in range(frames_num_to_return):
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
        crop_frames, bbox = self.video_bbox_process(frame_list, bbox)

        # (C x T x H x W) --> (T x C x H x W)
        crop_frames = crop_frames.permute(1, 0, 2, 3)
        return crop_frames, bbox


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


            gt_bbox     = self.bbox_xywh2xyxy(torch.tensor(pickle_file['bbox'], dtype=torch.float32))
            # Vehicle, Pedestrian, Infrastructure, Cyclist
            class_label = torch.tensor(pickle_file['label'])

            # find the video path in our machine
            imgs_path = pickle_file['image_path']
            assert (not imgs_path[-1].startswith('/')) and (imgs_path[-1].endswith('.png') or imgs_path[-1].endswith('.jpg'))
            imgs_path = [op.join(self.raw_video_path, i) for i in imgs_path]

            if len(imgs_path) == 1:
                raw_frames = np.array([cv2.resize(cv2.imread(imgs_path[-1]),(1024, 1024))[:,:,::-1], cv2.resize(cv2.imread(imgs_path[-1]),(1024, 1024))[:,:,::-1]])
            elif len(imgs_path) > 1:
                raw_frames = np.array([cv2.resize(cv2.imread(imgs_path[-1]),(1024, 1024))[:,:,::-1], cv2.resize(cv2.imread(imgs_path[-1]),(1024, 1024))[:,:,::-1]])
            else:
                raise Exception(f"the frame number in a video {imgs_path} is less than 1")

            # (T H W C) to (T C H W)
            raw_frames = (np.transpose(raw_frames, (0, 3, 1, 2)))

            caption_sample = pickle_file['caption'].replace('<startseq>', '').replace('<endseq>', '')
            assert caption_sample == caption, f"caption_sample: {caption_sample}, caption: {caption}" 
 

        # apply augmentation. frozen-in-time if the input is an image
        # preproc_frames: (T, C, H, W), C = 3, H = W = self.img_res, channel is RGB   
        if self.detr_aug:
            preproc_raw_frames, bbox_aug = self.apply_bbox_augmentations(raw_frames, gt_bbox)
            gt_bbox = bbox_aug
        else:
            preproc_raw_frames = self.apply_augmentations(raw_frames)


        # tokenize caption and generate attention maps
        # it will consider only # of visual tokens for building attention maps. # is args.max_img_seq_length 
        if isinstance(caption_sample, dict):
            caption = caption_sample["caption"]
        else:
            caption = caption_sample
            caption_sample = None

        example = self.tensorizer.tensorize_example_e2e(caption, preproc_raw_frames, text_meta=caption_sample)

        # preparing outputs
        meta_data = {}
        meta_data['caption'] = caption # raw text data, not tokenized
        meta_data['img_key'] = image_key
        meta_data['raw_image'] = raw_frames[0]

        example =  example + (gt_bbox, class_label)
        # return image_key, example, meta_data
        return image_key, example, meta_data


class INPAINTING_DRAMA_DATASET(object):
    def __init__(self, args, root_dir, tokenizer, tensorizer, is_train=True, mode='train'):

        self.args = args
        self.tokenizer = tokenizer
        self.tensorizer = tensorizer
        self.mode = mode

        self.root_dir = root_dir
        self.raw_video_path = "raw_videos/"
        self.inpaint_path = 'inpainted_img/train/'


        assert op.exists(self.root_dir)
        assert op.exists(self.raw_video_path)
        assert op.exists(self.inpaint_path)

        file_list = os.listdir(self.root_dir)
        self.filenames = file_list
        self.image_keys = [self.mode + '_' + filename for filename in self.filenames]

        self.detr_aug = getattr(args, 'detr_aug', False)

        self.is_composite = False

        if is_train:
            assert tokenizer is not None

        self.is_train = is_train

        self.is_train = is_train
        self.img_res = getattr(args, 'img_res', 224)
        self.patch_size = getattr(args, 'patch_size', 16)

        self.img_feature_dim = args.img_feature_dim
        self.decoder_num_frames = getattr(args, 'max_num_frames', 2)

        LOGGER.info(f'isTrainData: {self.is_train}\n[PyAV video parameters] '
                    f'Num of Frame: {self.decoder_num_frames}, ')

        if is_train==True:
            self.raw_video_crop_list = [
                Resize(self.img_res),
                Resize((self.img_res,self.img_res)),
                ClipToTensor(channel_nb=3),
                Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]
        else:
            self.raw_video_crop_list = [
                Resize(self.img_res),
                Resize((self.img_res,self.img_res)),
                ClipToTensor(channel_nb=3),
                Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]
        
        self.raw_video_prcoess = Compose(self.raw_video_crop_list)
        self.video_bbox_process = video_bbox_prcoess(is_train, self.img_res)


    def apply_augmentations(self, frames):

        # TODO: this should be changed when the levels are changed
        frames_num_to_return = self.decoder_num_frames

        # if failed to decode video, generate fake frames (should be corner case)
        if frames is None:
            frames = np.zeros((frames_num_to_return,self.img_res,self.img_res,3)).astype(np.uint8)
        # (T, C, H, W) -> (T, H, W, C), channel is RGB
        elif 'torch' in str(frames.dtype):
            frames = frames.numpy()
            frames = np.transpose(frames, (0, 2, 3, 1))
        else:
            frames = frames.astype(np.uint8)
            frames = np.transpose(frames, (0, 2, 3, 1))
        num_of_frames, height, width, channels = frames.shape

        frame_list = []
        for i in range(frames_num_to_return):
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
    
    def apply_bbox_augmentations(self, frames, bbox=None):

        # TODO: this should be changed when the levels are changed
        frames_num_to_return = self.decoder_num_frames

        # if failed to decode video, generate fake frames (should be corner case)
        if frames is None:
            frames = np.zeros((frames_num_to_return,self.img_res,self.img_res,3)).astype(np.uint8)
        # (T, C, H, W) -> (T, H, W, C), channel is RGB
        elif 'torch' in str(frames.dtype):
            frames = frames.numpy()
            frames = np.transpose(frames, (0, 2, 3, 1))
        else:
            frames = frames.astype(np.uint8)
            frames = np.transpose(frames, (0, 2, 3, 1))
        num_of_frames, height, width, channels = frames.shape

        frame_list = []
        for i in range(frames_num_to_return):
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
        crop_frames, bbox = self.video_bbox_process(frame_list, bbox)

        # (C x T x H x W) --> (T x C x H x W)
        crop_frames = crop_frames.permute(1, 0, 2, 3)
        return crop_frames, bbox


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
        inpaint_idx = filename.split('.')[0]

        inpaint_frames = np.array([cv2.imread(op.join(self.inpaint_path, inpaint_idx+'_1'+ '.jpg')),cv2.imread(op.join(self.inpaint_path,inpaint_idx+'_1'+ '.jpg'))])

        # (T H W C) to (T C H W)
        inpaint_frames = (np.transpose(inpaint_frames, (0, 3, 1, 2)))

        # raw_inpaint_frames = np.concatenate((raw_frames, inpaint_frames), axis=0)

        # apply augmentation. frozen-in-time if the input is an image
        # preproc_frames: (T, C, H, W), C = 3, H = W = self.img_res, channel is RGB   

        if self.detr_aug:
            preproc_inpaint_frames, _ = self.apply_bbox_augmentations(inpaint_frames)
        else:
            preproc_inpaint_frames = self.apply_augmentations(inpaint_frames)


        # return image_key, example, meta_data
        return preproc_inpaint_frames