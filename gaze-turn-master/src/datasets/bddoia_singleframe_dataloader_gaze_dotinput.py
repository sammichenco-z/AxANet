"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""
import os.path as op
from tkinter import E
import torch
from src.utils.comm import get_world_size
# from .BDDOIA_s_dataset import (BDDOIA_dataset)
from .caption_tensorizer import build_tensorizer
from .data_sampler import DistributedSamplerLimited, NodeSplitSampler
from src.utils.logger import LOGGER as logger

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

    def get_gaze_path(self, path):
        path = path.replace('data/', 'gaze/')
        dirname = op.dirname(path)
        basename = op.basename(path)
        gaze_path = op.join(dirname, 'saliency', basename)
        return gaze_path

    def clip_read(self, key):
        root_dirname = op.dirname(self.root)

        imgs_list = []
        # img = np.zeros(((720, 1280, 3)), dtype=np.uint8)
        
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
            gaze = cv2.imread(gaze_path)
            gaze = np.transpose(gaze[np.newaxis, ...], (0, 3, 1, 2))
            gazes_list.append(gaze)
            gazes_list.append(gaze)
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
            preproc_gaze_frames = self.apply_bbox_augmentations(gaze_frames, seed=my_seed_for_this_iter, is_gaze=False) # !!!!!!!!!
        else:
            preproc_frames = self.apply_augmentations(raw_frames)
            preproc_gaze_frames = self.apply_augmentations(gaze_frames)

        # preparing outputs
        meta_data = {}
        meta_data['img_key'] = img_key
        example =  (preproc_frames, label_action, label_reason, preproc_gaze_frames)

        return img_key, example, meta_data



def drama_dataset(args, tokenizer, tensorizer,mode='train', is_train=True):
    logger.info(f'data_dir:{args.data_dir}')

    root = op.join(args.data_dir, f"{mode}.json")
    
    return BDDOIA_dataset(args, root, tokenizer, tensorizer, is_train, mode)

def build_dataset(args, tokenizer, mode='train',  is_train=True):
    tensorizer = build_tensorizer(args, tokenizer, is_train=is_train)
    return drama_dataset(args, tokenizer, tensorizer,mode=mode, is_train=is_train)


class IterationBasedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def make_batch_data_sampler(sampler, images_per_gpu, num_iters=None, start_iter=0):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_gpu, drop_last=False
    )
    if num_iters is not None and num_iters >= 0:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_sampler(dataset, shuffle, distributed, random_seed, limited_samples=-1):
    if distributed:
        if dataset.is_composite:
            # first_epoch_skip_shuffle not working yet
            logger.info("Enable NodeSplitSampler with first_epoch_skip_shuffle=True")
            return NodeSplitSampler(
                dataset, shuffle=shuffle, random_seed=random_seed,
                first_epoch_skip_shuffle=True)
        elif limited_samples < 1:
            return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle, seed=random_seed)
        else:  # use limited distributed sampler
            return DistributedSamplerLimited(dataset, shuffle=shuffle, limited=limited_samples)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_data_loader(args, tokenizer,  is_distributed=True, mode='train',
        is_train=True, start_iter=0, num_gpus=8):

    dataset = build_dataset(args, tokenizer,  mode=mode, is_train=is_train)
    #a,b,c = dataset.__getitem__(100)
    # a,b,c = dataset.__getitem__(1100)
    if is_train==True:
        shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
    else:
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size
        num_iters = None
        start_iter = 0

    if hasattr(args, 'limited_samples'):
        limited_samples = args.limited_samples // num_gpus
    else:
        limited_samples = -1
    random_seed = args.seed
    sampler = make_data_sampler(
        dataset, shuffle, is_distributed, limited_samples=limited_samples,
        random_seed=random_seed)
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, num_iters, start_iter
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_sampler=batch_sampler,
        pin_memory=False, worker_init_fn=init_seeds,
    )
    return data_loader

def init_seeds(seed=88):
    import os, random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)