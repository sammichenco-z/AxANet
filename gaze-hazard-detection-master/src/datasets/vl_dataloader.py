"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""
import os.path as op
from tkinter import E
import torch
from src.utils.comm import get_world_size
from .drama_dataset import (DRAMA_DATASET, INPAINTING_DRAMA_DATASET)
from .caption_tensorizer import build_tensorizer
from .data_sampler import DistributedSamplerLimited, NodeSplitSampler
from src.utils.logger import LOGGER as logger


def drama_dataset(args, tokenizer, tensorizer, mode='train', is_train=True):
    logger.info(f'data_dir:{args.data_dir}')

    if mode == 'unlabeled':
        dataset_dir = op.join(args.data_dir, "train")
        return INPAINTING_DRAMA_DATASET(args, dataset_dir, tokenizer, tensorizer, is_train, mode)
    else:
        if mode == 'train':
            dataset_dir = op.join(args.data_dir, "train")
        elif mode == 'val':
            dataset_dir = op.join(args.data_dir, "val")
        elif mode == 'test':
            dataset_dir = op.join(args.data_dir, "test")
        else:
            raise Exception("running mode is not valid")

        return DRAMA_DATASET(args, dataset_dir, tokenizer, tensorizer, is_train, mode)

def build_dataset(args, tokenizer, mode='train', is_train=True):
    tensorizer = build_tensorizer(args, tokenizer, is_train=is_train)
    return drama_dataset(args, tokenizer, tensorizer, mode=mode, is_train=is_train)


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


def make_data_loader(args, tokenizer, is_distributed=True, mode='train',
        is_train=True, start_iter=0, num_gpus=8):

    dataset = build_dataset(args, tokenizer, mode=mode, is_train=is_train)
    a = dataset.__getitem__(100)
    # a,b,c = dataset.__getitem__(100)
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