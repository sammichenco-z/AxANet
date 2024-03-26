from __future__ import absolute_import, division, print_function

import os
import sys

pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)
import datetime
import gc
import json
import os.path as op
import pickle
import random
import time

import cv2
import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from numpy.random import randint
from PIL import Image
from src.configs.config import (basic_check_arguments,
                                restore_training_settings, shared_configs)
from src.datasets.data_utils.video_bbox_transforms import video_bbox_prcoess
from src.datasets.data_utils.video_transforms import (CenterCrop, ColorJitter,
                                                      Compose, Normalize,
                                                      RandomCrop,
                                                      RandomHorizontalFlip,
                                                      RandomResizedCrop,
                                                      Resize)
from src.datasets.data_utils.volume_transforms import ClipToTensor
from src.datasets.vl_dataloader import make_data_loader
from src.evalcap.utils_caption_evaluate import (
    bbox_eval, evaluate_on_coco_caption, two_cap_evaluate_on_coco_caption)
from src.modeling.drama_bbox import DRAMAVideoTransformer
from src.modeling.drama_bbox_tr_with_clip_caption import DRAMAVideoTransformerTRGTCAPTION
from src.modeling.load_bert import get_bert_model
from src.modeling.load_swin import get_swin_model, reload_pretrained_swin
from src.pytorch_grad_cam.grad_cam import GradCAM
from src.pytorch_grad_cam.utils.image import (preprocess_image,
                                              show_cam_on_image)
from src.solver import AdamW, WarmupLinearLR
from src.utils.comm import dist_init, get_rank, get_world_size, is_main_process
from src.utils.deepspeed import fp32_to_fp16, get_deepspeed_config
from src.utils.load_save import TrainingRestorer, TrainingSaver
from src.utils.logger import LOGGER as logger
from src.utils.logger import TB_LOGGER, RunningMeter, add_log_to_file
from src.utils.metric_logger import MetricLogger
from src.utils.miscellaneous import (NoOp, concat_tsv_files, delete_tsv_files,
                                     mkdir, set_seed, str_to_bool)
from src.utils.tsv_file_ops import (double_tsv_writer, reorder_tsv_keys,
                                    tsv_writer)
from tqdm import tqdm
from src.modeling.drama_detr_bbox import DRAMADetr


def reshape_transform(tensor, frame=1, height=7, width=7):
    result = tensor.reshape(1, frame,
                            height, width, tensor.size(-1))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.permute(0, 4, 1, 2, 3)
    return result

def inference(args, img_key, video_data, model, gt_bbox, class_label):
    model.float()
    model.eval()
    preproc_frames = video_data.to(args.device)
    gt_bbox = gt_bbox.to(args.device)
    class_label = class_label.to(args.device)

    cam = GradCAM(model=model,
                  target_layers=[model.detr_encoder.transformer.encoder.layers[-1].norm2],
                  use_cuda=True,
                  reshape_transform=reshape_transform
                )

    if True:
        inputs = {'is_decode': True,
            'img_feats': preproc_frames,
            'bbox': gt_bbox,
            'label': class_label,
            'caption': None,
        }
        tic = time.time()


        grayscale_cam, raw_outputs = cam(input_tensor=inputs,
                        targets=None,
                        eigen_smooth=False,
                        aug_smooth=False)

        time_meter = time.time() - tic
        logger.info(f"Grad cam computing time: {time_meter} seconds")

        # Here grayscale_cam has only one video in the batch
        grayscale_cam = grayscale_cam[0, :]

    return grayscale_cam, raw_outputs


    

def check_arguments(args):
    # shared basic checks
    basic_check_arguments(args)
    # additional sanity check:
    args.max_img_seq_length = int((args.max_num_frames/2)*(int(args.img_res)/32)*(int(args.img_res)/32))
    
    if args.freeze_backbone or args.backbone_coef_lr == 0:
        args.backbone_coef_lr = 0
        args.freeze_backbone = True
    
    if 'reload_pretrained_swin' not in args.keys():
        args.reload_pretrained_swin = False

    if not len(args.pretrained_checkpoint) and args.reload_pretrained_swin:
        logger.info("No pretrained_checkpoint to be loaded, disable --reload_pretrained_swin")
        args.reload_pretrained_swin = False

    if args.learn_mask_enabled==True and args.attn_mask_type != 'learn_without_crossattn' and args.attn_mask_type != 'learn_with_swap_crossattn': 
        args.attn_mask_type = 'learn_vid_att'

def update_existing_config_for_inference(args):
    ''' load swinbert args for evaluation and inference 
    '''
    assert args.do_test or args.do_eval
    checkpoint = args.eval_model_dir
    try:
        json_path = op.join(checkpoint, os.pardir, 'log', 'args.json')
        f = open(json_path,'r')
        json_data = json.load(f)

        from easydict import EasyDict
        train_args = EasyDict(json_data)
    except Exception as e:
        train_args = torch.load(op.join(checkpoint, 'training_args.bin'))

    train_args.eval_model_dir = args.eval_model_dir
    train_args.resume_checkpoint = args.eval_model_dir + 'model.bin'
    train_args.model_name_or_path = 'models/captioning/bert-base-uncased/'
    train_args.do_train = False
    train_args.do_eval = True
    train_args.do_test = True
    train_args.val_yaml = args.val_yaml
    train_args.test_video_fname = args.test_video_fname
    train_args.use_car_sensor = True if hasattr(args, 'use_car_sensor') and args.use_car_sensor else False
    train_args.multitask = True if hasattr(args, 'multitask') and args.multitask else False
    return train_args

def get_custom_args(base_config):
    parser = base_config.parser
    parser.add_argument('--max_num_frames', type=int, default=32)
    parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument("--grid_feat", type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument("--kinetics", type=str, default='400', help="400 or 600")
    parser.add_argument("--pretrained_2d", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument("--vidswin_size", type=str, default='base')
    parser.add_argument('--freeze_backbone', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_checkpoint', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--backbone_coef_lr', type=float, default=0.001)
    parser.add_argument("--reload_pretrained_swin", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--learn_mask_enabled', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--loss_sparse_w', type=float, default=0)
    parser.add_argument('--loss_sensor_w', type=float, default=0)
    parser.add_argument('--sparse_mask_soft2hard', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--transfer_method', type=int, default=-1,
                        help="0: load all SwinBERT pre-trained weights, 1: load only pre-trained sparse mask")
    parser.add_argument('--att_mask_expansion', type=int, default=-1,
                        help="-1: random init, 0: random init and then diag-based copy, 1: interpolation")
    parser.add_argument('--resume_checkpoint', type=str, default='None')
    parser.add_argument('--test_video_fname', type=str, default='None')
    args = base_config.parse_args()
    return args

def apply_augmentations(frames):

    if True:
        # TODO: this should be changed when the levels are changed
        frames_num_to_return = 2

        # if failed to decode video, generate fake frames (should be corner case)
        if frames is None:
            frames = np.zeros((frames_num_to_return,224,224,3)).astype(np.uint8)
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

        raw_video_crop_list = [
                Resize(224),
                Resize((224,224)),
                ClipToTensor(channel_nb=3),
                Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]
        
        raw_video_prcoess = Compose(raw_video_crop_list)

        crop_frames = raw_video_prcoess(frame_list)
        # (C x T x H x W) --> (T x C x H x W)
        crop_frames = crop_frames.permute(1, 0, 2, 3)
        return crop_frames 

def main(args):
    args = update_existing_config_for_inference(args)
    # global training_saver
    args.device = torch.device(args.device)
    # Setup CUDA, GPU & distributed training
    dist_init(args)
    check_arguments(args)
    set_seed(args.seed, args.num_gpus)
    fp16_trainning = None
    logger.info(
        "device: {}, n_gpu: {}, rank: {}, "
        "16-bits training: {}".format(
            args.device, args.num_gpus, get_rank(), fp16_trainning))

    if not is_main_process():
        logger.disabled = True

    logger.info(f"Pytorch version is: {torch.__version__}")
    logger.info(f"Cuda version is: {torch.version.cuda}")
    logger.info(f"cuDNN version is : {torch.backends.cudnn.version()}" )

     # Get Video Swin model 
    swin_model = get_swin_model(args)
    # Get BERT and tokenizer 
    bert_model, config, tokenizer = get_bert_model(args)
    # build SwinBERT based on training configs
    if args.multitask:
        vl_transformer = MultitaskVideoTransformer(args, config, swin_model, bert_model)
    elif args.pred_bbox:
        # vl_transformer = DRAMAVideoTransformer(args, config, swin_model, bert_model)
        vl_transformer = DRAMADetr(args)
    elif args.pred_bbox_with_clip_caption:
        if args.use_trans:
            vl_transformer = DRAMAVideoTransformerTRGTCAPTION(args, config, swin_model, bert_model)
        else:
            vl_transformer = DRAMAVideoTransformerGTCAPTION(args, config, swin_model, bert_model)
    elif args.inpainted_pred_bbox:
        vl_transformer = INPAINTDRAMAVideoTransformer(args, config, swin_model, bert_model)
    else:
        vl_transformer = VideoTransformer(args, config, swin_model, bert_model)
    vl_transformer.freeze_backbone(freeze=args.freeze_backbone)

    total_parameters = sum([param.nelement() for param in vl_transformer.parameters()])
    print("Number of parameter: %.2fM" % (total_parameters/1e6))

    # load weights for inference
    logger.info(f"Loading state dict from checkpoint {args.resume_checkpoint}")
    cpu_device = torch.device('cpu')
    pretrained_model = torch.load(args.resume_checkpoint, map_location=cpu_device)

    if isinstance(pretrained_model, dict):
        vl_transformer.load_state_dict(pretrained_model, strict=False)
    else:
        vl_transformer.load_state_dict(pretrained_model.state_dict(), strict=False)

    vl_transformer.to(args.device)
    vl_transformer.eval()

    raw_video_path = "datasets"
    root_path =  "anomaly_dataset_gt/anomaly_dataset_gt_test.json"

    inpainted_output = dict()

    for mode in ['test']:
        with open(root_path, "r") as f:
            json_data = json.load(f)
        filenames = [i for i in json_data.keys()]
        filenames.sort()
        for idx in tqdm(range(0, len(filenames))):
            filename = filenames[idx]
            
            if filename != "anomaly_dataset/v1/01/seq04/testset/5/mask_v/255.png" and filename != "anomaly_dataset/v1/01/seq01/testset/3/mask_v/243.png":
                continue

            if True:
                # gt_bbox     = torch.tensor([0.,0.,0.,0.])
                gt_bbox     = torch.tensor(json_data[filename]['x1y1x2y2_bbox'])
                # Vehicle, Pedestrian, Infrastructure, Cyclist
                # class_label = torch.tensor(0)
                class_label = torch.tensor(json_data[filename]['have_anomaly'])

                imgs_path = [json_data[filename]['image_path']]
                assert (not imgs_path[-1].startswith('/')) and (imgs_path[-1].endswith('.png') or imgs_path[-1].endswith('.jpg') or imgs_path[-1].endswith('.jpeg'))
                imgs_path = [op.join(raw_video_path, i) for i in imgs_path]

                if len(imgs_path) == 1:
                    raw_frames = np.array([cv2.resize(cv2.imread(imgs_path[-1]),(1024, 1024))[:,:,::-1], cv2.resize(cv2.imread(imgs_path[-1]),(1024, 1024))[:,:,::-1]])
                elif len(imgs_path) > 1:
                    raw_frames = np.array([cv2.resize(cv2.imread(imgs_path[-1]),(1024, 1024))[:,:,::-1], cv2.resize(cv2.imread(imgs_path[-1]),(1024, 1024))[:,:,::-1]])
                else:
                    raise Exception(f"the frame number in a video {imgs_path} is less than 1")

                # (T H W C) to (T C H W)
                raw_frames = (np.transpose(raw_frames, (0, 3, 1, 2)))

            # apply augmentation. frozen-in-time if the input is an image
            # preproc_frames: (T, C, H, W), C = 3, H = W = self.img_res, channel is RGB   
            preproc_raw_frames = apply_augmentations(raw_frames)

            grayscale_cam, outputs = inference(args, None, preproc_raw_frames.unsqueeze(0), vl_transformer, gt_bbox.unsqueeze(0), class_label.unsqueeze(0))

            all_bboxs = outputs["bbox_logits"][0]
            all_bbox_ious = outputs["bbox_iou"][0]
            all_classes = outputs["class_preds"][0]

            save_path = "grad_cam_fig4_t0_end"
            os.makedirs(save_path, exist_ok=True)


            frames = np.transpose(raw_frames, (0, 2, 3, 1))
            h, w = frames.shape[1], frames.shape[2]
            for i in range(len(grayscale_cam)):
                cam_img = grayscale_cam[i]
                cam_array = cam_img.copy()
                os.makedirs(os.path.dirname(os.path.join(save_path, filename)), exist_ok=True)
                # np.save(os.path.join(save_path, filename), cam_array)
                cam_img = cv2.resize(cam_img, (h, w))
                frame_id = 2*i+1
                raw_img = frames[frame_id][:, :, ::-1] / 255

                cam_image = show_cam_on_image(raw_img, cam_img)

                # cv2.rectangle(cam_image, (int(gt_bbox[0]*w), int(gt_bbox[1]*h)), (int(gt_bbox[2]*w), int(gt_bbox[3]*h)), (0, 0, 255), 4)
                # cv2.putText(cam_image, str(class_label.item()), (int(gt_bbox[0]*w), int(gt_bbox[1]*h)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # cv2.rectangle(cam_image, (int(all_bboxs[0]*w), int(all_bboxs[1]*h)), (int(all_bboxs[2]*w), int(all_bboxs[3]*h)), (0, 0, 0), 4)
                # cv2.putText(cam_image, str(all_classes.item()), (int(all_bboxs[0]*w), int(all_bboxs[1]*h)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                # cv2.putText(cam_image, str(round(all_bbox_ious.item(), 4)), (int(all_bboxs[0]*w)+20, int(all_bboxs[1]*h)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                os.makedirs(os.path.dirname(os.path.join(save_path, filename)), exist_ok=True)
                cv2.imwrite(os.path.join(save_path, filename), cam_image)

                cv2.rectangle(cam_image, (int(gt_bbox[0]*w), int(gt_bbox[1]*h)), (int(gt_bbox[2]*w), int(gt_bbox[3]*h)), (0, 0, 255), 4)
                cv2.rectangle(cam_image, (int(all_bboxs[0]*w), int(all_bboxs[1]*h)), (int(all_bboxs[2]*w), int(all_bboxs[3]*h)), (0, 0, 0), 4)
                os.makedirs(os.path.dirname(os.path.join(save_path+"_with_bbox", filename)), exist_ok=True)
                cv2.imwrite(os.path.join(save_path+"_with_bbox", filename), cam_image)
            
            # if idx > 1000:
            #     break

if __name__ == "__main__":
    shared_configs.shared_video_captioning_config(cbs=True, scst=True)
    args = get_custom_args(shared_configs)
    main(args)
