from __future__ import absolute_import, division, print_function

import os
import sys
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)
import os.path as op
import json
import time
import datetime
import torch
import torch.distributed as dist
import gc
import deepspeed
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from src.configs.config import (basic_check_arguments, shared_configs, restore_training_settings)
from src.datasets.bddoia_s_dataloader import make_data_loader
from src.evalcap.utils_caption_evaluate import class_eval
from src.utils.logger import LOGGER as logger
from src.utils.logger import (TB_LOGGER, RunningMeter, add_log_to_file)
from src.utils.load_save import TrainingRestorer, TrainingSaver
from src.utils.comm import (is_main_process,
                            get_rank, get_world_size, dist_init)
from src.utils.miscellaneous import (NoOp, mkdir, set_seed, str_to_bool,
                                    delete_tsv_files, concat_tsv_files)
from src.utils.metric_logger import MetricLogger
from src.utils.tsv_file_ops import tsv_writer, double_tsv_writer, reorder_tsv_keys
from src.utils.deepspeed import get_deepspeed_config, fp32_to_fp16
# from src.modeling.bddoia_classification import BDDOIAVideoTransformer
# from src.modeling.bddoia_classification_selfattn import BDDOIAVideoTransformer
from src.modeling.bddoia_classification_selfattn_alldetr import BDDOIAVideoTransformer
# from src.modeling.bddoia_classification_bert import BDDOIAVideoTransformer

from src.modeling.load_swin import get_swin_model, reload_pretrained_swin
from src.modeling.load_bert import get_bert_model
from src.solver import AdamW, WarmupLinearLR
from IPython import embed
from azureml.core.run import Run
import numpy as np
import cv2
from src.rollout.vit_rollout import VITAttentionRollout, rollout
from src.pytorch_grad_cam.grad_cam import GradCAM
from src.pytorch_grad_cam.utils.image import (preprocess_image,
                                              show_cam_on_image)

from src.datasets.data_utils.video_transforms import Compose, Resize, RandomCrop, ColorJitter, Normalize, CenterCrop, RandomHorizontalFlip, RandomResizedCrop
from PIL import Image
from src.datasets.data_utils.volume_transforms import ClipToTensor

aml_run = Run.get_context()

def compute_score_with_logits(logits, labels):#compute the true label from logits
    logits = torch.max(logits, -1)[1].data # argmax
    return logits == labels

def mixed_precision_init(args, model):#最大精度
    max_iter = args.max_iter
    max_global_step = args.max_global_step
    global_iters_per_epoch = args.global_iters_per_epoch

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_swin_param_tp = [(n, p) for n, p in decay_param_tp if "swin." in n]
    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "swin." not in n]

    no_decay_swin_param_tp = [(n, p) for n, p in no_decay_param_tp if "swin." in n]
    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "swin." not in n]

    weight_decay = 0.2
    coef_lr = args.backbone_coef_lr 
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_swin_param_tp], 
            'weight_decay': weight_decay,
            'lr': args.learning_rate * coef_lr},
        {'params': [p for n, p in decay_bert_param_tp], 
            'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_swin_param_tp],
            'weight_decay': 0.0,
            'lr': args.learning_rate * coef_lr},
        {'params': [p for n, p in no_decay_bert_param_tp], 
            'weight_decay': 0.0}
    ]
    
    if args.mixed_precision_method == "fairscale":
        from fairscale.optim.oss import OSS
        optimizer = OSS(
            params=optimizer_grouped_parameters, optim=AdamW, lr=args.learning_rate,
            eps=args.adam_epsilon)
    else:
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate,
            eps=args.adam_epsilon)
    if args.scheduler == "warmup_linear":
        scheduler = WarmupLinearLR(
            optimizer, max_global_step, warmup_ratio=args.warmup_ratio)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(max_iter/2.0), gamma=0.1)
    
    if args.mixed_precision_method == "deepspeed":
        config = get_deepspeed_config(args)
        model, optimizer, _, _ = deepspeed.initialize(
            config_params=config,
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler)
    elif args.mixed_precision_method == "fairscale":
        from fairscale.optim.grad_scaler import ShardedGradScaler
        scaler = ShardedGradScaler()
        # this is equivalent to deepspeed zero_opt_stage = 2
        from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
        model = ShardedDDP(
            model, optimizer,
            reduce_buffer_size= 0 if args.fairscale_fp16 else 2 ** 23, # 2 ** 23 is the default value
            reduce_fp16=args.fairscale_fp16)
    else:
        # opt_level is O0, Apex will run as fp32
        model, optimizer = amp.initialize(
            model, optimizer,
            enabled=True,
            opt_level=f'O{args.amp_opt_level}')
        if args.distributed: #
            model = DDP(model)
    return args, model, optimizer, scheduler

def get_predict_file(output_dir, args, data_yaml_file=None):
    cc = ['pred']
    data_yaml_file = 'datasets/coco_caption/test.yaml'
    # example data_yaml_file: datasets/coco_caption/test.yaml
    data = 'coco_caption'
    cc.append(op.splitext(op.basename(data_yaml_file))[0])
    cc.append('beam{}'.format(args.num_beams))
    cc.append('max{}'.format(args.max_gen_length))
    if args.num_keep_best != 1:
        cc.append('best{}'.format(args.num_keep_best))
    if args.output_hidden_states:
        cc.append('hidden')
    return op.join(output_dir, '{}.tsv'.format('.'.join(cc)))

def get_evaluate_file(predict_file):
    assert predict_file.endswith('.tsv')
    return op.splitext(predict_file)[0] + '.eval.json'

def get_class_evaluate_file(predict_file):
    assert predict_file.endswith('.tsv')
    return op.splitext(predict_file)[0] + '.class_eval.json'

def evaluate(args, val_dataloader, model, tokenizer, output_dir,global_step):
    predict_file = get_predict_file(output_dir, args)
    test(args, model, tokenizer, predict_file)

    if get_world_size() > 1:
        dist.barrier()
    evaluate_file = get_class_evaluate_file(predict_file)
    if is_main_process():
        # caption_file = op.join(args.data_dir, val_dataloader.dataset.mode+".caption_coco_format.json")
        # if args.use_sep_cap:
        #     result = two_cap_evaluate_on_coco_caption(predict_file, caption_file, outfile=evaluate_file)
        # else:
        #     result = evaluate_on_coco_caption(predict_file, caption_file, outfile=evaluate_file)
        
        class_eval_file = get_class_evaluate_file(predict_file)
        f1_total_all = 0.
        # f1_reason_all,f1_action_all,f1_total_all,f1_f,f1_s,f1_l,f1_r = class_eval(predict_file, outfile=class_eval_file)
        f1_reason_all,f1_action_all,f1_total_all,f1_f,f1_s,f1_l,f1_r, mf1_action, mf1_reason = class_eval(predict_file, outfile=class_eval_file)
        TB_LOGGER.add_scalar("f1_reason_all", f1_reason_all, global_step)
        TB_LOGGER.add_scalar("f1_action_all", f1_action_all, global_step)
        TB_LOGGER.add_scalar("f1_all", f1_total_all, global_step)
        TB_LOGGER.add_scalar("f1_f",f1_f, global_step)
        TB_LOGGER.add_scalar("f1_s", f1_s, global_step)
        TB_LOGGER.add_scalar("f1_l", f1_l, global_step)
        TB_LOGGER.add_scalar("f1_r", f1_r, global_step)
        TB_LOGGER.add_scalar("mf1_action", mf1_action, global_step)
        TB_LOGGER.add_scalar("mf1_reason", mf1_reason, global_step)
        #logger.info(f'evaluation result: {str(result)}')
        logger.info(f'f1 evaluation result: {str(f1_total_all)}')
        logger.info(f'evaluation result saved to {evaluate_file}')
    if get_world_size() > 1:
        dist.barrier()
    return evaluate_file

def reshape_transform(tensor, frame=1, height=7, width=7):
    result = tensor.reshape(1, frame,
                            height, width, tensor.size(-1))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.permute(0, 4, 1, 2, 3)
    return result

def test(args, model, tokenizer, predict_file):


    world_size = get_world_size()
    if world_size == 1:
        cache_file = predict_file
    else:
        # local_rank would not work for cross-node distributed training
        cache_file = op.splitext(predict_file)[0] + '_{}_{}'.format(get_rank(),
                world_size) + op.splitext(predict_file)[1]

    attn_weight_path = op.join(cache_file.split('/')[0], cache_file.split('/')[1], cache_file.split('/')[2], 'attn_weight')
    if not op.exists(attn_weight_path):
        mkdir(attn_weight_path)

    model.eval()
    if True:
        time_meter = 0
        # # restore existing results for long running inference tasks
        # exist_key2pred = {}
        # tmp_file = cache_file + '.tmp.copy'
        # if op.isfile(tmp_file):
        #     with open(tmp_file, 'r') as fp:
        #         for line in fp:
        #             parts = line.strip().split('\t')
        #             if len(parts) == 2:
        #                 exist_key2pred[parts[0]] = parts[1]

        # grad_rollout = VITAttentionRollout(model, discard_ratio=0.9, head_fusion='mean')

        cam = GradCAM(model=model,
                    target_layers=[model.module.detr_encoder.transformer.encoder.layers[-1].norm2],
                    use_cuda=True,
                    reshape_transform=reshape_transform
                    )

        path = "human_machine_raw_v3"
        filenames = sorted(os.listdir(path))

        if True:
            for step, filename in tqdm(enumerate(filenames)):
            
                img = Image.open(op.join(path, filename))
                frame_list = [img, img]

                raw_video_crop_list = [
                    Resize(224),
                    Resize((224, 224)),
                    ClipToTensor(channel_nb=3),
                    Normalize(mean=[0.485,0.456,0.406],std=[0.229, 0.224, 0.225])
                ]            
                raw_video_prcoess = Compose(raw_video_crop_list)

                crop_frames = raw_video_prcoess(frame_list)
                crop_frames = crop_frames.permute(1, 0, 2, 3)

                crop_frames = crop_frames.unsqueeze(0).to(args.device)

                label_action = torch.zeros((1, 4), device=args.device, dtype=torch.int64)
                label_reason = torch.zeros((1, 21), device=args.device, dtype=torch.int64)

                inputs = {'is_decode': True,
                    'img_feats': crop_frames,
                    'label_action': label_action,
                    'label_reason': label_reason,
                }

                tic = time.time()
                # captions, logprobs
                
                if args.deepspeed_fp16:
                    # deepspeed does not auto cast inputs.
                    inputs = fp32_to_fp16(inputs)

                action_id = 2

                grayscale_cam, raw_outputs = cam(input_tensor=inputs,
                                targets=None,
                                eigen_smooth=False,
                                aug_smooth=False,
                                action_id=2)

                time_meter += time.time() - tic

                logits_action, logits_reason, loss_action, loss_reason, attentions = raw_outputs[:]

                # grayscale_cam = rollout(attentions, 0.9, 'mean')

                #logits, loss = outputs[:]

                preds_action = torch.argmax(logits_action, dim=-1)
                preds_reason = torch.argmax(logits_reason, dim=-1)
                #pred = torch.argmax(logits ,dim=-1)

                labels_action = label_action
                
                #for i in range(len(labels_action)):
                #    labels_action[i][0]=1
                labels_reason = label_reason
                #label = labels_action[:,3]
                

                raw_imgs = cv2.imread(op.join(path, filename))
                h, w = raw_imgs.shape[0], raw_imgs.shape[1]


                grayscale_cam = grayscale_cam[0]

                if True:
                    for i in range(grayscale_cam.shape[0]):
                        # if preds_action[0][action_id] == 1:
                        if True:
                            save_path = "grad_cam_human_machine_detr_final_version_driver_no_pretrained_no_finetune"
                            os.makedirs(save_path, exist_ok=True)

                            cam_img = grayscale_cam[i]
                            cam_array = cam_img.copy()
                            # save_path = "grad_cam_human_machine_detr"
                            np.save(f"{save_path}/{filename}", cam_array)

                            cam_img = cv2.resize(cam_img, (w, h))
                            

                            cam_image = show_cam_on_image(raw_imgs / 255, cam_img).copy()

                            cv2.putText(cam_image, str(preds_action[0][action_id].item()) + "  " + str(labels_action[0][action_id].item()), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                            cv2.imwrite(f"{save_path}/{filename}", cam_image)




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
    train_args.visualize_attn = args.visualize_attn
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
    parser.add_argument('--loss_bbox_w', type=float, default=0)
    parser.add_argument('--sparse_mask_soft2hard', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--transfer_method', type=int, default=-1,
                        help="0: load all SwinBERT pre-trained weights, 1: load only pre-trained sparse mask")
    parser.add_argument('--att_mask_expansion', type=int, default=-1,
                        help="-1: random init, 0: random init and then diag-based copy, 1: interpolation")
    parser.add_argument('--resume_checkpoint', type=str, default='None')
    parser.add_argument('--test_video_fname', type=str, default='None')
    args = base_config.parse_args()
    return args

def main(args):
    if args.do_train==False or args.do_eval==True:
        args = update_existing_config_for_inference(args) 

    # global training_saver
    args.device = torch.device(args.device)
    # Setup CUDA, GPU & distributed training
    dist_init(args)
    check_arguments(args)
    mkdir(args.output_dir)
    logger.info(f"creating output_dir at: {args.output_dir}")
    set_seed(args.seed, args.num_gpus)
    
    if args.mixed_precision_method == "apex":
        fp16_trainning = f"apex O{args.amp_opt_level}"
    elif args.mixed_precision_method == "deepspeed":
        amp_info = '' if args.deepspeed_fp16 else f'amp, {args.amp_opt_level}'
        fp16_info = '' if not args.deepspeed_fp16 else f'fp16, {args.zero_opt_stage}'
        fp16_trainning = f"deepspeed, {amp_info}{fp16_info}"
    elif args.mixed_precision_method == "fairscale":
        assert args.distributed, "fairscale can only be used for distributed training"
        fp16_trainning = f"fairscale, fp16: {args.fairscale_fp16}, default zero_opt 2"
    else:
        fp16_trainning = None

    logger.info(
        "device: {}, n_gpu: {}, rank: {}, "
        "16-bits training: {}".format(
            args.device, args.num_gpus, get_rank(), fp16_trainning))

    if not is_main_process():
        logger.disabled = True
        training_saver = NoOp()
    else:
        training_saver = TrainingSaver(args.output_dir)
        TB_LOGGER.create(op.join(args.output_dir, 'log'))
        add_log_to_file(op.join(args.output_dir, 'log', "log.txt"))

    logger.info(f"Pytorch version is: {torch.__version__}")
    logger.info(f"Cuda version is: {torch.version.cuda}")
    logger.info(f"cuDNN version is : {torch.backends.cudnn.version()}" )

    # Get Video Swin model 
    swin_model = get_swin_model(args)
    # Get BERT and tokenizer 
    bert_model, config, tokenizer = get_bert_model(args)
    # build SwinBERT based on training configs
    if args.bddoia:
        vl_transformer = BDDOIAVideoTransformer(args)
    else:
        exit()
    vl_transformer.freeze_backbone(freeze=args.freeze_backbone)

    if args.do_eval:
        # load weights for eval/inference
        logger.info(f"Loading state dict from checkpoint {args.resume_checkpoint}")
        cpu_device = torch.device('cpu')
        pretrained_model = torch.load(args.resume_checkpoint, map_location=cpu_device)

        if isinstance(pretrained_model, dict):
            vl_transformer.load_state_dict(pretrained_model, strict=False)
        else:
            vl_transformer.load_state_dict(pretrained_model.state_dict(), strict=False)

    elif args.do_train and args.pretrained_checkpoint != '':
        ckpt_path = args.pretrained_checkpoint+'model.bin'
        assert op.exists(ckpt_path), f"{ckpt_path} does not exist"
        logger.info(f"Loading state dict from checkpoint {ckpt_path}")
        cpu_device = torch.device('cpu')
        pretrained_model = torch.load(ckpt_path, map_location=cpu_device)

        if args.learn_mask_enabled == False:
            if isinstance(pretrained_model, dict):
                vl_transformer.load_state_dict(pretrained_model, strict=False)
            else:
                vl_transformer.load_state_dict(pretrained_model.state_dict(), strict=False)

        elif args.learn_mask_enabled == True:
            pretrained_mask_shape = pretrained_model['learn_vid_att.weight'].shape
            init_mask_shape = vl_transformer.learn_vid_att.weight.shape

            #-------------------------------------------------------------
            # transfer at the same frame rate
            if pretrained_mask_shape==init_mask_shape: 
                # init using entire pre-trained SwinBERT weights
                if args.transfer_method==0:
                    if isinstance(pretrained_model, dict):
                        vl_transformer.load_state_dict(pretrained_model, strict=False)
                    else:
                        vl_transformer.load_state_dict(pretrained_model.state_dict(), strict=False)
                # init using only pre-trained sparse att mask weights
                else:
                    vl_transformer.reload_attn_mask(pretrained_model['learn_vid_att.weight'])
            #-------------------------------------------------------------
            # transfer across different frame rates
            else:  
                # init using entire pre-trained SwinBERT weights, except sparse attn mask
                if args.transfer_method==0:
                    if isinstance(pretrained_model, dict):
                        new_state_dict={}
                        for k,v in zip(pretrained_model.keys(), pretrained_model.values()):
                            if k!='learn_vid_att.weight' or k=='learn_vid_att.weight' and pretrained_mask_shape==init_mask_shape:
                                new_state_dict={k:v}
                        vl_transformer.load_state_dict(new_state_dict, strict=False)
                        del new_state_dict
                    else:
                        pretrained_model_state_dict = pretrained_model.state_dict()
                        new_state_dict={}
                        for k,v in zip(pretrained_model_state_dict.keys(), pretrained_model_state_dict.values()):
                            if k!='learn_vid_att.weight' or k=='learn_vid_att.weight' and pretrained_mask_shape==init_mask_shape:
                                new_state_dict={k:v}
                        vl_transformer.load_state_dict(new_state_dict, strict=False)
                        del new_state_dict

                # expand pre-trained sparse att mask to the desired size          
                if args.att_mask_expansion==0:
                    vl_transformer.diag_based_init_attn_mask(pretrained_model['learn_vid_att.weight'])
                elif args.att_mask_expansion==1:
                    vl_transformer.bilinear_init_attn_mask(pretrained_model['learn_vid_att.weight'])
                else:
                    vl_transformer.random_init_attn_mask()                                                                          

        del pretrained_model
        gc.collect()
        torch.cuda.empty_cache()

        args.eval_model_dir = args.pretrained_checkpoint
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        vl_transformer.max_img_seq_length = int(args.max_img_seq_length)
        vl_transformer.config.num_visual_tokens = int(args.max_img_seq_length)
        args.model_name_or_path = args.pretrained_checkpoint
        if args.reload_pretrained_swin:
            vl_transformer.swin = reload_pretrained_swin(vl_transformer.swin, args)

    vl_transformer.to(args.device)
    
    if args.do_eval:
        val_dataloader = make_data_loader(args, tokenizer, args.distributed, mode='test', is_train=False)
        args, vl_transformer, _, _ = mixed_precision_init(args, vl_transformer)
        evaluate_file = evaluate(args, val_dataloader, vl_transformer, tokenizer, args.eval_model_dir, global_step=1)
    
    if args.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    shared_configs.shared_video_captioning_config(cbs=True, scst=True)
    args = get_custom_args(shared_configs)
    main(args)
