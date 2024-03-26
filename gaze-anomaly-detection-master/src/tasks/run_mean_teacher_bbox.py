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
from src.datasets.vl_dataloader import make_data_loader
from src.evalcap.utils_caption_evaluate import evaluate_on_coco_caption, two_cap_evaluate_on_coco_caption, bbox_eval, level_bbox_eval
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
from src.utils import ramps
from src.modeling.video_captioning_e2e_vid_swin_bert import VideoTransformer
from src.modeling.multitask_e2e_vid_swin_bert import MultitaskVideoTransformer
from src.modeling.drama_bbox import DRAMAVideoTransformer
from src.modeling.drama_inpaint_bbox import INPAINTDRAMAVideoTransformer
from src.modeling.drama_mean_teacher_bbox import DRAMAMEANTEACHERVideoTransformer
from src.modeling.load_swin import get_swin_model, reload_pretrained_swin
from src.modeling.load_bert import get_bert_model
from src.solver import AdamW, WarmupLinearLR

from src.modeling.load_bbox_pred_head import get_iou, get_detect_loss

from azureml.core.run import Run
aml_run = Run.get_context()

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data # argmax
    return logits == labels

def mixed_precision_init(args, model):
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

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 2.0 * ramps.sigmoid_rampup(epoch, 50)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, 1 - alpha)


def train(args, train_dataloader, unlabeled_dataloader, val_dataloader, model, ema_model, tokenizer, training_saver, optimizer, scheduler):
    meters = MetricLogger(delimiter='  ')
    max_iter = args.max_iter
    max_global_step = args.max_global_step
    global_iters_per_epoch = args.global_iters_per_epoch

    start_training_time = time.time()
    end = time.time()
    log_start = time.time()
    # running_loss = RunningMeter('train_loss')
    # running_batch_acc = RunningMeter('train_batch_acc')

    if args.restore_ratio > 0:
        restorer = TrainingRestorer(args, model, optimizer)
        global_step = restorer.global_step
    else:
        global_step = 0

    TB_LOGGER.global_step = global_step
    if not is_main_process() or args.restore_ratio <= 0:
        restorer = NoOp()

    training_saver.save_args(args)
    training_saver.save_tokenizer(tokenizer)

    unlabeled_dataloader_iterator = iter(unlabeled_dataloader)

    for iteration, (img_keys, batch, meta_data) in enumerate(train_dataloader):
        epoch = global_step // global_iters_per_epoch
        data_time = time.time() - end

        try:
            batch_data_unlabeled = next(unlabeled_dataloader_iterator)
        except StopIteration:
            unlabeled_dataloader_iterator = iter(unlabeled_dataloader)
            batch_data_unlabeled = next(unlabeled_dataloader_iterator)

        batch_data_unlabeled = batch_data_unlabeled.to(args.device)
        batch = tuple(t.to(args.device) for t in batch)

        # merge labeled frames and unlabeled frames
        new_image_frames = torch.cat((batch[3], batch_data_unlabeled), dim=0).to(args.device)
        B, B_labeled, B_unlabeled = len(new_image_frames), len(batch[3]), len(batch_data_unlabeled)


        model.train()
        ema_model.train()

        # img_feats (B, #F, C, W, H)
        inputs = {
            'input_ids': batch[0], 'attention_mask': batch[1],
            'token_type_ids': batch[2], 'img_feats': new_image_frames,
            'masked_pos': batch[4], 'masked_ids': batch[5],
            'bbox': batch[6],
        }

        ema_inputs = {
            'input_ids': batch[0], 'attention_mask': batch[1],
            'token_type_ids': batch[2], 'img_feats': new_image_frames,
            'masked_pos': batch[4], 'masked_ids': batch[5],
            'bbox': batch[6],
        }

        if iteration == 1:
            for k, v in inputs.items():
                logger.info(f'{k} = {v.shape}')

        if args.deepspeed_fp16:
            # deepspeed does not autocast inputs
            inputs = fp32_to_fp16(inputs)

        if args.mixed_precision_method == "fairscale":
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(**inputs)
                ema_outputs = ema_model(**ema_inputs)
        else:
            outputs = model(**inputs)
            ema_outputs = ema_model(**ema_inputs)

        loss, logits = outputs[:2]

        if args.mean_teacher_pred_bbox:
            consistency_weight = get_current_consistency_weight(epoch)
            preds_bbox = outputs[-2]
            labeled_preds_bbox = outputs[-2][:B_labeled]
            unlabeled_preds_bbox = outputs[-2][B_labeled:]
            ema_preds_bbox = ema_outputs[-2]
            loss_labeled = get_detect_loss(labeled_preds_bbox, batch[6])
            loss_unlabeled = get_detect_loss(preds_bbox, ema_preds_bbox)
            loss += loss_labeled + loss_unlabeled*consistency_weight

        loss_dict = {'loss': loss}


        if args.mean_teacher_pred_bbox:
            loss_dict['loss_labeled'] = loss_labeled
            loss_dict['loss_unlabeled'] = loss_unlabeled
            loss_dict['loss_consistency'] = loss_unlabeled*consistency_weight

        meters.update(**loss_dict)

        # backward pass
        backward_now = iteration % args.gradient_accumulation_steps == 0
        if args.mixed_precision_method == "deepspeed":
            model.backward(loss)
        elif args.mixed_precision_method == "fairscale":
            scaler.scale(loss).backward()
        else:
            # apex
            with amp.scale_loss(loss, optimizer, delay_unscale=not backward_now) as scaled_loss:
                scaled_loss.backward()
        if backward_now:
            global_step += 1
            if args.mean_teacher_pred_bbox:
                TB_LOGGER.add_scalar('train/loss_labeled', loss_labeled.cpu(), global_step)
                TB_LOGGER.add_scalar('train/loss_unlabeled', loss_unlabeled.cpu(), global_step)
                TB_LOGGER.add_scalar('train/loss_consistency', (loss_unlabeled*consistency_weight).cpu(), global_step)
                TB_LOGGER.add_scalar('train/consistency_weight', consistency_weight, global_step)

            lr_VisBone = optimizer.param_groups[0]["lr"]
            lr_LM = optimizer.param_groups[1]["lr"]

            TB_LOGGER.add_scalar(
                "train/lr_lm", lr_LM, global_step)
            TB_LOGGER.add_scalar(
                "train/ls_visBone", lr_VisBone, global_step)
            
            if args.max_grad_norm != -1:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.max_grad_norm)
                TB_LOGGER.add_scalar("train/grad_norm", grad_norm, global_step)
            TB_LOGGER.step()
            if args.mixed_precision_method == "deepspeed":
                model.step()
            elif args.mixed_precision_method == "fairscale":
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
            else:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            update_ema_variables(model, ema_model, 0.999, global_step)
            restorer.step()

        batch_time = time.time() - end
        
        if backward_now:
            if global_step % args.logging_steps == 0 or global_step == max_global_step:
                if 'time_info' in meters.meters:
                    avg_time = meters.meters['time_info']['compute'].global_avg
                    eta_seconds = avg_time * (max_iter - iteration)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                else:
                    eta_string = 'Unknown'
                eta_seconds = batch_time * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                speed = args.num_gpus * args.logging_steps * len(batch[0]) / (time.time() - log_start)
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                logger.info(
                    meters.delimiter.join(
                        [
                            f"eta: {eta_string}",
                            f"iter: {iteration}",
                            f"global_step: {global_step}",
                            f'speed: {speed:.1f} images/sec',
                            f"{meters}",
                            f"lr (Visual Encoder): {lr_VisBone:.2e}",
                            f"lr (LM): {lr_LM:.2e}",
                            f"max mem: {memory:.0f}",
                        ]
                    )
                )
                TB_LOGGER.add_scalar("train/speed", speed, global_step)
                TB_LOGGER.add_scalar("train/memory", memory, global_step)
                TB_LOGGER.add_scalar("train/batch_time", batch_time, global_step)
                TB_LOGGER.add_scalar("train/data_time", data_time, global_step)
                log_start = time.time()


            if (args.save_steps > 0 and global_step % args.save_steps == 0) or global_step == max_global_step or global_step == 1:
                
                checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
                    epoch, global_step))
                if get_world_size() > 1:
                    dist.barrier()

                if get_world_size() > 1:
                    dist.barrier()    
                if args.evaluate_during_training:
                    logger.info(f"Perform evaluation at iteration {iteration}, global_step {global_step}")
                    evaluate_file = evaluate(args, val_dataloader, model, tokenizer, checkpoint_dir)
                    if get_world_size() > 1:
                        dist.barrier()
                    if is_main_process():
                        if args.mean_teacher_pred_bbox:
                            with open(evaluate_file, 'r') as f:
                                tmp_data = json.load(f)
                                try:
                                    raw_eval_iou = tmp_data['bbox_mean_iou']
                                except:
                                    print(tmp_data)
                                    raise Exception("json read error")
                            val_log = {f'valid/bbox_mean_iou': raw_eval_iou}
                            TB_LOGGER.log_scalar_dict(val_log)
                            aml_run.log(name='raw_bbox_mean_iou', value=float(raw_eval_iou))
                            if float(raw_eval_iou) > 0.6:
                                training_saver.save_model(
                                    checkpoint_dir, global_step, model, optimizer)
                    if get_world_size() > 1:
                        dist.barrier()                

        if iteration > 2:
            meters.update(
                batch_time=batch_time,
                data_time=data_time,
            )
        end = time.time()

        if global_step >= max_global_step and (max_iter - iteration):
            logger.info(f'Missing {max_iter - iteration} iterations, early break')
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(f'Total training time: {total_time_str} ({(total_training_time / max_iter):.4f} s / iter)')
    return checkpoint_dir

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

def get_bbox_evaluate_file(predict_file):
    assert predict_file.endswith('.tsv')
    return op.splitext(predict_file)[0] + '.bbox_eval.json'

def evaluate(args, val_dataloader, model, tokenizer, output_dir):
    predict_file = get_predict_file(output_dir, args)
    test(args, val_dataloader, model, tokenizer, predict_file)

    if get_world_size() > 1:
        dist.barrier()
    evaluate_file = get_evaluate_file(predict_file)
    bbox_eval_file = get_bbox_evaluate_file(predict_file)
    if is_main_process():
        # caption_file = op.join(args.data_dir, val_dataloader.dataset.mode+".caption_coco_format.json")
        # if args.use_sep_cap:
        #     result = two_cap_evaluate_on_coco_caption(predict_file, caption_file, outfile=evaluate_file)
        # else:
        #     result = evaluate_on_coco_caption(predict_file, caption_file, outfile=evaluate_file)
        
        iou_result = 0.
        if args.mean_teacher_pred_bbox:
            iou_result = bbox_eval(predict_file, outfile=bbox_eval_file)
        else:
            raise Exception("Args for pred_bbox should have one")
        
        # logger.info(f'evaluation result: {str(result)}')
        logger.info(f'iou evaluation result: {str(iou_result)}')
        logger.info(f'evaluation result saved to {evaluate_file}')
    if get_world_size() > 1:
        dist.barrier()
    return bbox_eval_file

def test(args, test_dataloader, model, tokenizer, predict_file):

    world_size = get_world_size()
    if world_size == 1:
        cache_file = predict_file
    else:
        # local_rank would not work for cross-node distributed training
        cache_file = op.splitext(predict_file)[0] + '_{}_{}'.format(get_rank(),
                world_size) + op.splitext(predict_file)[1]

    model.eval()
    def gen_rows():
        time_meter = 0
        # restore existing results for long running inference tasks
        exist_key2pred = {}
        tmp_file = cache_file + '.tmp.copy'
        if op.isfile(tmp_file):
            with open(tmp_file, 'r') as fp:
                for line in fp:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        exist_key2pred[parts[0]] = parts[1]

        with torch.no_grad():
            for step, (img_keys, batch,meta_data) in tqdm(enumerate(test_dataloader)):
                # torch.cuda.empty_cache()
                # is_exist = True
                # for k in img_keys:
                #     if k not in exist_key2pred:
                #         is_exist = False
                #         break
                # if is_exist:
                #     for k in img_keys:
                #         yield k, exist_key2pred[k]
                #         # return k, exist_key2pred[k]
                #     continue
                batch = tuple(t.to(args.device) for t in batch)

                inputs = {'is_decode': True,
                    'input_ids': batch[0], 'attention_mask': batch[1],
                    'token_type_ids': batch[2], 'img_feats': batch[3],
                    'masked_pos': batch[4],
                }
                gt_bboxes = batch[5]

                tic = time.time()
                # captions, logprobs
                
                if args.deepspeed_fp16:
                    # deepspeed does not auto cast inputs.
                    inputs = fp32_to_fp16(inputs)

                if args.mixed_precision_method == "fairscale":
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                time_meter += time.time() - tic
                # all_caps = outputs[0]  # batch_size * num_keep_best * max_len
                # all_confs = torch.exp(outputs[1])
                all_bboxs = outputs[-2]
                with torch.no_grad():
                    all_bbox_ious = get_iou(all_bboxs.detach(), gt_bboxes)

                for img_key, box, box_iou in zip(img_keys, all_bboxs, all_bbox_ious):
                    res = []
                    res.append({'box': str(box.tolist()), 'box_iou': str(box_iou.tolist())})
                    if isinstance(img_key, torch.Tensor):
                        img_key = img_key.item()
                    yield img_key, json.dumps(res)
                    # return img_key, json.dumps(res)

        logger.info(f"Inference model computing time: {(time_meter / (step+1))} seconds per batch")

    # a = gen_rows()
    if args.use_sep_cap:
        double_tsv_writer(gen_rows(), cache_file)
    else:
        tsv_writer(gen_rows(), cache_file)
    if world_size > 1:
        dist.barrier()
    if world_size > 1 and is_main_process():
        cache_files = [op.splitext(predict_file)[0] + '_{}_{}'.format(i, world_size) + \
            op.splitext(predict_file)[1] for i in range(world_size)]
        concat_tsv_files(cache_files, predict_file)
        delete_tsv_files(cache_files)
        reorder_tsv_keys(predict_file, test_dataloader.dataset.image_keys, predict_file)
    if world_size > 1:
        dist.barrier()

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
    parser.add_argument('--match_weight', type=float, default='0.5')
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

    def create_model(args, ema=False):
        # Get Video Swin model 
        swin_model = get_swin_model(args)
        # Get BERT and tokenizer 
        bert_model, config, tokenizer = get_bert_model(args)
        # build SwinBERT based on training configs
        if args.mean_teacher_pred_bbox:
            model = DRAMAMEANTEACHERVideoTransformer(args, config, swin_model, bert_model)
        else:
            raise Exception("Now only support risk localization")
        model.freeze_backbone(freeze=args.freeze_backbone)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model, tokenizer

    vl_transformer, tokenizer = create_model(args)
    ema_vl_transformer, _ = create_model(args, ema=True)

    if args.do_eval:
        # load weights for eval/inference
        logger.info(f"Loading state dict from checkpoint {args.resume_checkpoint}")
        cpu_device = torch.device('cpu')
        pretrained_model = torch.load(args.resume_checkpoint, map_location=cpu_device)

        if isinstance(pretrained_model, dict):
            vl_transformer.load_state_dict(pretrained_model, strict=False)
            ema_vl_transformer.load_state_dict(pretrained_model, strict=False)
        else:
            vl_transformer.load_state_dict(pretrained_model.state_dict(), strict=False)
            ema_vl_transformer.load_state_dict(pretrained_model.state_dict(), strict=False)

    elif args.do_train and args.pretrained_checkpoint != '':
        ckpt_path = args.pretrained_checkpoint+'model.bin'
        assert op.exists(ckpt_path), f"{ckpt_path} does not exist"
        logger.info(f"Loading state dict from checkpoint {ckpt_path}")
        cpu_device = torch.device('cpu')
        pretrained_model = torch.load(ckpt_path, map_location=cpu_device)

        if isinstance(pretrained_model, dict):
            vl_transformer.load_state_dict(pretrained_model, strict=False)
            ema_vl_transformer.load_state_dict(pretrained_model, strict=False)
        else:
            vl_transformer.load_state_dict(pretrained_model.state_dict(), strict=False)
            ema_vl_transformer.load_state_dict(pretrained_model.state_dict(), strict=False)

        del pretrained_model
        gc.collect()
        torch.cuda.empty_cache()

        # args.eval_model_dir = args.pretrained_checkpoint
        # checkpoint = args.eval_model_dir
        # assert op.isdir(checkpoint)
        # vl_transformer.max_img_seq_length = int(args.max_img_seq_length)
        # vl_transformer.config.num_visual_tokens = int(args.max_img_seq_length)
        # args.model_name_or_path = args.pretrained_checkpoint
        # if args.reload_pretrained_swin:
        #     vl_transformer.swin = reload_pretrained_swin(vl_transformer.swin, args)

    vl_transformer.to(args.device)
    ema_vl_transformer.to(args.device)
    
    if args.do_train:
        args = restore_training_settings(args)
        train_dataloader = make_data_loader(args, tokenizer, args.distributed, mode='train', is_train=True)
        unlabeled_dataloader = make_data_loader(args, tokenizer, args.distributed, mode='unlabeled', is_train=True)

        val_dataloader = make_data_loader(args, tokenizer, args.distributed, mode='test', is_train=False)
        #val_dataloader = make_data_loader(args, tokenizer, args.distributed, mode='train', is_train=False)
        args.max_iter = len(train_dataloader)
        args.max_global_step =  args.max_iter// args.gradient_accumulation_steps
        args.global_iters_per_epoch = args.max_global_step // args.num_train_epochs
        args.save_steps = args.global_iters_per_epoch
        # args.save_steps = 10

        args, vl_transformer, optimizer, scheduler = mixed_precision_init(args, vl_transformer)
        train(args, train_dataloader, unlabeled_dataloader, val_dataloader, vl_transformer, ema_vl_transformer, tokenizer, training_saver, optimizer, scheduler)

    elif args.do_eval:
        val_dataloader = make_data_loader(args, tokenizer, args.distributed, mode='test', is_train=False)
        args, vl_transformer, _, _ = mixed_precision_init(args, vl_transformer)
        evaluate_file = evaluate(args, val_dataloader, vl_transformer, tokenizer, args.eval_model_dir)
    
    if args.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    shared_configs.shared_video_captioning_config(cbs=True, scst=True)
    args = get_custom_args(shared_configs)
    main(args)
