CUDA_VISIBLE_DEVICES=0  \
OMPI_COMM_WORLD_SIZE="1" \
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=45655 src/tasks/run_bddoia_single_gaze.py \
--config src/configs/VidSwinBert/BDDoia.json \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 300 \
--learning_rate 0.0005 \
--max_num_frames 2 \
--pretrained_2d 0 \
--backbone_coef_lr 0.10 \
--mask_prob 0.5 \
--max_masked_token 45 \
--zero_opt_stage 1 \
--mixed_precision_method deepspeed \
--deepspeed_fp16 \
--gradient_accumulation_steps 1 \
--bddoia \
--loss_sparse_w 0.1 \
--loss_sensor_w 0.05 \
--num_workers 20 \
--gaze_pred \
--output_dir ./output/test

# --output_dir ./output/test0413_lr0005_bklr_10_bs32times6_selfattn_gaze_concatinput_4token_divlearn

# --output_dir ./output/test0405_lr0005_bklr_10_bs32times6_selfattn_gaze_concatinput_4token
# --detr_aug \

