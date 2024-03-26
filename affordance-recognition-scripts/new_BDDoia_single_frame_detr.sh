CUDA_VISIBLE_DEVICES=4,5,6,7 \
OMPI_COMM_WORLD_SIZE="4" \
NCCL_P2P_DISABLE=1 \
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_port=$RANDOM src/tasks/run_bddoia_single_detr.py \
--config src/configs/VidSwinBert/BDDoia.json \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 100 \
--learning_rate 0.0003 \
--max_num_frames 2 \
--pretrained_2d 0 \
--backbone_coef_lr 0.10 \
--mask_prob 0.5 \
--max_masked_token 45 \
--zero_opt_stage 1 \
--mixed_precision_method deepspeed \
--deepspeed_fp16 \
--gradient_accumulation_steps 4 \
--bddoia \
--loss_sparse_w 0.1 \
--loss_sensor_w 0.05 \
--num_workers 20 \
--model_size "base" \
--load_detr ./pretrained/detr/detr-r101-2c7b67e5.pth \
--output_dir ./oia_output/base_pretrained
# --load_detr ./pretrained/detr/detr-r101-2c7b67e5.pth
# --detr_aug \
# NCCL_P2P_DISABLE=1 \