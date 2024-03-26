CUDA_VISIBLE_DEVICES=3,4,5,7 \
OMPI_COMM_WORLD_SIZE="4" \
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_port=45651 src/tasks/run_bddoia_single.py \
--config src/configs/VidSwinBert/BDDoia.json \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 8 \
--num_train_epochs 100 \
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
--output_dir ./output/test0405_lr0005_bklr_10_bs32times6_selfattn_4token_1layer
# --detr_aug \
# NCCL_P2P_DISABLE=1 \