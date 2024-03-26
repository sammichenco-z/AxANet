CUDA_VISIBLE_DEVICES=0 \
OMPI_COMM_WORLD_SIZE="1" \
python src/tasks/run_bddoia.py \
--config src/configs/VidSwinBert/BDDoia.json \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 16 \
--num_train_epochs 100 \
--learning_rate 0.001 \
--max_num_frames 16 \
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
--num_workers 10 \
--output_dir ./output/test0323_frame_16_lr001_bklr_10_bs8_ep100
