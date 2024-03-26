CUDA_VISIBLE_DEVICES=2 \
OMPI_COMM_WORLD_SIZE="1" \
python src/tasks/run_bddoia_single_gaze_dotinput_gazeonly.py \
--config src/configs/VidSwinBert/BDDoia.json \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 300 \
--learning_rate 0.001 \
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
--num_workers 10 \
--gaze_pred \
--output_dir ./output/test0316_lr001_bklr_10_bs32_gaze_gazeonly
# --detr_aug \
