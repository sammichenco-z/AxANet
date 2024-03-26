CUDA_VISIBLE_DEVICES=6 \
OMPI_COMM_WORLD_SIZE="1" \
python -m pdb src/tasks/attention_visualize_raw_human_machine.py \
--config src/configs/VidSwinBert/BDDoia.json \
--per_gpu_train_batch_size 1 \
--per_gpu_eval_batch_size 1 \
--num_train_epochs 300 \
--learning_rate 0.0003 \
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
--do_eval \
--visualize_attn \
--eval_model_dir ./output_temp/new_driver/checkpoint-1-667/ \
--output_dir ./output/test \
--human_choose new \
--time 9_18_0.0

# --detr_aug \

# --eval_model_dir ./output_temp/test0814_detr_4query_res101_detrpretrain/checkpoint-100-8300/ \

