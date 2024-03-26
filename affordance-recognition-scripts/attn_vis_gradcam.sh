CUDA_VISIBLE_DEVICES=6 \
OMPI_COMM_WORLD_SIZE="1" \
python -m pdb src/tasks/attention_visualize_gradcam.py \
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
--eval_model_dir output/test0405_lr0005_bklr_10_bs32times6_selfattn_4token/checkpoint-300-24900/ \
--output_dir ./output/test

# --detr_aug \

