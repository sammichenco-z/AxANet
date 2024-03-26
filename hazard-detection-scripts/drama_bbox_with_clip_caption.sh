CUDA_VISIBLE_DEVICES=3,5,6,7 \
OMPI_COMM_WORLD_SIZE="4" \
NCCL_P2P_DISABLE=1 \
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_port=5$[RANDOM%10000] src/tasks/run_bbox_with_clip_caption.py \
        --config src/configs/VidSwinBert/drama_default.json \
        --train_yaml drama/training_32frames.yaml \
        --val_yaml drama/testing_32frames.yaml \
        --per_gpu_train_batch_size 32 \
        --per_gpu_eval_batch_size 16 \
        --num_train_epochs 100 \
        --learning_rate 0.0002 \
        --max_num_frames 2 \
        --pretrained_2d 0 \
        --backbone_coef_lr 0.5 \
        --zero_opt_stage 1 \
        --mixed_precision_method deepspeed \
        --deepspeed_fp16 \
        --gradient_accumulation_steps 1 \
        --pred_bbox_with_clip_caption \
        --text_embedding_type blip2_multimodal \
        --use_trans \
        --loss_bbox_w 5 \
        --loss_class_w 1 \
        --seed $RANDOM \
        --openset \
        --output_dir ./output/blip2/debug_blip2_openset_pedes_mlp1_lr0002_blr5_e100_multimodal_0
