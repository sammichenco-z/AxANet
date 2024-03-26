CUDA_VISIBLE_DEVICES=3,4,5,6 \
OMPI_COMM_WORLD_SIZE="4" \
NCCL_P2P_DISABLE=1 \
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_port=5$[RANDOM%10000] src/tasks/run_caption_VidSwinBert_with_point_prompt.py \
        --config src/configs/VidSwinBert/drama_default.json \
        --train_yaml drama/training_32frames.yaml \
        --val_yaml drama/testing_32frames.yaml \
        --per_gpu_train_batch_size 32 \
        --per_gpu_eval_batch_size 64 \
        --num_train_epochs 40 \
        --learning_rate 0.0004 \
        --max_num_frames 2 \
        --pretrained_2d 0 \
        --backbone_coef_lr 0.05 \
        --mask_prob 0.5 \
        --max_masked_token 45 \
        --zero_opt_stage 1 \
        --mixed_precision_method deepspeed \
        --deepspeed_fp16 \
        --gradient_accumulation_steps 1 \
        --loss_bbox_w 5 \
        --seed $RANDOM \
        --output_dir ./output/test_caption_with_prompt_onlypromptattn