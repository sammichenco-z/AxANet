CUDA_VISIBLE_DEVICES=0,1,2,3 \
OMPI_COMM_WORLD_SIZE="4" \
NCCL_P2P_DISABLE=1 \
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_port=5$[RANDOM%10000] src/tasks/run_bbox.py \
        --config src/configs/VidSwinBert/drama_default.json \
        --train_yaml drama/training_32frames.yaml \
        --val_yaml drama/testing_32frames.yaml \
        --per_gpu_train_batch_size 64 \
        --per_gpu_eval_batch_size 128 \
        --num_train_epochs 100 \
        --learning_rate 0.01 \
        --max_num_frames 2 \
        --pretrained_2d 0 \
        --backbone_coef_lr 0.05 \
        --zero_opt_stage 1 \
        --mixed_precision_method deepspeed \
        --deepspeed_fp16 \
        --gradient_accumulation_steps 1 \
        --pred_bbox \
        --loss_bbox_w 5 \
        --loss_class_w 1 \
        --seed $RANDOM \
        --output_dir ./output/test_detection
