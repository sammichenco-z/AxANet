# After launching the docker container 
EVAL_DIR='drama_output/transformer_baseline_pred_class_6layer_lr0002_detr_backbone/checkpoint-156-14352/'
CHECKPOINT='drama_output/transformer_baseline_pred_class_6layer_lr0002_detr_backbone/checkpoint-156-14352/model.bin'
CUDA_VISIBLE_DEVICES=1 python -m pdb src/tasks/visualize_attention_rollout.py \
       --resume_checkpoint $CHECKPOINT \
       --eval_model_dir $EVAL_DIR \
       --do_lower_case \
       --do_test