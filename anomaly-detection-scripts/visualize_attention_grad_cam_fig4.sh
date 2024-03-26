# After launching the docker container 
# EVAL_DIR='drama_output/transformer_baseline_pred_class_6layer_6head_lr0003/checkpoint-5-460/'
# CHECKPOINT='drama_output/transformer_baseline_pred_class_6layer_6head_lr0003/checkpoint-5-460/model.bin'
EVAL_DIR='anomaly_output/base_pretrained/checkpoint-70-8820/'
CHECKPOINT='anomaly_output/base_pretrained/checkpoint-70-8820/model.bin'
CUDA_VISIBLE_DEVICES=5 python src/tasks/visualize_attention_grad_cam_fig4.py \
       --resume_checkpoint $CHECKPOINT  \
       --eval_model_dir $EVAL_DIR \
       --do_lower_case \
       --do_test 