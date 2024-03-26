# After launching the docker container 
EVAL_DIR='output/transformer_baseline_pred_class_1layer/checkpoint-200-18400/'
CHECKPOINT='output/transformer_baseline_pred_class_1layer/checkpoint-200-18400/model.bin'
CUDA_VISIBLE_DEVICES=1 python -m pdb src/tasks/visualize_attention_grad_cam.py \
       --resume_checkpoint $CHECKPOINT  \
       --eval_model_dir $EVAL_DIR \
       --do_lower_case \
       --do_test 