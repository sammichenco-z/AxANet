# After launching the docker container 
EVAL_DIR='output/test_detection/checkpoint-100-18500/'
CHECKPOINT='output/test_detection/checkpoint-100-18500/model.bin'
CUDA_VISIBLE_DEVICES=6 python -m pdb src/tasks/run_bbox_inference.py \
       --resume_checkpoint $CHECKPOINT  \
       --eval_model_dir $EVAL_DIR \
       --do_lower_case \
       --do_test 