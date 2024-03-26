# After launching the docker container 
EVAL_DIR='output/test_caption_with_prompt_onlypromptattn/checkpoint-27-2484/'
CHECKPOINT='output/test_caption_with_prompt_onlypromptattn/checkpoint-27-2484/model.bin'
VIDEO='/DATA_EDS/zyp/jinbu/drama_lxy/drama/data/drama_test_raw_images/titan/clip_91_000402/frame_000402.png'
CUDA_VISIBLE_DEVICES=4 python -m pdb src/tasks/run_caption_VidSwinBert_inference_with_point_prompt.py \
       --resume_checkpoint $CHECKPOINT  \
       --eval_model_dir $EVAL_DIR \
       --test_video_fname $VIDEO \
       --do_lower_case \
       --do_test 