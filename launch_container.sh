export REPO_DIR=$PWD
DATA_DIR='/DATA_EDS/zyp/jinbu/processed/'
RAW_VIDEO_DIR='/DATA_EDS/zyp/jinbu/combined/'
INPAINTED_DIR='/DATA_EDS/zyp/jinbu/drama_lxy/drama_inpainted_images_v2/'
MODEL_DIR=$REPO_DIR'/models/'
OUTPUT=$REPO_DIR'/output/'


if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi

if [ "$1" = "--prepro" ]; then
    RO=""
else
    RO=",readonly"
fi

docker run --name lxy -p 8089:22  --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host  -it \
    --mount src=$(pwd),dst=/videocap,type=bind \
    --mount src=$DATA_DIR,dst=/videocap/datasets,type=bind$RO \
    --mount src=$RAW_VIDEO_DIR,dst=/videocap/raw_videos,type=bind$RO \
    --mount src=$INPAINTED_DIR,dst=/videocap/inpainted_img,type=bind$RO \
    --mount src=$MODEL_DIR,dst=/videocap/models,type=bind,readonly \
    --mount src=$OUTPUT,dst=/videocap/output,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /videocap jxbbb/adapt:latest \
    bash -c "source /videocap/setup.sh && bash" 
