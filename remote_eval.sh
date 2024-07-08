#!/bin/bash

bash /nfs/volume-902-16/tangwenbo/s3_all.sh
bash /nfs/ofs-902-1/object-detection/tangwenbo/ofs-vlm.sh

echo "---eval---"
cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/MiniCPM-V && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda3/envs/torch230/bin/python \
  eval_all.py --model_path "$1" --split "$2" --provide_bbox "$3"
