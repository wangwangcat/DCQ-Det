#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1

python main.py \
  --c config/DCQDet_res50_coco.py \
  --output_dir logs/DCQ-Det/Res50 \
  --batch_size 4 \
  --coco_path COCODIR \

# --resume logs/DCQ-Det/Intern/DCQDet_intern_coco.pth \
# --eval