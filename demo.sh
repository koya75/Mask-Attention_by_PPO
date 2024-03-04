#!/bin/bash

model=target_mask_double
#model=noattention
load="results/franka/target_mask_double_lstm/20231201T182104.783471/best"
#load="results/noattention_lstm/20220626T101450.974806/best"

# Start tuning hyperparameters
python target_train_franka_ppo.py \
    --outdir results/franka/demo/${model} \
    --model ${model} \
    --num-envs 12 \
    --num-items 3 \
    --item-names item21 item25 item38 \
    --target item21 \
    --isaacgym-assets-dir /opt/isaacgym/assets \
    --item-urdf-dir ./urdf \
    --load ${load} \
    --gpu 0 \
    --descentstep 12 \
    --demo \
    --mode normal \
    --hand \
    --render