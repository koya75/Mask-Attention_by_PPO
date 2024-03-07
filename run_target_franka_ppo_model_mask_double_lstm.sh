#!/bin/bash

model=target_mask_double
robot=128

# Start tuning hyperparameters
python target_train_franka_ppo.py \
    --outdir results/franka/${model}_lstm \
    --model ${model} \
    --epochs 10 \
    --gamma 0.99 \
    --step_offset 0 \
    --lambd 0.995 \
    --lr 0.0002 \
    --max-grad-norm 40 \
    --gpu 2 \
    --use-lstm \
    --num-envs ${robot} \
    --eval-n-runs ${robot} \
    --update-batch-interval 1 \
    --num-items 3 \
    --item-names item21 item25 item38 \
    --target item21 \
    --isaacgym-assets-dir /opt/isaacgym/assets \
    --item-urdf-dir ./urdf \
    --steps 25000000 \
    --eval-batch-interval 20 \
    --descentstep 12 \
    --mode normal \
    --hand \
    #  --render

#25000000
#results
#--hand \
#--mode normal \hard
#    --load results/${model}_lstm/20220622T182921.228762/9581440_except \
#    --use-lstm \--item-names item25 item38 item21 item22 item23 item24 \
