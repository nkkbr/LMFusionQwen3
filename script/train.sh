#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRITON_CACHE_DIR=/tmp/triton_autotune
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL

MODEL_NAME="/home/user1/U_06/20250704_LMFusion/20250715_init_weight"
DATA_PATH="/home/user1/U_06/data/cc12m-20250716" 
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="checkpoints/lmfusion_qwen3_pretrain_20250725" 

NUM_GPUS=4 

nvidia-smi
echo "Training output will be saved to: ${OUTPUT_DIR}"

alias python=python3

deepspeed --num_gpus=${NUM_GPUS} train.py \
    --deepspeed ./script/zero3_offload.json \
    --model_name_or_path ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --training_phase "pretrain" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 1069 \
    --save_total_limit 85 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to "wandb"

# usage
# conda activate lmfusion
# cd /home/user1/U_06/20250704_LMFusion
# bash script/train.sh >> script/train.log 2>&1