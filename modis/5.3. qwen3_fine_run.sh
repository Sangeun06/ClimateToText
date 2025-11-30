#!/bin/bash

# CUDA 디바이스 설정
export CUDA_VISIBLE_DEVICES=4  # 사용 가능한 GPU ID

# Qwen3-VL-8B-Thinking Fine-tuning
# 메모리 부족 시 --deepspeed zero2 또는 zero3 설정 필요
swift sft \
    --model Qwen/Qwen3-VL-8B-Thinking \
    --dataset weather_train_swift.jsonl \
    --output_dir output/qwen3_weather_finetune \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --freeze_vit false \
    --max_length 8192 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --warmup_ratio 0.03 \
    --save_steps 100 \
    --eval_steps 100 \
    --logging_steps 10 \
    --model_type qwen3_vl \