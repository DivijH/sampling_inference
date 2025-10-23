#!/bin/bash

NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=6,7
export TOKENIZERS_PARALLELISM=false

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    finetuning.py \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --data_path "open_math_instruct_train_10k/training_data/sft/tot.jsonl" \
    --output_dir "./trained_models/sft/tot_fine_tuned_llama_model" \
    --batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --epochs 2 \
    --run_name "tot-sft-Llama-3.2-3B-Instruct" \
    --save_limit 100