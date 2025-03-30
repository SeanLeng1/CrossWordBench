#!/bin/bash

# export NCCL_P2P_DISABLE=1
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [ -z "$1" ]; then
    echo "Please provide a model name as an argument"
    exit 1
fi

model="$1"

if ! config=$(cat scripts/configs/deploy_config.json); then
    echo "Error reading config.json"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "jq is required but not installed. Please install jq (commandline JSON processor) first."
    exit 1
fi

if ! seq_len=$(echo "$config" | jq -r ".[\"$model\"].max_seq_len"); then
    echo "Error: Could not find model configuration for $model"
    exit 1
fi

if ! n_gpus=$(echo "$config" | jq -r ".[\"$model\"].num_gpus"); then
    echo "Error: Could not find GPU configuration for $model"
    exit 1
fi

extra_args=$(echo "$config" | jq -r ".[\"$model\"].extra_args // \"\"")

echo "Model: $model"
echo "Seq Len: $seq_len"
echo "N_GPUS: $n_gpus"
echo "Extra Args: $extra_args"

vllm serve $model \
    --task generate \
    --dtype auto \
    --api-key abc123 \
    --trust-remote-code \
    --max-model-len $seq_len \
    --tensor-parallel-size  $n_gpus \
    --gpu-memory-utilization 0.85 \
    --max-seq-len-to-capture $seq_len \
    --seed 0 \
    $extra_args \

# --distributed-executor-backend ray \
# --disable-custom-all-reduce \

# bash scripts/deploy2.sh Qwen/Qwen2-VL-72B-Instruct
# bash scripts/deploy2.sh OpenGVLab/InternVL2_5-78B-MPO
# bash scripts/deploy2.sh mistralai/Pixtral-Large-Instruct-2411
# bash scripts/deploy2.sh llava-hf/llava-onevision-qwen2-72b-ov-chat-hf
# bash scripts/deploy2.sh deepseek-ai/deepseek-vl-7b-chat
# bash scripts/deploy2.sh openbmb/MiniCPM-V-2_6
# bash scripts/deploy2.sh Qwen/Qwen2-VL-7B-Instruct
# bash scripts/deploy2.sh meta-llama/Llama-3.3-70B-Instruct
# bash scripts/deploy2.sh allenai/Molmo-72B-0924
# bash scripts/deploy2.sh microsoft/Phi-3.5-vision-instruct
# bash scripts/deploy2.sh rhymes-ai/Aria
# bash scripts/deploy2.sh meta-llama/Llama-3.2-11B-Vision-Instruct
# bash scripts/deploy2.sh meta-llama/Llama-3.2-90B-Vision-Instruct
# bash scripts/deploy2.sh deepseek-ai/deepseek-vl2
# bash scripts/deploy2.sh Qwen/QwQ-32B-Preview
# bash scripts/deploy2.sh Qwen/QVQ-72B-Preview
# bash scripts/deploy2.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
# bash scripts/deploy2.sh deepseek-ai/DeepSeek-R1-Distill-Llama-70B
# bash scripts/deploy2.sh Qwen/Qwen2.5-72B-Instruct