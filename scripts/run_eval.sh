#!/bin/bash
# run_eval.sh - Simplified evaluation runner for CrossWordBench
# Usage: ./run_eval.sh [-m model] [-s subject] [-d difficulty] [-t template_type] [-h]

set -e

# Default paths
CONFIG_DIR="./scripts/configs"
MODELS_CONFIG="$CONFIG_DIR/models.yaml"
DEFAULT_VLLM_SERVER="http://localhost:8000/v1"  # Default vLLM server address

# Default values
model=""
subject=""
difficulty=""
template_type=""
first_round_results="none"
vllm_server=""      # Empty by default, will fall back to DEFAULT_VLLM_SERVER if needed
api_key="abc123"    # vllm api key
lora_module=""

# Parse command line arguments
function show_help {
    echo "Usage: ./run_eval.sh [-m model] [-s subject] [-d difficulty] [-t template_type] [-r effort] [-v vllm_server] [-h]"
    echo ""
    echo "Options:"
    echo "  -m model         Model name from models.yaml (e.g., gpt-4o-2024-11-20)"
    echo "  -s subject       Subject type - english or anagram"
    echo "  -d difficulty    Grid size - 7x7, 14x14, or 21x21"
    echo "  -t template_type Template type (e.g., img_cot, text_cot, extraction)"
    echo "  -f results       First round results path (for two-stage evaluations)"
    echo "  -v vllm_server   vLLM server URL (for locally served models) - default: $DEFAULT_VLLM_SERVER"
    echo "  -k api_key       API key for vLLM server (only needed for vLLM models)"
    echo "  -h               Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_eval.sh -m gpt-4o-2024-11-20 -s english -d 7x7 -t img_cot  # Run specific configuration"
    echo "  ./run_eval.sh -m Qwen/Qwen2-VL-72B-Instruct -s english -d 7x7 -t img_cot -v http://localhost:8000/v1 -k your_vllm_key  # Run vLLM model"
}

while getopts "m:s:d:t:r:f:v:k:h" opt; do
    case $opt in
        m) model=$OPTARG ;;
        s) subject=$OPTARG ;;
        d) difficulty=$OPTARG ;;
        t) template_type=$OPTARG ;;
        f) first_round_results=$OPTARG ;;
        v) vllm_server=$OPTARG ;;
        k) api_key=$OPTARG ;;
        h) show_help; exit 0 ;;
        *) echo "Unknown option: -$OPTARG" >&2; show_help; exit 1 ;;
    esac
done

# Check if required tools are available
command -v python >/dev/null 2>&1 || { echo "Python is required but not installed. Aborting."; exit 1; }
command -v yq >/dev/null 2>&1 || { echo "yq is required but not installed. Please install it with 'pip install yq'. Aborting."; exit 1; }

# Create config directory if it doesn't exist
mkdir -p "$CONFIG_DIR"

# Check if config files exist, create them if they don't
if [ ! -f "$MODELS_CONFIG" ]; then
    echo "Creating example models config file at $MODELS_CONFIG"
    cp models.yaml "$MODELS_CONFIG" 2>/dev/null || echo "Warning: Could not create $MODELS_CONFIG"
fi

# Function to get a value from YAML using yq
function get_yaml_value() {
    local file=$1
    local path=$2
    local default=$3
    
    # Check if file exists
    if [ ! -f "$file" ]; then
        echo "$default"
        return
    fi
    
    # Get value with yq - using eval to handle model names with special characters properly
    local result=$(yq -r "$path" "$file")
    
    # Check if result is null or empty
    if [ "$result" = "null" ] || [ -z "$result" ]; then
        echo "$default"
    else
        echo "$result"
    fi
}

# Validate required parameters
if [ -z "$model" ]; then
    echo "Error: Model is required"
    show_help
    exit 1
fi

if [ -z "$subject" ]; then
    echo "Error: Subject is required"
    show_help
    exit 1
fi

if [ -z "$difficulty" ]; then
    echo "Error: Difficulty is required"
    show_help
    exit 1
fi

if [ -z "$template_type" ]; then
    echo "Error: Template type is required"
    show_help
    exit 1
fi

# Get model specific configuration
api_type=$(get_yaml_value "$MODELS_CONFIG" ".models[\"$model\"].api_type" "openai")
echo "API Type detected: $api_type"

parallel=$(get_yaml_value "$MODELS_CONFIG" ".models[\"$model\"].parallel" 8)
parsing_model=$(get_yaml_value "$MODELS_CONFIG" ".default.parsing_model" "o3-mini-2025-01-31")
mm_limit=$(get_yaml_value "$MODELS_CONFIG" ".default.mm_limit" 4)
budget_tokens=$(get_yaml_value  "$MODELS_CONFIG" ".models[\"$model\"].budget_tokens" 0)
reasoning_effort=$(get_yaml_value "$MODELS_CONFIG" ".models[\"$model\"].reasoning_effort" "medium")
lora_module=$(get_yaml_value "$MODELS_CONFIG" ".models[\"$model\"].lora_module" "")
temperature=$(get_yaml_value "$MODELS_CONFIG" ".models[\"$model\"].temperature" 0.0)

# Handle anagram flag based on subject
anagram_flag=""
if [[ "$subject" == *"anagram"* ]]; then
    echo "Detected anagram subject."
    anagram_flag="--anagram_mix"
fi

# Setup API access for vLLM models
api_base=""
if [[ "$api_type" == "vllm" || "$api_type" == "vllm_lora" ]]; then
    # Use provided vLLM server URL or fall back to default
    if [ -z "$vllm_server" ]; then
        if [[ "$api_type" == "vllm_lora" ]]; then
            api_base="http://localhost:8000/v1/chat/completions"
            echo "Using default vLLM LoRA server: $api_base"
        else
            api_base="$DEFAULT_VLLM_SERVER"
            echo "No vLLM server specified, using default: $api_base"
        fi
    else
        api_base="$vllm_server"
        echo "Using specified vLLM server: $api_base"
    fi
    
    # Check if API key is needed for this vLLM server
    if [ -n "$api_key" ]; then
        echo "Using provided API key for vLLM server"
    else
        echo "No API key provided for vLLM server. Assuming no authentication is required."
    fi
fi

# Default output settings
output_dir="./eval_results"
eval_name="crossword"
# dataset_name="JixuanLeng/CrossWordBench"
dataset_name="HINT-lab/CrossWordBench"
vllm_flag=""
lora_module_flag=""

echo "================================================"
echo "Running evaluation with the following settings:"
echo "Model: $model"
echo "Subject: $subject"
echo "Difficulty: $difficulty"
echo "Template type: $template_type"
echo "API type: $api_type"
echo "Budget tokens: $budget_tokens"
echo "LoRA module: $lora_module"
echo "Temperature: $temperature"

# vllm lora is not using the same api as vllm
if [[ "$api_type" == "vllm" ]]; then
    vllm_flag="--use_vllm"
    echo "vLLM server: $api_base"
fi
if [[ "$api_type" == "vllm_lora" ]]; then
    lora_module_flag="--lora_module $lora_module"
    echo "vLLM LoRA server: $api_base"
fi

echo "Reasoning effort: $reasoning_effort"
echo "Parallel: $parallel"
echo "================================================"

# Run for each difficulty (may be comma-separated)
IFS=',' read -ra SUBJECT_ARRAY <<< "$subject"
for sub in "${SUBJECT_ARRAY[@]}"; do
    sub=$(echo "$sub" | xargs)  # Trim whitespace
    echo "Processing subject: $sub"


    IFS=',' read -ra DIFFICULTY_ARRAY <<< "$difficulty"
    for diff in "${DIFFICULTY_ARRAY[@]}"; do
        diff=$(echo "$diff" | xargs)  # Trim whitespace
        echo "Processing difficulty: $diff"
        
        # Get max_completion_tokens for this specific difficulty
        max_completion_tokens=$(get_yaml_value "$MODELS_CONFIG" ".models[\"$model\"].max_completion_tokens[\"$diff\"]" 8192)
        echo "Max completion tokens for $diff: $max_completion_tokens"
        
        # Run for each template type (may be comma-separated)
        IFS=',' read -ra TEMPLATE_ARRAY <<< "$template_type"
        for temp in "${TEMPLATE_ARRAY[@]}"; do
            temp=$(echo "$temp" | xargs)  # Trim whitespace
            echo "Processing template type: $temp"
            
            # Base command with common parameters
            cmd="python -m eval.run \
                --parallel \"$parallel\" \
                --mm_limit $mm_limit \
                --eval_name \"$eval_name\" \
                --dataset_name \"$dataset_name\" \
                --model \"$model\" \
                --parsing_model \"$parsing_model\" \
                --template_type \"$temp\" \
                --output_dir \"$output_dir\" \
                --subject \"$sub\" \
                --difficulty \"$diff\" \
                --first_round_results \"$first_round_results\" \
                --max_completion_tokens $max_completion_tokens \
                --reasoning_effort \"$reasoning_effort\" \
                --budget_tokens $budget_tokens \
                --temperature $temperature \
                $anagram_flag \
                $vllm_flag \
                $lora_module_flag"
            
            # Add API parameters only if needed (for vLLM)
            if [[ "$api_type" == "vllm" || "$api_type" == "vllm_lora" ]]; then
                cmd+=" --api_base \"$api_base\""
                if [ -n "$api_key" ]; then
                    cmd+=" --api_key \"$api_key\""
                fi
            fi
            
            # Show the full command with redacted API key for debugging
            cmd_display="$cmd"
            if [ -n "$api_key" ]; then
                cmd_display="${cmd_display//--api_key \"$api_key\"/--api_key \"REDACTED\"}"
            fi
            # echo "Executing command: $cmd_display"
            
            # Execute the command
            eval "$cmd"
                
            echo "Completed evaluation for $subject $diff with $temp template using $model"
        done
    done
done

echo "All evaluations completed successfully!"