# CrossWordBenchmark Evaluation Framework

A simplified framework for evaluating language models on crossword puzzles.

## Setup

## Usage

### Basic Usage

Run evaluations by specifying the model, subject, difficulty, and template type:

```bash
./run_eval.sh -m gpt-4o-2024-11-20 -s english -d 7x7 -t img_cot
```

### API Model Examples (OpenAI, Anthropic, Google)

For these models, set environment variables for the respective API keys:

```bash
# OpenAI example
./run_eval.sh -m gpt-4o-2024-11-20 -s english -d 7x7 -t img_cot

# Anthropic example
./run_eval.sh -m anthropic/claude-3-5-sonnet-latest -s english -d 7x7 -t img_cot

# Google example
./run_eval.sh -m gemini/gemini-2.0-flash -s english -d 7x7 -t img_cot
```

### vLLM Models

For models served via vLLM, specify the server URL and optional API key:

```bash
./run_eval.sh -m Qwen/Qwen2-VL-72B-Instruct -s english -d 7x7 -t img_cot -v http://localhost:8000/v1
```

With authentication:

```bash
./run_eval.sh -m deepseek-ai/deepseek-vl2 -s english -d 7x7 -t img_cot -v http://your-server:8000/v1 -k your_api_key
```

### Command Line Options

```
-m model         Model name from models.yaml
-s subject       Subject type (english or anagram)
-d difficulty    Grid size (7x7, 14x14, or 21x21)
-t template_type Template type (img_cot, text_cot, extraction)
-r effort        Reasoning effort (low, medium, high) - default: high
-f results       First round results path (for two-stage evaluations)
-v vllm_server   vLLM server URL - default: http://localhost:8000/v1
-k api_key       API key for vLLM server (if needed)
-h               Show help message
```

## Model Configuration

All model configurations are stored in `configs/models.yaml`. Each model entry includes:

- `api_type`: The API provider (openai, anthropic, google, vllm)
- `parallel`: Number of parallel processes for evaluation
- `max_completion_tokens`: Maximum tokens for completion, by grid size

## Multiple Evaluations

You can specify multiple difficulties or template types with comma separation:

```bash
cd CrossWordBenchmark
# Multiple difficulties
./scripts/run_eval.sh -m gpt-4o-2024-11-20 -s english -d 7x7,14x14 -t img_cot

# Multiple template types
./scripts/run_eval.sh -m o3-mini -s english -d 7x7 -t text_cot

# For two Rounds you just need to add _two_round after prompt type, for example img_cot_two_round, text_cot_two_round, and set the path to first-round results (this is for the model to read first-round conversation to construct follow-up message)
./scripts/run_eval.sh -m anthropic/claude-3-7-sonnet-20250219 -s chinese,english_simple,commonsenseqa -d 7x7 -t text_cot_two_round -f ./path/previous_results
```
fireworks_ai/accounts/fireworks/models/deepseek-r1
anthropic/claude-3-7-sonnet-20250219
gemini/gemini-2.0-pro-exp-02-05

## To add more models
If you want to add more model and avoid using default settings, You can modify the configs/deploy.json to specific vllm parameters
configs/model.yaml to specify model-specific generation configs