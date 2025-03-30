<div align="center">
<img src='assets/logo.png'  width=300px>

# <p align="center"><b>CrossWordBench: Evaluating the Reasoning Capabilities of LLMs and LVLMs with Controllable Puzzle Generation</b></p>
<p align="center">
<a href="">[📄 Paper]</a>
<a href="https://huggingface.co/datasets/HINT-lab/CrossWordBench">[🤗 HF Dataset]</a>
<a href="https://huggingface.co/datasets/HINT-lab/CrossWordBench-Results">[📝 Evaluation Results]</a>
</p>

---
</div>



## CrossWordBench
<div align="center">
<img src='assets/data-pipeline.png'>
</div>

This repo contains the source code (both data generation and evaluation) for $\textbf{CrossWordBench}$, a benchmark designed to evaluate the reasoning capabilities of both LLMs and LVLMs, featuring controllable puzzle generation and evaluation strategies.

**Table of Contents**
1. [Installation](#installation) 
2. [Data Generation](#data-generation)
3. [Evaluation](#evaluation)
4. [Acknowledgements](#references-and-acknowledgements)

## Setup

### Installation
```shell
# create a new conda environment
conda create -n crossword python=3.10 -y
conda activate crossword

# install necessary conda packages
conda install -c conda-forge poppler cairo pycairo expat fontconfig pygobject pango glib gobject-introspection poppler wkhtmltopdf -y

# Install vLLM (version 0.7.3 used for evaluations; 
# if your evaluated models are not supported in this version, 
# you can install the latest version via: pip install -U vllm)
pip install vllm==0.7.3

# install the remaining dependencies
pip install --no-deps -r requirements.txt
```

**API Keys**

Create a .env file in the root directory and add your API keys:

```env
OPENAI_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
FIREWORKS_AI_API_KEY=your_key_here
```


---
###  Data Generation 
We have host CrossWordBench evaluation data on [Huggingface](https://huggingface.co/datasets/HINT-lab/CrossWordBench), which can be directly used for evaluation.

> [!NOTE]
> We also provide code to automatically generate crossword puzzles. While this code is available, please be aware of potential **data contamination** if you use the generated data for training purposes.

---
#### Word-Clue Pairs curation
We provide the code used to generate the word-clue pairs for the **English**, **Chinese**, **English_Simple**, and **CommonsenseQA** puzzles used in our paper. These can be found in the ```word_lists``` directory. Additional information about the sources of these word lists is also available in that directory.

If you wish to create your own word-clue pairs, ensure they are in .txt format, with each word-clue pair on a separate line, separated by a space. You can refer to ```word_lists/english_words_simple.txt``` for an example.

---
#### Puzzle Generation
We provide bash scripts for generating puzzles across the full dataset, as well as scripts for generating a few sample puzzles for testing. You can check configurations and parameters in ```scripts/gen_full.sh``` or directly in ```gen_crossword.py```.
```shell
# Generate the full dataset
# This generates 100 samples for both 7x7 and 14x14 grid sizes using english wordlist.
# It also generates partially filled puzzles at three pre-fill ratios (0.25, 0.5, and 0.75),
# as well as a leave-one-out option where only one word is unfilled in each puzzle.
bash scripts/gen_full.sh -i word_lists/english_words.txt -m false -s 100

# Generate a few sample puzzles for quick testing
bash scripts/gen_puzzle.sh
```


### Evaluation

You can find our evaluation results on [Huggingface](https://huggingface.co/datasets/HINT-lab/CrossWordBench-Results).

---
#### Supported Models
Currently, we support only API-based models and models that can be deployed online using **vLLM**, in order to maintain a unified inference interface.

You can find the list of supported models in the [vLLM documentation](https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-multimodal-language-models).

---
#### Instructions
**Step 1**: Host a model using vLLM.

As mentioned, only API-based models or models deployable with vLLM are supported. If you wish to add your own custom model, you can implement it in ```eval/model.py``` and adjust the prompt template in ```eval/template.py```.

We also provide an example bash script ```scripts/deploy.sh``` to help you set up a vLLM server.

Example:
```shell
# You can customize parameters in scripts/configs/deploy_config.json,
# or add your own model configuration there.
bash scripts/deploy.sh Qwen/Qwen2-VL-72B-Instruct 
```

**Step 2**: Evaluate models.

We provide an example bash script, ```scripts/run_eval.sh```, for evaluating models on CrossWordBench.

```shell
# Example: Evaluate GPT-4o on English puzzles with 7x7 grid size,
# using zero-shot Chain-of-Thought (CoT) prompting on images
bash scripts/run_eval.sh -m gpt-4o-2024-11-20 -s english -d 7x7 -t img_cot

# You can also evaluate multiple tasks, grid sizes, and prompt types at once. For example:

# Evaluate both "img_cot" and "interactive" prompt templates
bash scripts/run_eval.sh -m gpt-4o-2024-11-20 -s english -d 7x7 -t img_cot,interactive

# Evaluate on both English and Chinese puzzles
bash scripts/run_eval.sh -m gpt-4o-2024-11-20 -s english,chinese -d 7x7 -t img_cot,interactive

# Evaluate on multiple categories and grid sizes
bash scripts/run_eval.sh -m gpt-4o-2024-11-20 -s english,chinese -d 7x7,14x14 -t img_cot,interactive

# Use -h to see available options and parameters
bash scripts/run_eval.sh -h
```
> [!NOTE]
> You must have ```yq``` installed, as we use it to parse model-specific generation configurations from ```scripts/configs/models.yaml```.

You can review and modify the configurations in ```scripts/configs/models.yaml``` based on your needs. If your model is not listed, feel free to add it and define custom settings accordingly.

---
#### Supported Prompt Templates

We support multiple prompting strategies, including:

- **Zero-shot Chain-of-Thought (CoT)** on both images and text
- **Interactive mode** on images
- **Grid-parsing**

You can find more experimental features in ```eval/template.py```. If you want to implement your own prompting strategies, you can modify ```eval/template.py```.

```shell
# Prompt template abbreviations (defined in template.py):
# Below are the templates used in our paper
-t img_cot,text_cot,interactive,extraction
```

Feel free to modify ```template.py``` to suit your specific use cases or research needs.

---
#### Reported Metrics
After evaluation, a `metrics` directory will be created containing the following files:

- `metric.json` — Aggregated metrics across all evaluated puzzles.
- `metric.txt` — A human-readable version of the metrics for quick reference.
- `raw_metrics.json` — Pre-aggregated metrics for each individual puzzle.

These outputs help analyze both high-level performance and per-puzzle behavior.

---
#### Plot Tools

We provide the tools used to generate all the figures presented in our paper. You can find them in the `tools` directory.

> ⚠️ Note: You may need to modify the scripts to fit your working environment and result structure, as they assume a complete `eval_results` folder is available.


## References and Acknowledgements

Our data generation pipeline builds upon [genxword](https://github.com/riverrun/genxword), an open-source project licensed under the GNU General Public License v3.0 (GPLv3).

We sincerely thank the project and its contributors for their valuable work.


## Citation
Please cite our paper if you find the repo helpful in your work:
```bibtex

```

