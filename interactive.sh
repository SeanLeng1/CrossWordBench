#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number_of_gpus>"
    exit 1
fi

gpu_num=$1

if [ "$gpu_num" -eq 0 ]; then
    bsub -m "jiaxinh01" -q interactive -Is /bin/bash
else
    bsub -gpu "num=$gpu_num:gmodel=NVIDIAA100_SXM4_80GB" -m "jiaxinh01" -q interactive -Is /bin/bash
fi


bsub -gpu "num=1" -q interactive  -Is /bin/bash
bsub -q interactive  -Is /bin/bash
bsub -gpu "num=4" -m "jiaxinh01" -q interactive -Is /bin/bash
bsub -gpu "num=1" -m "jiaxinh02" -q interactive -Is /bin/bash
bsub -m jiaxinh02 -a 'docker(continuumio/anaconda3)' -gpu "num=1:gmodel=NVIDIAA100_SXM4_80GB" -q interactive -Is /bin/bash

cd /storage1/jiaxinh/Active/jixuan/VLM/CrossWordBenchmark

conda activate /ib-scratch/jiaxinh02/jixuan/envs/cw

/ib-scratch/jiaxinh02/project/jixuan/VLM_Backbone_Checkpoints

python utils/convert.py --percentage 25 --vlm-type llava-next --selection largest --structure channel

bash scripts/opencompass_single.sh -m /ib-scratch/jiaxinh02/project/jixuan/VLM_Backbone_Checkpoints/llama3-llava-next-8b-hf-elementwise-merged -d gsm8k_gen_1d7fe4 -t chat -n 4

bash scripts/opencompass_single.sh -m Efficient-Large-Model/Llama-3-VILA1.5-8B -d gsm8k_gen_1d7fe4 -t chat -n 4


python convert_llava_weights_to_hf.py --text_model_id meta-llama/Meta-Llama-3-8B-Instruct --vision_model_id openai/clip-vit-large-patch14-336 --output_hub_path HINT-lab/open-llava-next-llama3-8b-hf --old_state_dict_id Lin-Chen/open-llava-next-llama3-8b



conda install -c conda-forge cairo pycairo expat fontconfig pygobject pango glib gobject-introspection
pip install genxword