import os
import json
import pandas as pd
import plotly.express as px
import numpy as np
import argparse

MODEL_NAME_MAP = {
    'gpt-4o-2024-11-20': 'GPT-4o',
    'claude-3-7-sonnet-20250219 (thinking)': 'claude-3-7-sonnet (thinking)',
    'claude-3-7-sonnet-20250219': 'claude-3-7-sonnet',
    'DeepSeek-R1-Distill-Llama-70B': 'R1-Distill-Llama-70B',
    'deepseek-reasoner': 'Deepseek-R1',
    'gemini-2.0-pro-exp-02-05': 'gemini-2.0-pro',
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Mode"
    )
    parser.add_argument('--results_root', type=str, default='../eval_results', help='Root directory for evaluation results')
    return parser.parse_args()

def compute_proper_cdf(data, max_step=8):
    """Compute a proper CDF with unique step values and extend to max_step."""
    if not data:
        return pd.DataFrame(columns=["Step", "CDF"])
        
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    unique_values, counts = np.unique(sorted_data, return_counts=True)
    
    result = []
    if unique_values[0] > 0:
        result.append({"Step": 0, "CDF": 0.0})
    
    cumulative = 0
    for value, count in zip(unique_values, counts):
        cumulative += count
        result.append({
            "Step": value,
            "CDF": cumulative / n
        })
    
    if result and result[-1]["Step"] < max_step:
        last_cdf = result[-1]["CDF"]
        for step in range(int(result[-1]["Step"]) + 1, max_step + 1):
            result.append({
                "Step": step,
                "CDF": last_cdf
            })
    
    return pd.DataFrame(result)

args = parse_args()
results_root = args.results_root

model_data = {}

for model_name in os.listdir(results_root):
    model_dir = os.path.join(results_root, model_name)
    if not os.path.isdir(model_dir):
        continue

    model_dir = os.path.join(model_dir, 'english', '7x7')
    interactive_dir = os.path.join(model_dir, 'interactive')
    
    raw_metrics_path = os.path.join(interactive_dir, 'metrics', 'raw_metrics.json')
    
    if os.path.exists(raw_metrics_path):
        print(f"Processing raw_metrics for {model_name}")
        try:
            with open(raw_metrics_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            iss_list = raw_data.get("Interactive Success Step", [])
            
            if iss_list:
                model_data[model_name] = iss_list
                
        except Exception as e:
            print(f"Error processing raw_metrics for {model_name}: {e}")
            continue

all_cdf_data = []

for model_name, iss_list in model_data.items():
    model_cdf = compute_proper_cdf(iss_list)
    model_cdf["Model"] = MODEL_NAME_MAP.get(model_name, model_name)
    all_cdf_data.append(model_cdf)

if all_cdf_data:
    df_cdf = pd.concat(all_cdf_data, ignore_index=True)
else:
    print("No data found. Check your directory paths.")
    exit(1)

fig = px.line(
    df_cdf,
    x='Step',
    y='CDF',
    color='Model',
    title='CDF of Interactive Success Step per Model',
    labels={
        'Step': 'Step',
        'CDF': '', 
    },
    template='plotly_white'
)

fig.update_traces(
    mode='lines+markers',
    marker=dict(size=8),
    line=dict(width=8) 
)

fig.update_layout(
    title_x=0.5,
    margin=dict(l=80, r=80, t=150, b=100),
    title_font=dict(size=55, family='Palatino, serif', color='black', weight='bold'),
    xaxis_title_font=dict(size=50, family='Palatino, serif', color='black', weight='bold'),
    xaxis=dict(tickfont=dict(size=48, family='Palatino, serif', color='black')),
    yaxis=dict(tickfont=dict(size=48, family='Palatino, serif', color='black')),
    legend=dict(
        font=dict(size=55, family='Palatino, serif', color='black'),
        x=0.5,
        y=0.1,
    )
)

os.makedirs('../plots', exist_ok=True)
fig.write_image('../plots/interactive-step-cdf.svg', format='svg', scale=1, width=1800, height=1000)
print("CDF plot saved as ../plots/interactive-step-cdf.svg")