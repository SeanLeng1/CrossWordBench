import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import math

MODELS = ['claude-3-7-sonnet-20250219', 'deepseek-reasoner', 'QwQ-32B', 'Pixtral-Large-Instruct-2411']
MODEL_NAME_MAP = {
    'gpt-4o-2024-11-20': 'gpt-4o',
    'claude-3-7-sonnet-20250219': 'claude-3-7-sonnet',
    'DeepSeek-R1-Distill-Llama-70B': 'R1-Distill-Llama-70B',
    'deepseek-reasoner': 'Deepseek-R1',
    'Pixtral-Large-Instruct-2411': 'Pixtral-Large-Instruct',
    'QwQ-32B': 'QwQ-32B'
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Self-Reflection Effectiveness using individual radar plots"
    )
    parser.add_argument('--results_root', type=str, default='../eval_results', help='Root directory for evaluation results')
    return parser.parse_args()

def calculate_differences(df):
    """Calculate the differences between second_round and first_round for each metric"""
    result_df = df.copy()
    for metric in metrics_config:
        first_key = f"first_round_{metric['key']}"
        second_key = f"second_round_{metric['key']}"
        diff_key = f"diff_{metric['key']}"
        
        if first_key in result_df.columns and second_key in result_df.columns:
            result_df[diff_key] = result_df[second_key] - result_df[first_key]
    
    return result_df

args = parse_args()
results_root = args.results_root

text_settings = {
    'first_round': 'text_cot',
    'second_round': 'text_cot_two_round',
    'category': 'LLM'
}

img_settings = {
    'first_round': 'img_cot',
    'second_round': 'img_cot_two_round',
    'category': 'LVLM'
}

legend_mapping = {
    'first_round': "Without Self-Reflection",
    'second_round': "With Self-Reflection"
}

metrics_config = [
    {"key": "Word Coverage Rate_all_mean", "display": "WCR"},
    {"key": "Letter Coverage Rate_across_mean", "display": "LCR"},
    {"key": "Intersection Rate_mean", "display": "ICR"}
]

puzzle_size = '7x7'

def collect_data(first_round, second_round, category):
    data_records = []
    
    for model_name in os.listdir(results_root):
        if model_name not in MODELS:
            continue
            
        model_path = os.path.join(results_root, model_name)
        if not os.path.isdir(model_path):
            continue

        model_dir = os.path.join(model_path, 'english', puzzle_size)
        metrics = {}
        
        for condition_name, condition_path in [('first_round', first_round), ('second_round', second_round)]:
            condition_dir = os.path.join(model_dir, condition_path)
            if os.path.isdir(condition_dir):
                metrics_json_path = os.path.join(condition_dir, 'metrics', 'metrics.json')
                if os.path.exists(metrics_json_path):
                    try:
                        with open(metrics_json_path, 'r', encoding='utf-8') as f:
                            metrics_data = json.load(f)
                        
                        # Create a record for this condition
                        for metric in metrics_config:
                            metric_key = f"{condition_name}_{metric['key']}"
                            metric_value = metrics_data.get(metric["key"])
                            if metric_value is not None:
                                metrics[metric_key] = metric_value
                    except Exception as e:
                        print(f"Error processing {model_name} for {condition_path}: {e}")
        
        required_keys = [f"{cond}_{metric['key']}" for cond in ['first_round', 'second_round'] for metric in metrics_config]
        if all(key in metrics for key in required_keys):
            display_name = MODEL_NAME_MAP.get(model_name, model_name)
            metrics['Model'] = display_name
            metrics['Category'] = category
            data_records.append(metrics)
    
    return data_records

text_data = collect_data(text_settings['first_round'], text_settings['second_round'], text_settings['category'])
img_data = collect_data(img_settings['first_round'], img_settings['second_round'], img_settings['category'])

# combine all data but ensure models are kept separate by category
# this is important for models like GPT-4o that might be in both categories
for item in text_data:
    item['Model'] = f"{item['Model']} ({item['Category']})"

for item in img_data:
    item['Model'] = f"{item['Model']} ({item['Category']})"

all_data = text_data + img_data
df = pd.DataFrame(all_data)

n_cols = 2
n_models = len(df)
n_rows = math.ceil(n_models / n_cols)

text_df = df[df['Category'] == text_settings['category']]
img_df = df[df['Category'] == img_settings['category']]

text_df = calculate_differences(text_df)
img_df = calculate_differences(img_df)

print("Text prompting data (LLMs):")
text_columns = ['Model']
for metric in metrics_config:
    text_columns.extend([
        f"first_round_{metric['key']}", 
        f"second_round_{metric['key']}", 
        f"diff_{metric['key']}"
    ])
print(text_df[text_columns])

print("\nImage prompting data (VLMs):")
img_columns = ['Model']
for metric in metrics_config:
    img_columns.extend([
        f"first_round_{metric['key']}", 
        f"second_round_{metric['key']}", 
        f"diff_{metric['key']}"
    ])
print(img_df[img_columns])

print(f"\nCollected data for {len(df)} models from the MODELS list")

plot_data = []

for _, row in df.iterrows():
    model = row['Model']
    category = row['Category']
    
    for condition in ['first_round', 'second_round']:
        condition_label = legend_mapping[condition]
        
        for metric in metrics_config:
            metric_key = f"{condition}_{metric['key']}"
            if metric_key in row:
                plot_data.append({
                    'Model': model,
                    'Category': category,
                    'Condition': condition_label,
                    'Metric': metric['display'],
                    'Value': row[metric_key]
                })

plot_df = pd.DataFrame(plot_data)

fig = make_subplots(
    rows=n_rows, 
    cols=n_cols,
    specs=[[{"type": "polar"} for _ in range(n_cols)] for _ in range(n_rows)],
    subplot_titles=[f"{row['Model']}" for _, row in df.iterrows()] + [""] * (n_rows * n_cols - n_models),
    horizontal_spacing=0.0,  # set to 0 for minimal gap
    vertical_spacing=0.2
)

colors = {
    "Without Self-Reflection": "#b7daf5",
    "With Self-Reflection": "#f9bdb6"
}

min_val = max(0, plot_df['Value'].min() * 0.7) 
max_val = plot_df['Value'].max() * 1.2 

for i, (_, model_row) in enumerate(df.iterrows()):
    model_name = model_row['Model']
    row_idx = i // n_cols + 1
    col_idx = i % n_cols + 1
    
    model_data = plot_df[plot_df['Model'] == model_name]
    
    for condition in legend_mapping.values():
        condition_data = model_data[model_data['Condition'] == condition]
        condition_data = condition_data.sort_values('Metric')
        
        r_values = condition_data['Value'].tolist() + [condition_data['Value'].iloc[0]]
        theta_values = condition_data['Metric'].tolist() + [condition_data['Metric'].iloc[0]]
        
        fig.add_trace(
            go.Scatterpolar(
                r=r_values,
                theta=theta_values,
                name=condition,
                line=dict(
                    color=colors[condition],
                    width=2
                ),
                showlegend=True if i == 0 else False  
            ),
            row=row_idx, col=col_idx
        )

fig.update_layout(
    title=dict(
        # text="Self-Reflection Effectiveness by Model",
        text="",
        font=dict(size=32, family='Palatino, serif', color='black', weight='bold'),
        # y=1,
        x=0.5
    ),
    margin=dict(l=0, r=0, t=40, b=5),
    title_pad=dict(t=20),
    showlegend=True,
    legend=dict(
        title="",
        font=dict(size=24, family='Palatino, serif', color='black'),
        orientation="h",
        y=-0.05,
        x=0.5,
        xanchor="center"
    ),
    width=650,
    height=200 * n_rows,
    font=dict(size=20)  
)

for i in range(1, n_models + 1):
    polar_key = f"polar{i if i > 1 else ''}"
    fig.update_layout(**{
        polar_key: dict(
            radialaxis=dict(
                visible=True,
                range=[min_val, max_val],
                tickfont=dict(size=13),
                linecolor='black',
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                tickfont=dict(size=18),
                linecolor='black',
                gridcolor='lightgray'
            ),
            bgcolor='rgba(0,0,0,0)'  
        )
    })

for i in range(len(fig.layout.annotations)):
    if i < n_models:  
        fig.layout.annotations[i].font.size = 24
        fig.layout.annotations[i].font.family = 'Palatino, serif'
        fig.layout.annotations[i].font.color = 'black'
        fig.layout.annotations[i].font.weight = 'bold'
        fig.layout.annotations[i].y = fig.layout.annotations[i].y + 0.03

os.makedirs('../plots', exist_ok=True)
save_name = '../plots/self_reflection_individual_radar_comparison.svg'
fig.write_image(save_name, format='svg', scale=1)

print(f"Radar plot saved as {save_name}")
