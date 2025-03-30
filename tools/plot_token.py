import os
import json
import pandas as pd
import plotly.express as px
import argparse

MODEL_NAME_MAP = {
    'gpt-4o-2024-11-20': 'GPT-4o',
    'claude-3-7-sonnet-20250219': 'claude-3-7-sonnet',
    'claude-3-7-sonnet-20250219 (thinking)': 'claude-3-7-sonnet\U0001F4A1',
    'DeepSeek-R1-Distill-Llama-70B': 'R1-Distill-Llama-70B',
    'deepseek-reasoner': 'Deepseek-R1',
    'llava-onevision-qwen2-72b-ov-chat-hf': 'llava-onevision-72b',
    'gemini-2.0-pro-exp-02-05': 'gemini-2.0-pro',
    'llama-v3p1-405b-instruct': 'llama-405b-instruct',
    'o3-mini': 'o3-mini (high)',
}


REASONING_MODELS = {
    'o3-mini',
    'o3-mini (medium)',
    'o3-mini (low)',
    'deepseek-reasoner',
    'QwQ-32B',
    'DeepSeek-R1-Distill-Llama-70B',
    'QVQ-72B-Preview',
    'Open-Reasoner-Zero-32B',
    'claude-3-7-sonnet-20250219 (thinking)',
}

def parse_args():
    parser = argparse.ArgumentParser(description="Plot token usage comparison for models")
    parser.add_argument('--text', action='store_true', help='Use text COT instead of image COT')
    parser.add_argument('--results_root', type=str, default='../eval_results', help='Root directory for evaluation results')
    return parser.parse_args()

args = parse_args()
cot_file = 'img_cot' if not args.text else 'text_cot'

results_root = args.results_root

data_records = []

puzzle_sizes = ['7x7', '14x14']

for model_name in os.listdir(results_root):
    model_path = os.path.join(results_root, model_name)
    if not os.path.isdir(model_path):
        continue

    for puzzle_size in puzzle_sizes:
        model_dir = os.path.join(model_path, 'english', puzzle_size)
        img_cot_dir = os.path.join(model_dir, cot_file)
        
        if os.path.isdir(img_cot_dir):
            metrics_json_path = os.path.join(img_cot_dir, 'metrics', 'metrics.json')
            if os.path.exists(metrics_json_path):
                print(f"Processing {model_name} for {puzzle_size}")
                try:
                    with open(metrics_json_path, 'r', encoding='utf-8') as f:
                        metrics_data = json.load(f)
                    token_usage = metrics_data.get("Effective Token Usage_effective_tokens_total") / metrics_data.get("Effective Token Usage_valid_response_total")
                    if token_usage is not None:
                        display_name = MODEL_NAME_MAP.get(model_name, model_name)
                        data_records.append({
                            'Model': model_name,  
                            'Display Name': display_name,  
                            'Puzzle Size': puzzle_size,
                            'Token Usage': token_usage
                        })
                except Exception as e:
                    print(f"Error processing {model_name} for {puzzle_size}: {e}")
                    continue


df = pd.DataFrame(data_records)

model_counts = df.groupby('Model')['Puzzle Size'].nunique()
models_with_both = model_counts[model_counts == 2].index.tolist()

if not models_with_both:
    print("No models have data for both 7x7 and 14x14 puzzles!")
else:
    print(f"Models with both puzzle sizes: {models_with_both}")

df = df[df['Model'].isin(models_with_both)]

df['Model'] = df['Display Name']
df = df.drop('Display Name', axis=1)

print(df)

color_sequence = ['#b7daf5', '#f9bdb6']  

title_name = "LVLM" if not args.text else "LLM"
fig = px.bar(
    df,
    x='Model',
    y='Token Usage',
    color='Puzzle Size',
    barmode='group',
    title=f'Token Usage Per {title_name} for 7x7 and 14x14 Puzzles',
    labels={'Token Usage': 'Token Usage'},
    template='plotly_white',
    color_discrete_sequence=color_sequence
)

fig.update_traces(
    width=0.4,  # Slightly increased bar width
    marker_line_width=1,  
    marker_line_color='black'  
)

# Get list of unique models after filtering
model_list = df['Model'].unique().tolist()

# Identify reasoning models (mapped names) for highlighting
reasoning_model_names = {MODEL_NAME_MAP.get(m, m) for m in REASONING_MODELS}

# Create tick text with red highlighting for reasoning models
ticktext = [
    f"<span style='color:red'>{model}</span>" if model in reasoning_model_names else model
    for model in model_list
]

fig.update_xaxes(
    showline=True, 
    linewidth=2, 
    linecolor='black', 
    zeroline=False, 
    showgrid=False,
    tickangle=30,  
    automargin=True,
    # Add tickmode and ticktext for highlighted reasoning models
    tickmode='array',
    tickvals=model_list,
    ticktext=ticktext
)

fig.update_yaxes(
    showline=True, 
    linewidth=2, 
    linecolor='black', 
    zeroline=False, 
    showgrid=True,
    gridcolor='lightgray'
)

fig.update_layout(
    # Fix the gap between bars in the same group
    bargap=0.2,         # Space between different model groups
    bargroupgap=0.0,    # Set to 0 to remove the gap between bars in the same group
    title=dict(
        text=f'Token Usage Per {title_name} for 7x7 and 14x14 Puzzles',
        font=dict(size=60, family='Palatino, serif', color='black', weight='bold'),
        x=0.5
    ),
    xaxis_title="",
    yaxis_title=dict(
        text="Token Usage",
        font=dict(size=55, family='Palatino, serif', color='black')
    ),
    xaxis=dict(
        tickfont=dict(size=50, family='Palatino, serif', color='black')
    ),
    yaxis=dict(
        tickfont=dict(size=50, family='Palatino, serif', color='black')
    ),
    legend=dict(
        title=dict(
            text='',
            font=dict(size=55, family='Palatino, serif', color='black')
        ),
        font=dict(size=50, family='Palatino, serif', color='black'),
        orientation='h',  
        yanchor='bottom',
        y=1.02,  
        xanchor='center',
        x=0.5,
        bgcolor='rgba(255,255,255,0)'
    ),
    plot_bgcolor='white',
    margin=dict(l=80, r=30, t=150, b=80),
    width=2000,  
    height=1000  
)

os.makedirs('../plots', exist_ok=True)
mode_suffix = '_text' if args.text else ''
save_name = f'../plots/token_usage_comparison{mode_suffix}.svg'
fig.write_image(save_name, format='svg', scale=1)
print(f"Bar chart saved to {save_name}")