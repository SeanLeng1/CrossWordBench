import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse

# Define model name mapping
MODEL_NAME_MAP = {
    'gpt-4o-2024-11-20': 'gpt-4o',
    'claude-3-7-sonnet-20250219 (nonthinking)': 'claude-3-7-sonnet',
    'DeepSeek-R1-Distill-Llama-70B': 'R1-Distill-Llama-70B',
    'deepseek-reasoner': 'Deepseek-R1',
}
REASONING_MODELS = {
    'deepseek-reasoner',
    'QwQ-32B',
    'DeepSeek-R1-Distill-Llama-70B',
}
EXCLUDE_MODELS = {
    'reka-flash-3'
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Word Coverage Rate for two prompting settings for each model (both text and image)"
    )
    parser.add_argument('--results_root', type=str, default='../eval_results', help='Root directory for evaluation results')
    return parser.parse_args()

args = parse_args()
results_root = args.results_root

text_settings = {
    'cot': 'text_cot',
    'shot': 'text_shot',
    'title': 'LLMs'
}

img_settings = {
    'cot': 'img_cot',
    'shot': 'img_shot',
    'title': 'VLMs'
}

legend_mapping = {
    'cot': "zero-shot cot",
    'shot': "few-shot progressive"
}

puzzle_size = '7x7'

def collect_data(prompt_cot, prompt_shot):
    data_records = []
    
    for model_name in os.listdir(results_root):
        original_model_name = model_name  
        if original_model_name in EXCLUDE_MODELS:
            continue
        model_path = os.path.join(results_root, original_model_name)
        if not os.path.isdir(model_path):
            continue

        model_dir = os.path.join(model_path, 'english', puzzle_size)
        metrics = {}
        for prompt_type, prompt_name in [('cot', prompt_cot), ('shot', prompt_shot)]:
            prompt_dir = os.path.join(model_dir, prompt_name)
            if os.path.isdir(prompt_dir):
                metrics_json_path = os.path.join(prompt_dir, 'metrics', 'metrics.json')
                if os.path.exists(metrics_json_path):
                    try:
                        with open(metrics_json_path, 'r', encoding='utf-8') as f:
                            metrics_data = json.load(f)
                        word_cov = metrics_data.get("Word Coverage Rate_all_mean")
                        if word_cov is not None:
                            metrics[prompt_type] = word_cov
                    except Exception as e:
                        print(f"Error processing {original_model_name} for {prompt_name}: {e}")
        
        if 'cot' in metrics and 'shot' in metrics:
            display_name = MODEL_NAME_MAP.get(original_model_name, original_model_name)
            data_records.append({
                'Model': display_name,
                'cot': metrics['cot'],
                'shot': metrics['shot']
            })
    
    return pd.DataFrame(data_records)

def sort_by_improvement(df, ascending=True):
    df['pct_improvement'] = ((df['shot'] - df['cot']) / df['cot']) * 100
    return df.sort_values(by='pct_improvement', ascending=ascending)

text_df = collect_data(text_settings['cot'], text_settings['shot'])
img_df = collect_data(img_settings['cot'], img_settings['shot'])

print("Text models with original and mapped names:")
for model_name in os.listdir(results_root):
    if os.path.isdir(os.path.join(results_root, model_name)):
        mapped_name = MODEL_NAME_MAP.get(model_name, model_name)
        print(f"Original: {model_name} -> Mapped: {mapped_name}")

# Sort both dataframes by percentage improvement (ascending=True means lower to higher)
text_df = sort_by_improvement(text_df, ascending=True)
img_df = sort_by_improvement(img_df, ascending=True)

print("\nText prompting data (sorted by improvement):")
print(text_df)
print("\nImage prompting data (sorted by improvement):")
print(img_df)

def create_traces(df, prompt_type):
    df_melted = df.melt(id_vars=['Model', 'pct_improvement'], value_vars=['cot', 'shot'],
                      var_name='Prompt Type', value_name='Word Coverage Rate')
    
    df_melted['Prompt Type'] = df_melted['Prompt Type'].map(legend_mapping)
    
    traces = []
    
    prompt_types = df_melted['Prompt Type'].unique()
    
    color_map = {
        'zero-shot cot': '#EF553B',
        'few-shot progressive': '#636EFA',
    }
    
    for pt in prompt_types:
        df_pt = df_melted[df_melted['Prompt Type'] == pt]
        trace = go.Bar(
            x=df_pt['Model'],
            y=df_pt['Word Coverage Rate'],
            name=pt,
            marker_color=color_map[pt],
            opacity=0.7,
            showlegend=(prompt_type == 'text')
        )
        traces.append(trace)
    
    annotations = []
    for _, row in df.iterrows():
        model = row['Model']
        val_cot = row['cot']
        val_shot = row['shot']
        
        abs_diff = val_shot - val_cot
        
        if val_cot > 0:
            pct_improvement = (abs_diff / val_cot) * 100
            pct_text = f"{pct_improvement:+.1f}%"
        else:
            pct_text = "N/A"
        
        # Place annotation above the higher of the two bars
        y_pos = max(val_cot, val_shot) + 0.03  
        annotations.append(
            dict(
                x=model,
                y=y_pos,
                text=pct_text,
                showarrow=False,
                font=dict(size=32, color="black")
            )
        )
    
    return traces, annotations

fig = make_subplots(
    rows=1, 
    cols=2,
    subplot_titles=(text_settings['title'], img_settings['title']),
    horizontal_spacing=0.1,
    shared_yaxes=True
)

# Update subplot title font styles
for i in range(len(fig.layout.annotations)):
    fig.layout.annotations[i].font.size = 32
    fig.layout.annotations[i].font.family = 'Palatino, serif'
    fig.layout.annotations[i].font.color = 'black'
    fig.layout.annotations[i].font.weight = 'bold'

text_traces, text_annotations = create_traces(text_df, 'text')
for trace in text_traces:
    fig.add_trace(trace, row=1, col=1)

img_traces, img_annotations = create_traces(img_df, 'image')
for trace in img_traces:
    fig.add_trace(trace, row=1, col=2)

for annotation in text_annotations:
    annotation['xref'] = 'x'
    annotation['yref'] = 'y'
    fig.add_annotation(annotation)

for annotation in img_annotations:
    annotation['xref'] = 'x2'
    annotation['yref'] = 'y2'
    fig.add_annotation(annotation)

for i in range(1, 3):
    fig.update_xaxes(
        showline=True,
        linewidth=3,
        linecolor='black',
        zeroline=False,
        showgrid=False,
        tickfont=dict(size=38, family='Palatino, serif', color='black'),
        row=1, col=i
    )

fig.update_yaxes(
    title_text="Word Coverage Rate",
    showline=True,
    linewidth=3,
    linecolor='black',
    zeroline=False,
    showgrid=True,
    tickfont=dict(size=38, family='Palatino, serif', color='black'),
    row=1, col=1
)

fig.update_yaxes(
    showline=False,
    linewidth=3,
    linecolor='black',
    zeroline=False,
    showgrid=True,  
    tickfont=dict(size=38, family='Palatino, serif', color='black'),
    row=1, col=2
)

fig.update_layout(
    barmode='overlay',
    title_text="Word Coverage Rate (WCR) Improvement with Few-Shot Progressive Examples",
    title_font=dict(size=42, family='Palatino, serif', color='black', weight='bold'),
    title_x=0.5,
    xaxis_title_font=dict(size=38, family='Palatino, serif', color='black', weight='bold'),
    yaxis_title_font=dict(size=38, family='Palatino, serif', color='black', weight='bold'),
    legend=dict(
        title='Prompt Type',
        font=dict(size=32, family='Palatino, serif', color='black', weight='bold'),
        # Position the legend at the top right corner of the second subplot:
        x=0.7,
        y=1.10,
        xanchor='right',
        yanchor='top',
        orientation='v',
        bgcolor='rgba(255,255,255,0)'
    ),
    margin=dict(l=80, r=80, t=150, b=100),
    width=1800,
    height=900
)

# Highlight reasoning models in red
reasoning_model_names = {MODEL_NAME_MAP.get(m, m) for m in REASONING_MODELS}

# Update x-axis tick labels for text subplot
tickvals_text = text_df['Model'].tolist()
ticktext_text = [
    f"<span style='color:red'>{model}</span>" if model in reasoning_model_names else model
    for model in tickvals_text
]
fig.update_xaxes(
    tickmode='array',
    tickvals=tickvals_text,
    ticktext=ticktext_text,
    row=1, col=1
)

# Update x-axis tick labels for image subplot
tickvals_img = img_df['Model'].tolist()
ticktext_img = [
    f"<span style='color:red'>{model}</span>" if model in reasoning_model_names else model
    for model in tickvals_img
]
fig.update_xaxes(
    tickmode='array',
    tickvals=tickvals_img,
    ticktext=ticktext_img,
    row=1, col=2
)

os.makedirs('../plots', exist_ok=True)
save_name = '../plots/shot_difference_combined.svg'
fig.write_image(save_name, format='svg', scale=1)

print(f"Plot saved as {save_name}")
