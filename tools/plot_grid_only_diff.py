import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import argparse

MODEL_NAME_MAP = {
    'gpt-4o-2024-11-20': 'GPT-4o',
    'claude-3-7-sonnet-20250219': 'claude-3-7-sonnet',
    'Deepseek-R1-Distill-Llama-70B': 'R1-Distill-Llama-70B',
    'Pixtral-Large-Instruct-2411': 'Pixtral-Large-Instruct',
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Word Coverage Rate between img_grid_cot_grid_only and img_cot"
    )
    parser.add_argument('--results_root', type=str, default='../eval_results', help='Root directory for evaluation results')
    return parser.parse_args()

args = parse_args()
results_root = args.results_root

settings = {
    'grid_cot': 'img_cot_grid_only',
    'regular_cot': 'img_cot',
}

legend_mapping = {
    'grid_cot': "Image Grid + Text Clue",
    'regular_cot': "Image Only"
}

puzzle_size = '7x7'

def collect_data(prompt_grid, prompt_regular):
    data_records = []
    
    for model_name in os.listdir(results_root):
        model_path = os.path.join(results_root, model_name)
        if not os.path.isdir(model_path):
            continue

        model_dir = os.path.join(model_path, 'english', puzzle_size)
        metrics = {}
        for prompt_type, prompt_name in [('grid_cot', prompt_grid), ('regular_cot', prompt_regular)]:
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
                        print(f"Error processing {model_name} for {prompt_name}: {e}")
        
        if 'grid_cot' in metrics and 'regular_cot' in metrics:
            diff = metrics['grid_cot'] - metrics['regular_cot']
            
            data_records.append({
                'Model': MODEL_NAME_MAP.get(model_name, model_name),
                'grid_cot': metrics['grid_cot'],
                'regular_cot': metrics['regular_cot'],
                'difference': diff
            })
    
    return pd.DataFrame(data_records)

def sort_by_difference(df, ascending=True):
    """Sort the dataframe by difference"""
    return df.sort_values(by='difference', ascending=ascending)


df = collect_data(settings['grid_cot'], settings['regular_cot'])

df = sort_by_difference(df, ascending=True)

print("Comparison data (sorted by difference):")
print(df)


df_melted = df.melt(id_vars=['Model', 'difference'], value_vars=['grid_cot', 'regular_cot'],
                  var_name='Type', value_name='Word Coverage Rate')

df_melted['Type'] = df_melted['Type'].map(legend_mapping)

color_sequence = ['#b7daf5', '#f9bdb6']

fig = px.bar(
    df_melted,
    x='Model',
    y='Word Coverage Rate',
    color='Type',
    barmode='group',
    title='Image Grid + Text Clue vs. Image Only',
    labels={'Word Coverage Rate': 'Word Coverage Rate'},
    template='plotly_white',
    color_discrete_sequence=color_sequence
)

fig.update_traces(
    width=0.4,  
    marker_line_width=1,  
    marker_line_color='black'  
)

for _, row in df.iterrows():
    model = row['Model']
    diff = row['difference']
    max_val = max(row['grid_cot'], row['regular_cot'])

    diff_text = f"{diff:+.3f}"
    
    fig.add_annotation(
        x=model,
        y=max_val + 0.05,  
        text=diff_text,
        showarrow=False,
        font=dict(size=50, color="black", family='Palatino, serif', weight='bold')
    )

fig.update_xaxes(
    showline=True, 
    linewidth=2, 
    linecolor='black', 
    zeroline=False, 
    showgrid=False,
    tickangle=15,  
    automargin=True
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
    bargap=0.2,         
    bargroupgap=0.0,    
    title=dict(
        text='Image Grid + Text Clue vs. Image Only',
        font=dict(size=60, family='Palatino, serif', color='black', weight='bold'),
        x=0.5
    ),
    xaxis_title="",
    yaxis_title=dict(
        text="Word Coverage Rate",
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
        y=1.0,  
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
save_name = '../plots/grid_vs_regular_cot_comparison.svg'
fig.write_image(save_name, format='svg', scale=1)

print(f"Plot saved as {save_name}")