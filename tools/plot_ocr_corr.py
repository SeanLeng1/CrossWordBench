import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from transformers import set_seed
import argparse

# Model name mapping for cleaner labels
MODEL_NAME_MAP = {
    'claude-3-7-sonnet-20250219': 'claude-3-7-sonnet',
    'claude-3-7-sonnet-20250219 (thinking)': 'claude-3-7-sonnet\U0001F4A1',
    'llava-onevision-qwen2-72b-ov-chat-hf': 'llava-onevision-qwen2-72b',
    'gemini-2.0-pro-exp-02-05': 'gemini-2.0-pro',
    'gpt-4o-2024-11-20': 'GPT-4o',
    'Pixtral-Large-Instruct-2411': 'Pixtral-Large-Instruct',
    'MiniCPM-V-2_6': 'MiniCPM-V-2.6',
}

# Forced placement for edge-case models.
# "left" means label appears to the left of the dot (xanchor="right", negative xshift),
# "right" means label appears to the right (xanchor="left", positive xshift),
# "top" means label appears above the dot (xanchor="center", yanchor="bottom", positive yshift),
# "bottom" means label appears below the dot (xanchor="center", yanchor="top", negative yshift).
MODEL_NAME_PLACEMENT = {
    'Qwen2.5-VL-3B-Instruct': 'right',
    'MiniCPM-V-2.6': 'right',
    'gemini-2.0-pro': 'bottom',
    'claude-3-7-sonnet': 'left',
    'claude-3-7-sonnet\U0001F4A1': 'top',
    'Aria': 'top',
    'llava-onevision-qwen2-72b': 'top',
    'NVLM-D-72B': 'bottom',
    'gemma-3-27b-it': 'right',
    'InternVL2_5-78B-MPO': 'right',
    'Qwen2.5-VL-72B-Instruct': 'right',
    'QVQ-72B-Preview': 'top',
    'Pixtral-Large-Instruct': 'left',
    'gemini-2.0-flash': 'right',
    'GPT-4o': 'bottom',
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Self-Reflection Effectiveness by comparing Word Coverage Rate for two rounds"
    )
    parser.add_argument('--results_root', type=str, default='../eval_results', help='Root directory for evaluation results')
    return parser.parse_args()

args = parse_args()

set_seed(42)
results_root = args.results_root
data = {
    'Model': [],
    'DisplayName': [],
    'ExtractionRate': [],
    'ImgCotRate': []
}
samples_data = {}

for model_name in os.listdir(results_root):
    model_dir = os.path.join(results_root, model_name)
    if not os.path.isdir(model_dir):
        continue
    model_dir = os.path.join(model_dir, 'english', '7x7')
    extraction_dir = os.path.join(model_dir, 'extraction')
    img_cot_dir = os.path.join(model_dir, 'img_cot')
    if os.path.isdir(extraction_dir) and os.path.isdir(img_cot_dir):
        extraction_json_path = os.path.join(extraction_dir, 'metrics', 'metrics.json')
        img_cot_json_path = os.path.join(img_cot_dir, 'metrics', 'metrics.json')
        extraction_sample_json_path = os.path.join(extraction_dir, 'metrics', 'raw_metrics.json')
        img_cot_sample_json_path = os.path.join(img_cot_dir, 'metrics', 'raw_metrics.json')
        if (os.path.exists(extraction_json_path) and os.path.exists(img_cot_json_path)
            and os.path.exists(extraction_sample_json_path) and os.path.exists(img_cot_sample_json_path)):
            try:
                with open(extraction_json_path, 'r', encoding='utf-8') as f:
                    extraction_data = json.load(f)
                with open(img_cot_json_path, 'r', encoding='utf-8') as f:
                    img_cot_data = json.load(f)
                with open(extraction_sample_json_path, 'r', encoding='utf-8') as f:
                    extraction_sample_data = json.load(f)
                with open(img_cot_sample_json_path, 'r', encoding='utf-8') as f:
                    img_cot_sample_data = json.load(f)

                print(f"Processing {model_name}")
                extraction_rate = extraction_data.get("Word Coverage Rate_all_mean")
                img_cot_rate = img_cot_data.get("Word Coverage Rate_all_mean")
                extraction_samples = extraction_sample_data.get("Word Coverage Rate_all")
                img_cot_samples = img_cot_sample_data.get("Word Coverage Rate_all")
                
                display_name = MODEL_NAME_MAP.get(model_name, model_name)
                
                data['Model'].append(model_name)
                data['DisplayName'].append(display_name)
                data['ExtractionRate'].append(extraction_rate)
                data['ImgCotRate'].append(img_cot_rate)
                
                samples_data[model_name] = {
                    'extraction_samples': extraction_samples,
                    'img_cot_samples': img_cot_samples
                }
            except Exception as e:
                print(f"Error processing {model_name}: {e}")
                continue

df = pd.DataFrame(data)
correlation, p_value = stats.pearsonr(df['ExtractionRate'], df['ImgCotRate'])
print(f"Direct Pearson correlation coefficient: {correlation:.6f}, p-value: {p_value:.7f}")

# Bootstrapping
n_boot = 10000
boot_corrs = []
for i in tqdm(range(n_boot), desc='Bootstrapping'):
    boot_ext_means = []
    boot_img_means = []
    for model in samples_data.keys():
        ext_samples = np.array(samples_data[model]['extraction_samples'])
        img_samples = np.array(samples_data[model]['img_cot_samples'])
        n_samples = len(ext_samples)
        indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
        boot_ext_means.append(np.mean(ext_samples[indices]))
        boot_img_means.append(np.mean(img_samples[indices]))
    boot_corr, _ = stats.pearsonr(boot_ext_means, boot_img_means)
    boot_corrs.append(boot_corr)

boot_corrs = np.array(boot_corrs)
ci_lower = np.percentile(boot_corrs, 2.5)
ci_upper = np.percentile(boot_corrs, 97.5)
print(f"Bootstrap 95% CI = [{ci_lower:.6f}, {ci_upper:.6f}]")

fig = go.Figure()

colors = [px.colors.qualitative.Alphabet[i % len(px.colors.qualitative.Alphabet)] for i in range(len(df))]
fig.add_trace(go.Scatter(
    x=df['ExtractionRate'],
    y=df['ImgCotRate'],
    mode='markers',
    marker=dict(
        size=20,
        color=colors,
        line=dict(width=2, color='DarkSlateGrey')
    ),
    showlegend=False
))

x_min = df['ExtractionRate'].min()
x_max = df['ExtractionRate'].max()
x_margin = (x_max - x_min) * 0.05  
x_range = [x_min - x_margin, x_max + x_margin]

for i, row in df.iterrows():
    x_val = row['ExtractionRate']
    y_val = row['ImgCotRate']
    display_name = row['DisplayName']
    forced_side = MODEL_NAME_PLACEMENT.get(display_name, "left")
    
    xshift = 0
    yshift = 0
    
    if forced_side == "left":
        xanchor = "right"  
        yanchor = "middle"
        xshift = 25  
    elif forced_side == "right":
        xanchor = "left"  
        yanchor = "middle"
        xshift = -30   
    elif forced_side == "top":
        xanchor = "center"  
        yanchor = "bottom"  
        yshift = 5   
    elif forced_side == "bottom":
        xanchor = "center" 
        yanchor = "top"     
        yshift = -5  
    else:
        xanchor = "right"
        yanchor = "middle"
        xshift = -25  

    fig.add_annotation(
        x=x_val,
        y=y_val,
        text=display_name,
        showarrow=False,
        font=dict(
            size=55,
            family='Palatino, serif',
            color='black',
            weight='bold'
        ),
        align="center",
        xanchor=xanchor,
        yanchor=yanchor,
        xshift=xshift,
        yshift=yshift
    )

X = df['ExtractionRate'].values.reshape(-1, 1)
y = df['ImgCotRate'].values
lr_model = LinearRegression()
lr_model.fit(X, y)
x_fit = np.linspace(df['ExtractionRate'].min(), df['ExtractionRate'].max(), 100).reshape(-1, 1)
y_fit = lr_model.predict(x_fit)
print(f"Linear regression: y = {lr_model.coef_[0]:.6f}x + {lr_model.intercept_:.6f}")

fit_equation = f"y = {lr_model.coef_[0]:.4f}x + {lr_model.intercept_:.4f}"
fig.add_trace(
    go.Scatter(
        x=x_fit.flatten(),
        y=y_fit,
        mode='lines',
        name=f"Linear Fit ({fit_equation})",
        line=dict(color='red', dash='dash', width=3),
        showlegend=True
    )
)

annotation_text = (
    f"<b>Direct Corr (r) = {correlation:.4f}<br>p-value = {p_value:.7f}</b><br>"
    f"<b>Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]</b>"
)
fig.add_annotation(
    text=annotation_text,
    xref='paper',
    yref='paper',
    x=0.015,
    y=0.96,
    showarrow=False,
    borderwidth=1,
    bordercolor='black',
    bgcolor='rgba(255,255,255,0.7)',
    font=dict(size=55, family='Palatino, serif', color='black', weight='bold')
)

fig.update_layout(
    title='Grid Parsing and Performance Correlation<br>(Metrics: Word Coverage Rate)',
    xaxis_title='Grid Parsing WCR',
    yaxis_title='7x7 WCR',
    template='plotly_white',
    title_font=dict(size=55, family='Palatino, serif', color='black', weight='bold'),
    title_x=0.5,
    xaxis=dict(
        domain=[0, 1],
        range=x_range,
        title_font=dict(size=48, family='Palatino, serif', color='black', weight='bold'),
        tickfont=dict(size=45, family='Palatino, serif', color='black'),
        showline=True,
        linewidth=3,
        linecolor='black',
        zeroline=False,
        showgrid=False,
    ),
    yaxis=dict(
        title_font=dict(size=48, family='Palatino, serif', color='black', weight='bold'),
        tickfont=dict(size=45, family='Palatino, serif', color='black'),
        showline=True,
        linewidth=3,
        linecolor='black',
        zeroline=False,
        showgrid=False,
    ),
    legend=dict(
        x=0.5,
        y=0.2,
        xanchor='left',
        yanchor='top',
        title="",
        font=dict(size=55, family='Palatino, serif', color='black', weight='bold'),
        borderwidth=1,
        bordercolor='black',
        orientation="h",
        entrywidth=800,
    ),
    margin=dict(l=80, r=80, t=100, b=120)
)

os.makedirs('../plots', exist_ok=True)
fig.write_image('../plots/ocr-performance-correlation.svg', format='svg', scale=1, width=1900, height=1200)
