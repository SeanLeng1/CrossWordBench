import os
import json
import plotly.graph_objects as go
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create radar or bar plot for reasoning effort comparison"
    )
    parser.add_argument('--results_root', type=str, default='../eval_results', 
                        help='Root directory for evaluation results')
    parser.add_argument('--puzzle_size', type=str, default='7x7',
                        help='Puzzle size to analyze')
    parser.add_argument('--plot_type', type=str, default='radar', choices=['radar', 'bar'],
                        help='Type of plot to generate: "radar" or "bar"')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Define reasoning effort mapping for models
    MODELS = {
        'o3-mini': 'High',
        'o3-mini (medium)': 'Medium',
        'o3-mini (low)': 'Low',
    }
    
    # Define metrics to collect: WCR, LCR, and ICR
    metrics_config = [
        {"key": "Word Coverage Rate_all_mean", "display": "WCR"},
        {"key": "Letter Coverage Rate_all_mean", "display": "LCR"},
        {"key": "Intersection Rate_mean", "display": "ICR"}
    ]
    
    condition = 'text_cot'
    puzzle_size = args.puzzle_size
    
    # Containers for collected data
    values = []          # metric values (one list per model)
    tokens = []          # average token usage per model
    reasoning_efforts = []  # effort levels for labels
    
    for model_name, effort_level in MODELS.items():
        model_path = os.path.join(args.results_root, model_name)
        if not os.path.isdir(model_path):
            print(f"Warning: Model directory {model_path} not found")
            continue
            
        model_dir = os.path.join(model_path, 'english', puzzle_size)
        condition_dir = os.path.join(model_dir, condition)
        if not os.path.isdir(condition_dir):
            print(f"Warning: Condition directory {condition_dir} not found")
            continue
            
        metrics_json_path = os.path.join(condition_dir, 'metrics', 'metrics.json')
        if not os.path.exists(metrics_json_path):
            print(f"Warning: Metrics file {metrics_json_path} not found")
            continue
            
        try:
            with open(metrics_json_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            
            model_values = []
            for metric in metrics_config:
                metric_value = metrics_data.get(metric["key"])
                if metric_value is None:
                    print(f"Warning: Metric {metric['key']} not found for {model_name}")
                    continue
                model_values.append(metric_value)
            
            tokens_total = metrics_data.get("Effective Token Usage_tokens_total")
            tokens_count = metrics_data.get("Effective Token Usage_tokens_count")
            if tokens_total is None or tokens_count is None:
                print(f"Warning: Token usage metrics not found for {model_name}")
                continue
            avg_tokens = tokens_total / tokens_count
            
            values.append(model_values)
            tokens.append(avg_tokens)
            reasoning_efforts.append(effort_level)
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    
    if not values:
        print("Error: No valid data collected for plotting")
        return
    
    categories = [metric["display"] for metric in metrics_config]
    
    # Define colors for the different reasoning efforts
    colors = {
        'High': '#f9bdb6',    
        'Medium': '#b7daf5',  
        'Low': '#74a892'     
    }
    
    # Create the appropriate plot based on the plot_type argument
    if args.plot_type == "radar":
        # Calculate min and max values for scale (with a 20% buffer on max)
        all_values = [val for sublist in values for val in sublist]
        min_val = max(0, min(all_values) * 0.7)
        max_val = max(all_values) * 1.2

        fig = go.Figure()
        # Add traces for each reasoning effort level in a radar plot
        for i, effort in enumerate(reasoning_efforts):
            # Close the polygon by appending the first value at the end
            values_to_plot = values[i] + [values[i][0]]
            theta = categories + [categories[0]]

            # Format token count with commas for display
            token_display = "{:,}".format(int(tokens[i]))
            hex_color = colors[effort]

            fig.add_trace(go.Scatterpolar(
                r=values_to_plot,
                theta=theta,
                mode='lines',
                name=f"{effort} Effort ({token_display} tokens)",
                line=dict(color=hex_color, width=2)
            ))

        # Update layout for radar plot
        fig.update_layout(
            title=dict(
                text="o3-mini Reasoning Effort vs. Puzzle-Solving",
                font=dict(size=25, family='Palatino, serif', color='black', weight='bold'),
                x=0.5,
                y=0.9,
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[min_val, max_val],
                    tickfont=dict(size=15),
                    linecolor='black',
                    gridcolor='lightgray'
                ),
                angularaxis=dict(
                    tickfont=dict(size=18, family='Palatino, serif'),
                    linecolor='black',
                    gridcolor='lightgray',
                    direction="clockwise"
                ),
                bgcolor='rgba(0,0,0,0)'  # transparent background
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.4,
                xanchor="center",
                x=0.55,
                font=dict(size=18, family='Palatino, serif', color='black')
            ),
            margin=dict(l=0, r=0, t=105, b=0),
            font=dict(family='Palatino, serif')
        )

        # Save radar plot
        os.makedirs('../plots', exist_ok=True)
        output_path = '../plots/reasoning_effort_radar.svg'
        fig.write_image(output_path, format='svg', scale=1, width=400, height=400)
        print(f"Radar plot successfully saved to {os.path.abspath(output_path)}")

    elif args.plot_type == "bar":
        fig = go.Figure()
        for i, effort in enumerate(reasoning_efforts):
            token_display = "{:,}".format(int(tokens[i]))
            fig.add_trace(go.Bar(
                x=categories,
                y=values[i],
                name=f"{effort} Effort ({token_display} tokens)",
                marker_color=colors[effort]
            ))

        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            title=dict(
                text="o3-mini Reasoning Effort vs. Puzzle-Solving",
                font=dict(size=25, family='Palatino, serif', color='black', weight='bold'),
                x=0.5,
                y=0.99,
            ),
            xaxis=dict(
                title='',
                tickfont=dict(size=22, family='Palatino, serif', color='black'),
                showline=True,           
                #linewidth=1,            
                linecolor='black',                 
            ),
            yaxis=dict(
                title='',
                tickfont=dict(size=22, family='Palatino, serif', color='black'),
                showline=True,           
                #linewidth=1,            
                linecolor='black',                
            ),
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                y=0.75,
                xanchor="center",
                x=0.5,
                font=dict(size=18, family='Palatino, serif', color='black')
            ),
            margin=dict(l=0, r=0, t=60, b=0),
            font=dict(family='Palatino, serif')
        )
        fig.update_traces(
            marker_line_width=1,  
            marker_line_color='black'  
        )

        # Save bar plot
        os.makedirs('../plots', exist_ok=True)
        output_path = '../plots/reasoning_effort_bar.svg'
        fig.write_image(output_path, format='svg', scale=1, width=600, height=300)
        fig.write_image('../plots/reasoning_effort_bar.pdf', format='pdf', scale=1, width=600, height=300)
        print(f"Bar plot successfully saved to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
