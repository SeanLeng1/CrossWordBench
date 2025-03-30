import json
from tabulate import tabulate
from typing import Dict, Any, Optional
from pathlib import Path
import argparse

def process_metrics_file(json_path: str) -> str:
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    parts = []
    stats_tables = []
    count_tables = []
    
    # get all base metric names (excluding statistical suffixes)
    metric_names = set()
    for key in data.keys():
        if key.endswith(('_mean', '_total')):
            base_metric = key.rsplit('_', 1)[0]
            metric_names.add(base_metric)
    
    stats_data = []
    for metric in sorted(metric_names):
        # skip if this is a count metric
        if f"{metric}_mean" not in data:
            continue  
        try:
            mean = data[f"{metric}_mean"]
            margin = abs(data[f"{metric}_ci_upper"] - data[f"{metric}_ci_lower"]) / 2
            stats_data.append([
                metric,
                f"{mean:.3f} ± {margin:.3f}",
                f"[{data[f'{metric}_ci_lower']:.3f}, {data[f'{metric}_ci_upper']:.3f}]",
                data[f"{metric}_median"],
                data[f"{metric}_min"],
                data[f"{metric}_max"],
                data[f"{metric}_count"]
            ])
        except (KeyError, ValueError):
            continue
    
    if stats_data:
        table = tabulate(
            stats_data,
            headers=['Metric', 'Mean ± Margin', '95% CI', 'Median', 'Min', 'Max', 'N'],
            tablefmt='fancy_grid',
            floatfmt='.3f'
        )
        stats_tables.append(table)
    
    count_data = []
    for metric in sorted(metric_names):
        if f"{metric}_total" in data and f"{metric}_mean" not in data:
            try:
                total = data[f"{metric}_total"]
                count = data[f"{metric}_count"]
                if count > 0:
                    count_data.append([metric, total, count])
            except (KeyError, ValueError):
                continue
    
    if count_data:
        table = tabulate(
            count_data,
            headers=['Metric', 'Total', 'Puzzles'],
            tablefmt='fancy_grid',
            floatfmt='.3f'
        )
        count_tables.append(table)
    
    if stats_tables:
        parts.extend([
            "=" * 80,
            "STATISTICS SUMMARY",
            "=" * 80,
            *stats_tables
        ])
    
    if count_tables:
        parts.extend([
            "\n" + "=" * 80,
            "ERROR COUNTS",
            "=" * 80,
            *count_tables
        ])
    
    additional_stats = [
        ['Model', data.get('model', 'N/A')],
        ['Parsing Model', data.get('parsing_model', 'N/A')],
        ['Temperature', data.get('temperature', 'N/A')],
        ['Top P', data.get('top_p', 'N/A')]
    ]
    
    usage = data.get('usage', {})
    if usage:
        additional_stats.extend([
            ['Prompt Tokens', usage.get('prompt_tokens', 'N/A')],
            ['Completion Tokens', usage.get('completion_tokens', 'N/A')],
            ['Total Tokens', usage.get('total_tokens', 'N/A')]
        ])
    
    if additional_stats:
        parts.extend([
            "\n" + "=" * 80,
            "ADDITIONAL INFORMATION",
            "=" * 80,
            tabulate(additional_stats, headers=['Metric', 'Value'], tablefmt='fancy_grid')
        ])
    
    return "\n\n".join(parts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    with open('results.txt', 'w') as f:
        pass
    
    for folder in output_dir.iterdir():
        if folder.is_dir():
            for subfolder in folder.iterdir():
                for difficulty_folder in subfolder.iterdir():
                    if difficulty_folder.is_dir():
                        for template_folder in difficulty_folder.iterdir():
                            if template_folder.is_dir():
                                json_path = template_folder / 'metrics' / 'metrics.json'
                                if json_path.exists():
                                    header = f"\n{'=' * 100}\n{json_path}\n{'=' * 100}"
                                    print(header)
                                    formatted_output = process_metrics_file(str(json_path))
                                    print(formatted_output)
                                    with open('results.txt', 'a') as f:
                                        f.write(header + '\n')
                                        f.write(formatted_output)
                                        f.write('\n\n')
                                else:
                                    json_path = template_folder / 'all' / 'metrics' / 'metrics.json'
                                    if json_path.exists():
                                        header = f"\n{'=' * 100}\n{json_path}\n{'=' * 100}"
                                        print(header)
                                        formatted_output = process_metrics_file(str(json_path))
                                        print(formatted_output)
                                        with open('results.txt', 'a') as f:
                                            f.write(header + '\n')
                                            f.write(formatted_output)
                                            f.write('\n\n')

if __name__ == '__main__':
    main()