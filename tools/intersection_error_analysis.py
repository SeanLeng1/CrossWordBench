import os
import json
from typing import List, Dict, Any
import argparse
from collections import defaultdict
import numpy as np
from scipy import stats
import pandas as pd
import plotly.express as px

# List of reasoning models to be analyzed.
REASONING_MODELS = {
    'o3-mini',
    'deepseek-reasoner',
    'QwQ-32B',
    'DeepSeek-R1-Distill-Llama-70B',
    'QVQ-72B-Preview',
    #'Open-Reasoner-Zero-32B',
    'claude-3-7-sonnet-20250219 (thinking)',
}

EXCLUDED_MODELS = {
    'Open-Reasoner-Zero-32B',
    'o3-mini (low)',
    'o3-mini (medium)',
    'Qwen2.5-VL-72B-Instruct',
    'InternVL2_5-78B-MPO',
}


class CrosswordGridAnalyzer:
    @staticmethod
    def get_direction(reference_answer: List[Dict[str, Any]], word: str):
        # Return the direction (key) from the reference_answer that corresponds to the given word.
        for entry in reference_answer:
            if entry["answer"] == word:
                return entry["direction"]
        return None

    @staticmethod
    def find_intersections(wordlist: List[List[Any]], grid: List[List[str]], reference_answer: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Find all intersections between across and down words.
        """
        across_words = []
        down_words = []
        for entry in wordlist:
            word, clue, start_row, start_col, orientation = entry
            if orientation == 0:  # across
                across_words.append({
                    "word": word,
                    "clue": clue,
                    "row": start_row,
                    "col": start_col,
                })
            else:  # down
                down_words.append({
                    "word": word,
                    "clue": clue,
                    "row": start_row,
                    "col": start_col,
                })

        intersections = []
        # For each across word, compute its column span and check for intersections with each down word.
        for across in across_words:
            a_word = across["word"]
            a_row = across["row"]
            a_col_start = across["col"]
            a_col_end = a_col_start + len(a_word) - 1
            for down in down_words:
                d_word = down["word"]
                d_row_start = down["row"]
                d_col = down["col"]
                d_row_end = d_row_start + len(d_word) - 1

                if (d_col >= a_col_start and d_col <= a_col_end and
                    a_row >= d_row_start and a_row <= d_row_end):

                    across_index = d_col - a_col_start
                    down_index = a_row - d_row_start
                    grid_letter = grid[a_row][d_col]

                    intersections.append({
                        "across_word": a_word,
                        "down_word": d_word,
                        "across_index": across_index,
                        "down_index": down_index,
                        "row": a_row,
                        "col": d_col,
                        "grid_letter": grid_letter,
                        "across_key": CrosswordGridAnalyzer.get_direction(reference_answer, a_word),
                        "down_key": CrosswordGridAnalyzer.get_direction(reference_answer, d_word),
                    })

        return intersections


def count_word_crossings(intersections):
    """Count how many crossing letters each word (by direction) has."""
    word_crossings = defaultdict(int)
    for inter in intersections:
        across_key = inter["across_key"]
        down_key = inter["down_key"]
        if across_key:
            word_crossings[across_key] += 1
        if down_key:
            word_crossings[down_key] += 1
    return dict(word_crossings)


def analyze_puzzle_structure(puzzle_state, reference_answer):
    """
    Analyze the puzzle structure by:
      - Finding intersections between words.
      - Counting the number of crossings per word (by its direction).
    """
    intersections = CrosswordGridAnalyzer.find_intersections(
        puzzle_state['wordlist'],
        puzzle_state['grid'],
        reference_answer
    )
    word_crossings = count_word_crossings(intersections)

    # Ensure every word (from reference_answer) appears in the crossing count, assigning 0 if needed.
    all_directions = {entry["direction"] for entry in reference_answer if "direction" in entry}
    for direction in all_directions:
        if direction not in word_crossings:
            word_crossings[direction] = 0

    words_by_crossings = defaultdict(list)
    for word_key, num_crossings in word_crossings.items():
        words_by_crossings[num_crossings].append(word_key)
    crossing_counts = {num: len(words) for num, words in words_by_crossings.items()}

    return crossing_counts, intersections, word_crossings


def calculate_correctness_by_crossings(model_answer, reference_answer, word_crossings):
    """
    Calculate correctness rate grouped by number of crossing letters.
    """
    if not reference_answer or not model_answer:
        return {}, 0

    ref_dict = {entry["direction"]: entry["answer"] for entry in reference_answer}
    crossing_groups = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total_words = 0

    for direction, ref_word in ref_dict.items():
        num_crossings = word_crossings.get(direction, 0)
        if direction not in model_answer:
            crossing_groups[num_crossings]["total"] += 1
            total_words += 1
            continue
        model_word = model_answer[direction].get("answer", "")
        is_correct = model_word.upper() == ref_word.upper()
        if is_correct:
            crossing_groups[num_crossings]["correct"] += 1
            total_correct += 1
        crossing_groups[num_crossings]["total"] += 1
        total_words += 1

    crossing_accuracy = {num: stats["correct"] / stats["total"]
                         for num, stats in crossing_groups.items() if stats["total"] > 0}
    overall_accuracy = total_correct / total_words if total_words > 0 else 0
    return crossing_accuracy, overall_accuracy


def analyze_models(base_dir, required_puzzle_count=100, only_reasoning=False):
    """
    Analyze models by:
      - Validating required puzzle files.
      - Computing structure information (crossings per word).
      - Calculating accuracy per crossing category.
    """
    valid_model_dirs = {}
    all_model_dirs = os.listdir(base_dir)
    if only_reasoning:
        model_list = REASONING_MODELS
    else:
        model_list = set(model_dir for model_dir in all_model_dirs if model_dir not in REASONING_MODELS and model_dir not in EXCLUDED_MODELS)
    print('Analyzing models:', len(model_list))
    for model_name in model_list:
        print(f"Checking model {model_name}...")
        model_dir = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_dir):
            print(f"  Directory not found for {model_name}")
            continue

        grid_dir = os.path.join(model_dir, "english/7x7")
        if not os.path.exists(grid_dir):
            print(f"  No english/7x7 directory for {model_name}")
            continue

        cot_type = "text_cot"
        cot_dir = os.path.join(grid_dir, cot_type)
        if not os.path.exists(cot_dir):
            print(f"  No {cot_type} directory for {model_name}")
            continue

        all_puzzle_files = [f for f in os.listdir(cot_dir) if f.endswith(".json")]
        if len(all_puzzle_files) == 0:
            print(f"  No JSON files found for {model_name}")
            continue

        all_puzzle_files.sort()
        puzzle_files = all_puzzle_files[:required_puzzle_count]
        if len(puzzle_files) >= required_puzzle_count:
            valid_model_dirs[model_name] = {"dir": cot_dir, "puzzles": puzzle_files}
            print(f"  Found {len(puzzle_files)} puzzles for {model_name}")
        else:
            print(f"  Skipping model {model_name}: Only has {len(puzzle_files)} puzzles, {required_puzzle_count} required")

    model_type = "reasoning" if only_reasoning else "all"
    print(f"Found {len(valid_model_dirs)} {model_type} models with at least {required_puzzle_count} puzzles")

    puzzle_structures = {}
    total_words_by_crossings = defaultdict(int)
    all_puzzles = set()
    for model_name, model_info in valid_model_dirs.items():
        for filename in model_info["puzzles"]:
            all_puzzles.add(filename)
            if filename in puzzle_structures:
                continue
            file_path = os.path.join(model_info["dir"], filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                puzzle_state = data.get("puzzle_state", {})
                reference_answer = data.get("reference_answer", [])
                if puzzle_state and reference_answer:
                    crossing_counts, intersections, word_crossings = analyze_puzzle_structure(
                        puzzle_state, reference_answer
                    )
                    puzzle_structures[filename] = {
                        "crossing_counts": crossing_counts,
                        "word_crossings": word_crossings,
                        "reference_answer": reference_answer
                    }
                    for num_crossings, count in crossing_counts.items():
                        total_words_by_crossings[num_crossings] += count
            except Exception as e:
                print(f"Error analyzing puzzle structure for {file_path}: {e}")

    results = {}
    for model_name, model_info in valid_model_dirs.items():
        model_dir = model_info["dir"]
        model_results = {
            "overall_accuracy": [],
            "crossing_accuracy": defaultdict(list),
            "processed_puzzles": 0,
            "puzzle_results": {}
        }
        for filename in model_info["puzzles"]:
            if filename not in puzzle_structures:
                print(f"Warning: No structure info for {filename}, skipping for model {model_name}")
                continue
            file_path = os.path.join(model_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                model_answer = data.get("model_answer", {})
                if model_answer:
                    puzzle_info = puzzle_structures[filename]
                    crossing_acc, overall_acc = calculate_correctness_by_crossings(
                        model_answer,
                        puzzle_info["reference_answer"],
                        puzzle_info["word_crossings"]
                    )
                    model_results["overall_accuracy"].append(overall_acc)
                    model_results["processed_puzzles"] += 1
                    model_results["puzzle_results"][filename] = {
                        "overall": overall_acc,
                        "by_crossings": crossing_acc
                    }
                    for num_crossings, accuracy in crossing_acc.items():
                        model_results["crossing_accuracy"][num_crossings].append(accuracy)
            except Exception as e:
                print(f"Error processing {file_path} for model {model_name}: {e}")

        if model_results["processed_puzzles"] == required_puzzle_count:
            avg_overall = (sum(model_results["overall_accuracy"]) /
                           len(model_results["overall_accuracy"]) if model_results["overall_accuracy"] else 0)
            avg_by_crossings = {}
            for num_crossings, accuracies in model_results["crossing_accuracy"].items():
                avg_by_crossings[num_crossings] = sum(accuracies) / len(accuracies) if accuracies else 0
            results[model_name] = {
                "file_count": model_results["processed_puzzles"],
                "avg_overall_accuracy": avg_overall,
                "avg_by_crossings": avg_by_crossings
            }
        else:
            print(f"Skipping model {model_name} in results: Only processed {model_results['processed_puzzles']} of {required_puzzle_count} required puzzles")

    model_count = len(results)
    if model_count > 0:
        for num_crossings in total_words_by_crossings:
            total_words_by_crossings[num_crossings] = total_words_by_crossings[num_crossings] / model_count

    return results, total_words_by_crossings, len(all_puzzles)


def analyze_accuracy_by_crossings(results, total_words_by_crossings):
    """
    Analyze accuracy by individual crossing count and then group into fixed ranges:
      - Low: 0-1 intersections
      - Medium: 2 intersections
      - High: 3+ intersections
    """
    crossing_accuracies = defaultdict(list)
    range_accuracies = {
        "low (1)": [],
        "medium (2)": [],
        "high (3+)": []
    }
    for model_name, model_data in results.items():
        for num_crossings, accuracy in model_data['avg_by_crossings'].items():
            if num_crossings < 0:
                continue
            crossing_accuracies[num_crossings].append(accuracy)
            if num_crossings == 0:
                raise ValueError("0 crossings should not be present in the data")
            elif num_crossings <= 1:
                range_accuracies["low (1)"].append(accuracy)
            elif num_crossings == 2:
                range_accuracies["medium (2)"].append(accuracy)
            else:
                range_accuracies["high (3+)"].append(accuracy)

    avg_by_crossing = {}
    for num_crossings, accuracies in crossing_accuracies.items():
        if accuracies:
            avg_by_crossing[num_crossings] = sum(accuracies) / len(accuracies)

    range_averages = {}
    range_std_devs = {}
    for range_name, accuracies in range_accuracies.items():
        if accuracies:
            range_averages[range_name] = sum(accuracies) / len(accuracies)
            range_std_devs[range_name] = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0
        else:
            range_averages[range_name] = 0
            range_std_devs[range_name] = 0

    return avg_by_crossing, range_averages, range_std_devs


def generate_grouped_plot(range_averages, range_std_devs, model_type="all", output_dir="plots"):
    """
    Generate a bar plot with error bars for one set of models.
    """
    df = pd.DataFrame({
        "Range": list(range_averages.keys()),
        "AvgAccuracy": list(range_averages.values()),
        "StdDev": [range_std_devs.get(range_name, 0) for range_name in range_averages.keys()]
    })

    bar_color = '#b7daf5'
    error_color = 'black'
    title_prefix = "Reasoning Models" if model_type == "reasoning" else "Non-Reasoning Models"

    fig = px.bar(
        df,
        x="Range",
        y="AvgAccuracy",
        title=f"Average WCR vs. Crossing Letter Count<br>Across {title_prefix}",
        labels={"Range": "Crossing Letter Count Range", "AvgAccuracy": "Average WCR"},
        template="plotly_white",
        error_y="StdDev"
    )

    fig.update_traces(
        marker_line_width=1,
        marker_line_color='black',
        marker_color=bar_color,
        error_y=dict(
            type='data',
            color=error_color,
            thickness=2,
            width=10
        )
    )

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', zeroline=False, showgrid=False, automargin=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', zeroline=False, showgrid=True, gridcolor='lightgray')
    fig.update_layout(
        title=dict(
            text=f'Average WCR vs. Crossing Letter Count<br>Across {title_prefix}',
            font=dict(size=70, family='Palatino, serif', color='black', weight='bold'),
            x=0.5
        ),
        xaxis_title=dict(
            text="Crossing Letter Count Range",
            font=dict(size=60, family='Palatino, serif', color='black', weight='bold')
        ),
        yaxis_title=dict(
            text="Average WCR",
            font=dict(size=60, family='Palatino, serif', color='black', weight='bold')
        ),
        xaxis=dict(tickfont=dict(size=58, family='Palatino, serif', color='black')),
        yaxis=dict(tickfont=dict(size=58, family='Palatino, serif', color='black'),
                   range=[0, min(1, (df['AvgAccuracy'] + df['StdDev']).max() * 1.15)]),
        plot_bgcolor='white',
        margin=dict(l=80, r=30, t=150, b=80),
        width=1800,
        height=1000
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_prefix = "reasoning_" if model_type == "reasoning" else "non_reasoning_"
    output_file = os.path.join(output_dir, f"{file_prefix}average_accuracy_by_range_plot.svg")
    fig.write_image(output_file, width=2000, height=1000, scale=1, format="svg")

    return df


def generate_combined_grouped_plot(results_reasoning, results_non_reasoning, total_words_reasoning, total_words_non_reasoning, output_dir="plots"):
    """
    Generate a combined bar plot with error bars for reasoning and non-reasoning models.
    The two groups will appear side by side (grouped) with different colors.
    """
    # Obtain aggregated range accuracy stats for both groups.
    _, range_averages_reasoning, range_std_devs_reasoning = analyze_accuracy_by_crossings(results_reasoning, total_words_reasoning)
    _, range_averages_non_reasoning, range_std_devs_non_reasoning = analyze_accuracy_by_crossings(results_non_reasoning, total_words_non_reasoning)

    # Build a combined DataFrame using your specified legend names.
    data = []
    for range_name, avg in range_averages_reasoning.items():
        data.append({
            "Range": range_name,
            "AvgAccuracy": avg,
            "StdDev": range_std_devs_reasoning.get(range_name, 0),
            "ModelType": "Five Reasoning LLMs<br>(exclude Open-Reasoner-Zero-32B) "
        })
    for range_name, avg in range_averages_non_reasoning.items():
        data.append({
            "Range": range_name,
            "AvgAccuracy": avg,
            "StdDev": range_std_devs_non_reasoning.get(range_name, 0),
            "ModelType": "Nine Non-reasoning LLMs"
        })

    df = pd.DataFrame(data)

    # Create grouped bar plot with your specified colors.
    fig = px.bar(
        df,
        x="Range",
        y="AvgAccuracy",
        color="ModelType",
        barmode="group",
        error_y="StdDev",
        title="Average WCR vs. Crossing Letter Count",
        labels={"Range": "Crossing Letter Count Range", "AvgAccuracy": "Average WCR", "ModelType": ""},
        color_discrete_map={"Five Reasoning LLMs<br>(exclude Open-Reasoner-Zero-32B) ": "#f9bdb6", "Nine Non-reasoning LLMs": "#b7daf5"}
    )

    fig.update_traces(
        marker_line_width=1,
        marker_line_color='black',
        error_y=dict(
            type='data',
            color='black',
            thickness=2,
            width=10
        )
    )

    fig.update_xaxes(
        showline=True, 
        linewidth=2, 
        linecolor='black', 
        zeroline=False, 
        showgrid=False,
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

    # Place legend inside the plot and remove its title.
    fig.update_layout(
        title=dict(
            text='Average WCR vs. Crossing Letter Count',
            font=dict(size=70, family='Palatino, serif', color='black', weight='bold'),
            x=0.5
        ),
        xaxis_title=dict(
            text="Crossing Letter Count Range",
            font=dict(size=60, family='Palatino, serif', color='black', weight='bold')
        ),
        yaxis_title=dict(
            text="Average WCR",
            font=dict(size=60, family='Palatino, serif', color='black', weight='bold')
        ),
        xaxis=dict(
            tickfont=dict(size=58, family='Palatino, serif', color='black')
        ),
        yaxis=dict(
            tickfont=dict(size=58, family='Palatino, serif', color='black')
        ),
        legend=dict(
            x=0.02,            # Position inside plot on the left
            y=1.05,            # Position inside plot at the top
            xanchor="left",
            bgcolor='rgba(0,0,0,0)',  # Transparent background
            bordercolor='rgba(0,0,0,0)',  # Transparent border
            borderwidth=0,
            font=dict(size=50, color='black', family='Palatino, serif'),
            title=dict(text=""), 
            itemsizing="constant",  
            itemwidth=150         
        ),
        plot_bgcolor='white',
        margin=dict(l=80, r=40, t=120, b=90),
        width=2000,
        height=800
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "combined_average_accuracy_by_range_plot.svg")
    fig.write_image(output_file, width=2000, height=800, scale=1, format="svg")

    return df



def main():
    parser = argparse.ArgumentParser(description="Analyze crossword correctness rates by number of crossing letters.")
    parser.add_argument("--directory", help="Base directory containing model folders", default="../eval_results")
    parser.add_argument("--puzzles", type=int, default=100, help="Required number of puzzles per model")
    parser.add_argument("--output", help="Directory for output plots", default="../plots")
    parser.add_argument("--reasoning", action="store_true", help="Analyze only reasoning models")
    parser.add_argument("--combined", action="store_true", help="Generate combined plot for reasoning and non-reasoning models")
    args = parser.parse_args()

    if args.combined:
        print("Analyzing reasoning models...")
        results_reasoning, total_words_reasoning, puzzles_count_r = analyze_models(args.directory, args.puzzles, only_reasoning=True)
        print("Analyzing non-reasoning models...")
        results_non_reasoning, total_words_non_reasoning, puzzles_count_nr = analyze_models(args.directory, args.puzzles, only_reasoning=False)
        print(f"\nAnalyzed {puzzles_count_r} puzzles for reasoning models and {puzzles_count_nr} puzzles for non-reasoning models")
        combined_df = generate_combined_grouped_plot(results_reasoning, results_non_reasoning, total_words_reasoning, total_words_non_reasoning, args.output)
        print("Combined plot generated.")
    else:
        results, total_words_by_crossings, total_puzzles = analyze_models(args.directory, args.puzzles, args.reasoning)
        model_type = "reasoning" if args.reasoning else "all"
        print(f"\nAnalyzed {total_puzzles} unique puzzles across {len(results)} {model_type} models")
        for model_name, model_data in results.items():
            print(f"\n=== Model: {model_name} ({model_data['file_count']} puzzles) ===")
            print(f"Overall Correctness Accuracy: {model_data['avg_overall_accuracy']:.4f}")
            print("Accuracy by Number of Crossing Letters:")
            all_crossings = sorted(set(list(model_data['avg_by_crossings'].keys()) + list(total_words_by_crossings.keys())))
            for num_crossings in all_crossings:
                if num_crossings < 0:
                    continue
                accuracy = model_data['avg_by_crossings'].get(num_crossings, 0)
                total_words = total_words_by_crossings.get(num_crossings, 0) * model_data['file_count']
                avg_per_puzzle = total_words / model_data['file_count'] if model_data['file_count'] > 0 else 0
                print(f"  {num_crossings} crossings: {accuracy:.4f} (words: {int(total_words)}, avg per puzzle: {avg_per_puzzle:.2f})")

        if 0 in total_words_by_crossings:
            print(f"\nWords with 0 crossings exist: average count per puzzle: {total_words_by_crossings[0]:.2f}")
        else:
            print("\nNo words with 0 crossings exist in the puzzles.")

        _, overall_correlation = analyze_correlations(results)
        if overall_correlation:
            pearson_r, pearson_p = overall_correlation["pearson"]
            spearman_r, spearman_p = overall_correlation["spearman"]
            print(f"\nOverall Pearson correlation: r={pearson_r:.4f} (p-value: {pearson_p:.4f})")
            print(f"Overall Spearman correlation: ρ={spearman_r:.4f} (p-value: {spearman_p:.4f})")
            if abs(pearson_r) < 0.1:
                corr_strength = "negligible"
            elif abs(pearson_r) < 0.3:
                corr_strength = "weak"
            elif abs(pearson_r) < 0.5:
                corr_strength = "moderate"
            elif abs(pearson_r) < 0.7:
                corr_strength = "strong"
            else:
                corr_strength = "very strong"
            corr_direction = "positive" if pearson_r > 0 else "negative"
            print(f"Interpretation: {corr_strength} {corr_direction} correlation between number of crossings and accuracy for {model_type} models")

        _, range_averages, range_std_devs = analyze_accuracy_by_crossings(results, total_words_by_crossings)
        grouped_df = generate_grouped_plot(range_averages, range_std_devs, model_type, args.output)
        print("\n=== Accuracy by Intersection Range ===")
        for range_name, avg_accuracy in range_averages.items():
            std_dev = range_std_devs.get(range_name, 0)
            print(f"  {range_name}: {avg_accuracy:.4f} ± {std_dev:.4f}")


def analyze_correlations(results):
    """Analyze correlations between number of crossings and accuracy across models."""
    model_correlations = {}
    all_crossing_numbers = []
    all_accuracies = []
    for model_name, model_data in results.items():
        crossing_numbers = []
        accuracies = []
        for num_crossings, accuracy in model_data['avg_by_crossings'].items():
            if num_crossings >= 0:
                crossing_numbers.append(num_crossings)
                accuracies.append(accuracy)
                all_crossing_numbers.append(num_crossings)
                all_accuracies.append(accuracy)
        if len(crossing_numbers) >= 3:
            pearson_r, pearson_p = stats.pearsonr(crossing_numbers, accuracies)
            spearman_r, spearman_p = stats.spearmanr(crossing_numbers, accuracies)
            model_correlations[model_name] = {
                "pearson": (pearson_r, pearson_p),
                "spearman": (spearman_r, spearman_p),
                "data": list(zip(crossing_numbers, accuracies))
            }
    overall_correlation = None
    if len(all_crossing_numbers) >= 3:
        pearson_r, pearson_p = stats.pearsonr(all_crossing_numbers, all_accuracies)
        spearman_r, spearman_p = stats.spearmanr(all_crossing_numbers, all_accuracies)
        overall_correlation = {
            "pearson": (pearson_r, pearson_p),
            "spearman": (spearman_r, spearman_p)
        }
    return model_correlations, overall_correlation


if __name__ == "__main__":
    main()
