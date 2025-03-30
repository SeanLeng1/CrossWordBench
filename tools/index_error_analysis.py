import os
import json
from typing import List, Dict, Any, Tuple
import argparse
from collections import defaultdict
import re
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class CrosswordClueAnalyzer:
    @staticmethod
    def extract_clue_number(direction: str) -> int:
        """Extract the clue number from a direction string like 'across 2'."""
        match = re.search(r'\d+', direction)
        if match:
            return int(match.group())
        return 0
    
    @staticmethod
    def group_by_clue_number(reference_answer: List[Dict[str, Any]]) -> Dict[int, List[str]]:
        """Group words by their clue number from the reference answer."""
        clue_groups = defaultdict(list)
        
        for entry in reference_answer:
            direction = entry.get("direction", "")
            clue_number = CrosswordClueAnalyzer.extract_clue_number(direction)
            clue_groups[clue_number].append(direction)
            
        return dict(clue_groups)

def analyze_puzzle_structure(reference_answer):
    """Analyze puzzle structure and count words by clue number."""
    # Group words by clue number
    clue_groups = CrosswordClueAnalyzer.group_by_clue_number(reference_answer)
    
    # Count words for each clue number
    clue_counts = {num: len(words) for num, words in clue_groups.items()}
    
    return clue_counts, clue_groups

def calculate_correctness_by_clue_number(model_answer, reference_answer, clue_groups):
    """Calculate correctness rate grouped by clue number."""
    if not reference_answer or not model_answer:
        return {}, 0
    
    ref_dict = {}
    for entry in reference_answer:
        ref_dict[entry["direction"]] = entry["answer"]
    
    # Group accuracy by clue number
    clue_number_groups = defaultdict(lambda: {"correct": 0, "total": 0})
    
    total_correct = 0
    total_words = 0
    
    for direction, ref_word in ref_dict.items():
        # Skip if the model didn't provide an answer for this direction
        if direction not in model_answer:
            total_words += 1
            continue
            
        model_word = model_answer[direction].get("answer", "")
        clue_number = CrosswordClueAnalyzer.extract_clue_number(direction)
        
        # Check if model's answer matches reference
        is_correct = model_word.upper() == ref_word.upper()
        
        # Update counts
        if is_correct:
            clue_number_groups[clue_number]["correct"] += 1
            total_correct += 1
        
        clue_number_groups[clue_number]["total"] += 1
        total_words += 1
    
    # Calculate accuracy for each clue number group
    clue_number_accuracy = {}
    for clue_number, stats in clue_number_groups.items():
        if stats["total"] > 0:
            clue_number_accuracy[clue_number] = stats["correct"] / stats["total"]
    
    overall_accuracy = total_correct / total_words if total_words > 0 else 0
    
    return clue_number_accuracy, overall_accuracy

def analyze_models(base_dir, use_text=False, required_puzzle_count=100):
    """First analyze puzzle structures, then calculate correctness rates for each model."""
    # Step 1: Collect all models and validate they have the required number of puzzles
    valid_model_dirs = {}
    
    for model_name in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, model_name)
        
        if not os.path.isdir(model_dir):
            continue
        
        # Look for 7x7 directory
        grid_dir = os.path.join(model_dir, "english/7x7")
        if not os.path.exists(grid_dir):
            continue
        
        # Choose between img_cot and text_cot based on the flag
        cot_type = "text_cot" if use_text else "img_cot"
        cot_dir = os.path.join(grid_dir, cot_type)
        
        if not os.path.exists(cot_dir):
            continue
        
        # Check if this model has the required number of puzzles
        puzzle_files = [f for f in os.listdir(cot_dir) if f.endswith(".json")]
        
        if len(puzzle_files) >= required_puzzle_count:
            valid_model_dirs[model_name] = {
                "dir": cot_dir,
                "puzzles": puzzle_files[:required_puzzle_count]  # Take only the required number
            }
        else:
            print(f"Skipping model {model_name}: Only has {len(puzzle_files)} puzzles, {required_puzzle_count} required")
    
    print(f"Found {len(valid_model_dirs)} models with at least {required_puzzle_count} puzzles")
    
    # Step 2: Analyze puzzle structures
    # Maps puzzle filename to its structure analysis
    puzzle_structures = {}
    # Count words by clue number across all puzzles
    total_words_by_clue_number = defaultdict(int)
    
    # Loop through all models and collect structure information for each puzzle
    all_puzzles = set()
    for model_name, model_info in valid_model_dirs.items():
        for filename in model_info["puzzles"]:
            all_puzzles.add(filename)
            
            if filename in puzzle_structures:
                continue  # Already analyzed this puzzle
                
            file_path = os.path.join(model_info["dir"], filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                reference_answer = data.get("reference_answer", [])
                
                if reference_answer:
                    clue_counts, clue_groups = analyze_puzzle_structure(reference_answer)
                    
                    puzzle_structures[filename] = {
                        "clue_counts": clue_counts,
                        "clue_groups": clue_groups,
                        "reference_answer": reference_answer
                    }
                    
                    # Add to total counts (we'll divide by model count later)
                    for clue_number, count in clue_counts.items():
                        total_words_by_clue_number[clue_number] += count
            
            except Exception as e:
                print(f"Error analyzing puzzle structure for {file_path}: {e}")
    
    # Step 3: Calculate correctness rates for each model
    results = {}
    
    for model_name, model_info in valid_model_dirs.items():
        model_dir = model_info["dir"]
        model_results = {
            "overall_accuracy": [],
            "clue_number_accuracy": defaultdict(list),
            "processed_puzzles": 0,
            "puzzle_results": {}  # Store individual puzzle results for debugging
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
                    # Use pre-computed clue groups and reference answer
                    puzzle_info = puzzle_structures[filename]
                    
                    clue_number_acc, overall_acc = calculate_correctness_by_clue_number(
                        model_answer, 
                        puzzle_info["reference_answer"],
                        puzzle_info["clue_groups"]
                    )
                    
                    model_results["overall_accuracy"].append(overall_acc)
                    model_results["processed_puzzles"] += 1
                    model_results["puzzle_results"][filename] = {
                        "overall": overall_acc,
                        "by_clue": clue_number_acc
                    }
                    
                    # Collect accuracy by clue number
                    for clue_number, accuracy in clue_number_acc.items():
                        model_results["clue_number_accuracy"][clue_number].append(accuracy)
            
            except Exception as e:
                print(f"Error processing {file_path} for model {model_name}: {e}")
        
        # Calculate averages
        if model_results["processed_puzzles"] == required_puzzle_count:  # Ensure we processed all required puzzles
            avg_overall = sum(model_results["overall_accuracy"]) / len(model_results["overall_accuracy"]) if model_results["overall_accuracy"] else 0
            
            avg_by_clue_number = {}
            for clue_number, accuracies in model_results["clue_number_accuracy"].items():
                avg_by_clue_number[clue_number] = sum(accuracies) / len(accuracies) if accuracies else 0
            
            results[model_name] = {
                "file_count": model_results["processed_puzzles"],
                "avg_overall_accuracy": avg_overall,
                "avg_by_clue_number": avg_by_clue_number,
                "puzzle_results": model_results["puzzle_results"]  # Include for debugging
            }
        else:
            print(f"Skipping model {model_name} in results: Only processed {model_results['processed_puzzles']} of {required_puzzle_count} required puzzles")
    
    # Calculate the correct average word counts per puzzle
    model_count = len(results)
    if model_count > 0:
        for clue_number in total_words_by_clue_number:
            # Divide by number of models to get the average per puzzle
            total_words_by_clue_number[clue_number] = total_words_by_clue_number[clue_number] / model_count
    
    return results, total_words_by_clue_number, len(all_puzzles)

def analyze_correlations(results):
    """Analyze correlations between clue index and accuracy."""
    model_correlations = {}
    all_clue_numbers = []
    all_accuracies = []
    
    for model_name, model_data in results.items():
        clue_numbers = []
        accuracies = []
        
        for clue_number, accuracy in model_data['avg_by_clue_number'].items():
            if clue_number > 0:  # Skip clue 0 if it exists
                clue_numbers.append(clue_number)
                accuracies.append(accuracy)
                all_clue_numbers.append(clue_number)
                all_accuracies.append(accuracy)
        
        # Calculate correlation only if we have enough data points
        if len(clue_numbers) >= 5:
            pearson_r, pearson_p = stats.pearsonr(clue_numbers, accuracies)
            spearman_r, spearman_p = stats.spearmanr(clue_numbers, accuracies)
            
            model_correlations[model_name] = {
                "pearson": (pearson_r, pearson_p),
                "spearman": (spearman_r, spearman_p),
                "data": list(zip(clue_numbers, accuracies))
            }
    
    # Calculate overall correlation across all models
    overall_correlation = None
    if len(all_clue_numbers) >= 5:
        pearson_r, pearson_p = stats.pearsonr(all_clue_numbers, all_accuracies)
        spearman_r, spearman_p = stats.spearmanr(all_clue_numbers, all_accuracies)
        
        overall_correlation = {
            "pearson": (pearson_r, pearson_p),
            "spearman": (spearman_r, spearman_p)
        }
    
    return model_correlations, overall_correlation

def generate_correlation_plots(model_correlations, output_dir="correlation_plots"):
    """Generate scatter plots with trend lines for each model."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate individual model plots
    for model_name, correlation_data in model_correlations.items():
        clue_numbers, accuracies = zip(*correlation_data["data"])
        
        plt.figure(figsize=(10, 6))
        plt.scatter(clue_numbers, accuracies, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(clue_numbers, accuracies, 1)
        p = np.poly1d(z)
        plt.plot(clue_numbers, p(clue_numbers), "r--", alpha=0.7)
        
        plt.title(f"{model_name}: Clue Number vs. Accuracy")
        plt.xlabel("Clue Number")
        plt.ylabel("Accuracy")
        
        # Add correlation information
        pearson_r, pearson_p = correlation_data["pearson"]
        spearman_r, spearman_p = correlation_data["spearman"]
        
        plt.figtext(0.15, 0.85, f"Pearson r: {pearson_r:.4f} (p={pearson_p:.4f})")
        plt.figtext(0.15, 0.82, f"Spearman ρ: {spearman_r:.4f} (p={spearman_p:.4f})")
        
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{model_name}_correlation.png"))
        plt.close()
    
    if len(model_correlations) > 1:
        plt.figure(figsize=(12, 8))
        
        all_x = []
        all_y = []
        model_points = {}
        
        for model_name, correlation_data in model_correlations.items():
            clue_numbers, accuracies = zip(*correlation_data["data"])
            model_points[model_name] = (clue_numbers, accuracies)
            all_x.extend(clue_numbers)
            all_y.extend(accuracies)
        
        for i, (model_name, (x_values, y_values)) in enumerate(model_points.items()):
            plt.scatter(x_values, y_values, alpha=0.7, label=model_name)
        
        if all_x and all_y:
            z = np.polyfit(all_x, all_y, 1)
            p = np.poly1d(z)
            
            x_range = np.linspace(min(all_x), max(all_x), 100)
            plt.plot(x_range, p(x_range), "k--", linewidth=2, alpha=0.8, label="Overall trend")
            
            pearson_r, pearson_p = stats.pearsonr(all_x, all_y)
            plt.figtext(0.15, 0.85, f"Overall Pearson r: {pearson_r:.4f} (p={pearson_p:.4f})")
        
        plt.title("Clue Number vs. Accuracy Across All Models")
        plt.xlabel("Clue Number")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, "combined_correlation.png"))
        plt.close()

def analyze_accuracy_by_clue_range(results, total_words_by_clue_number):
    """Analyze accuracy for early vs. late clue numbers."""
    ranges = {
        "early (1-3)": [],
        "middle (4-7)": [],
        "late (8+)": []
    }
    
    for model_name, model_data in results.items():
        model_ranges = {
            "early (1-3)": {"correct": 0, "total": 0},
            "middle (4-7)": {"correct": 0, "total": 0},
            "late (8+)": {"correct": 0, "total": 0}
        }
        
        for clue_number, accuracy in model_data['avg_by_clue_number'].items():
            if clue_number <= 0:
                continue
                
            total_words = total_words_by_clue_number[clue_number] * model_data['file_count']
            correct_words = accuracy * total_words
            
            # Categorize by range
            if 1 <= clue_number <= 3:
                range_key = "early (1-3)"
            elif 4 <= clue_number <= 7:
                range_key = "middle (4-7)"
            else:
                range_key = "late (8+)"
            
            model_ranges[range_key]["correct"] += correct_words
            model_ranges[range_key]["total"] += total_words
        
        # Calculate accuracy for each range
        for range_key, data in model_ranges.items():
            if data["total"] > 0:
                range_accuracy = data["correct"] / data["total"]
                ranges[range_key].append((model_name, range_accuracy, data["total"]))
    
    return ranges

def main():
    parser = argparse.ArgumentParser(description="Analyze crossword correctness rates by clue number.")
    parser.add_argument("--directory", help="Base directory containing model folders", default="../eval_results")
    parser.add_argument("--text", action="store_true", help="Use text_cot instead of img_cot")
    parser.add_argument("--puzzles", type=int, default=100, help="Required number of puzzles per model")
    parser.add_argument("--plot", action="store_true", help="Generate correlation plots")
    
    args = parser.parse_args()
    
    results, total_words_by_clue_number, total_puzzles = analyze_models(args.directory, args.text, args.puzzles)
    
    print(f"\nAnalyzed {total_puzzles} unique puzzles across {len(results)} models")
    
    for model_name, model_data in results.items():
        print(f"\n=== Model: {model_name} ({model_data['file_count']} puzzles) ===")
        print(f"Overall Correctness Accuracy: {model_data['avg_overall_accuracy']:.4f}")
        print("Accuracy by Clue Number:")
        
        for clue_number in sorted(total_words_by_clue_number.keys()):
            if clue_number in model_data['avg_by_clue_number']:
                accuracy = model_data['avg_by_clue_number'][clue_number]
                total_words = total_words_by_clue_number[clue_number] * model_data['file_count']
                avg_per_puzzle = total_words / model_data['file_count'] if model_data['file_count'] > 0 else 0
                print(f"  Clue {clue_number}: {accuracy:.4f} (words: {int(total_words)}, avg per puzzle: {avg_per_puzzle:.2f})")
    
    print("\n=== Correlation Analysis ===")
    model_correlations, overall_correlation = analyze_correlations(results)
    
    if overall_correlation:
        pearson_r, pearson_p = overall_correlation["pearson"]
        spearman_r, spearman_p = overall_correlation["spearman"]
        
        print(f"Overall Pearson correlation: r={pearson_r:.4f} (p-value: {pearson_p:.4f})")
        print(f"Overall Spearman correlation: ρ={spearman_r:.4f} (p-value: {spearman_p:.4f})")
        
        corr_strength = ""
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
        print(f"Interpretation: {corr_strength} {corr_direction} correlation between clue number and accuracy")
    
    print("\n=== Accuracy by Clue Range ===")
    range_accuracy = analyze_accuracy_by_clue_range(results, total_words_by_clue_number)
    
    for range_name, models_data in range_accuracy.items():
        if models_data:
            accuracies = [acc for _, acc, _ in models_data]
            avg_accuracy = sum(accuracies) / len(accuracies)
            total_words = sum(words for _, _, words in models_data)
            print(f"Clues {range_name}: {avg_accuracy:.4f} average accuracy (across {len(models_data)} models, {int(total_words)} total words)")
    
    if args.plot and model_correlations:
        print("\nGenerating correlation plots...")
        generate_correlation_plots(model_correlations)

if __name__ == "__main__":
    main()