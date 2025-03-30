import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze crossword puzzles for duplicate clues")
    parser.add_argument(
        "--data_dirs", 
        type=str, 
        nargs="+",
        default=["../data/english/7x7"],
        help="Multiple directories containing puzzle folders (e.g., --data_dirs data/english/7x7 data/english/14x14)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed information about duplicate clues"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Output file to save results (optional)"
    )
    parser.add_argument(
        "--cross_size", 
        action="store_true",
        help="Look for duplicates across different size puzzles (default: analyze each size separately)"
    )
    parser.add_argument(
        "--cross_size_only", 
        action="store_true",
        help="Only show clues that appear in multiple different size categories"
    )
    return parser.parse_args()

def find_puzzle_files(data_dirs):
    all_puzzle_files = []

    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"Warning: Directory '{data_dir}' does not exist. Skipping.")
            continue
        
        puzzle_files = list(data_path.glob("**/puzzle_state.json"))
        
        if not puzzle_files:
            print(f"Warning: No puzzle_state.json files found in '{data_dir}'. Skipping.")
            continue
        
        all_puzzle_files.extend(puzzle_files)
        print(f"Found {len(puzzle_files)} puzzle files in '{data_dir}'")
    
    if not all_puzzle_files:
        print(f"Error: No puzzle_state.json files found in any specified directories.")
        sys.exit(1)
    
    return all_puzzle_files

def extract_clues(puzzle_file):
    with open(puzzle_file, 'r') as f:
        try:
            puzzle_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {puzzle_file}")
            return []
    
    wordlist = puzzle_data.get('wordlist', [])
    puzzle_id = puzzle_data.get('meta_data', {}).get('id', 'unknown')
    puzzle_path = str(puzzle_file.parent)
    
    size_match = None
    for part in puzzle_path.split(os.sep):
        if "x" in part and part.replace("x", "").isdigit():
            size_match = part
            break
    
    puzzle_size = size_match if size_match else "unknown_size"
    
    clues = []
    for word_entry in wordlist:
        if len(word_entry) > 1:
            # format is: [word, clue, y_pos, x_pos, orientation]
            clue = word_entry[1]
            word = word_entry[0]
            clues.append((clue, word, puzzle_id, puzzle_path, puzzle_size))
    
    return clues

def analyze_clues(puzzle_files, cross_size=False, cross_size_only=False, verbose=False):
    """Analyze clues from all puzzle files and identify duplicates."""
    all_clues = []
    puzzles_count = len(puzzle_files)
    
    print(f"Analyzing {puzzles_count} puzzles for duplicate clues...")
    
    for puzzle_file in puzzle_files:
        clues = extract_clues(puzzle_file)
        all_clues.extend(clues)
    
    clues_by_size = defaultdict(list)
    sizes = set()
    for clue in all_clues:
        size = clue[4]
        sizes.add(size)
        clues_by_size[size].append(clue)
    
    if cross_size_only:
        clue_to_sizes = defaultdict(set)
        for clue, word, puzzle_id, puzzle_path, puzzle_size in all_clues:
            clue_to_sizes[clue].add(puzzle_size)
        
        cross_size_duplicates = {clue: sizes for clue, sizes in clue_to_sizes.items() if len(sizes) > 1}
        
        cross_size_details = defaultdict(list)
        for clue, word, puzzle_id, puzzle_path, puzzle_size in all_clues:
            if clue in cross_size_duplicates:
                cross_size_details[clue].append((word, puzzle_id, puzzle_path, puzzle_size))
        
        return {
            "cross_size_only": True,
            "total_puzzles": puzzles_count,
            "total_clues": len(all_clues),
            "cross_size_duplicates_count": len(cross_size_duplicates),
            "cross_size_duplicates": cross_size_duplicates,
            "cross_size_details": cross_size_details,
            "sizes": sorted(sizes)
        }
    
    if not cross_size:
        results_by_size = {}
        
        for size in sizes:
            size_clues = clues_by_size[size]
            size_puzzles_count = len(set(clue[2] for clue in size_clues))
            
            clue_counts = Counter([clue[0] for clue in size_clues])
            duplicates = {clue: count for clue, count in clue_counts.items() if count > 1}
            
            clue_to_word = defaultdict(list)
            for clue, word, puzzle_id, puzzle_path, _ in size_clues:
                if clue in duplicates:
                    clue_to_word[clue].append((word, puzzle_id, puzzle_path))
            
            results_by_size[size] = {
                "total_puzzles": size_puzzles_count,
                "total_clues": len(size_clues),
                "unique_clues": len(clue_counts),
                "duplicate_clues_count": len(duplicates),
                "duplicates": duplicates,
                "duplicate_details": clue_to_word
            }
        
        clue_to_sizes = defaultdict(set)
        for clue, word, puzzle_id, puzzle_path, puzzle_size in all_clues:
            clue_to_sizes[clue].add(puzzle_size)
        
        cross_size_duplicates = {clue: sizes for clue, sizes in clue_to_sizes.items() if len(sizes) > 1}
        
        cross_size_details = defaultdict(list)
        for clue, word, puzzle_id, puzzle_path, puzzle_size in all_clues:
            if clue in cross_size_duplicates:
                cross_size_details[clue].append((word, puzzle_id, puzzle_path, puzzle_size))
        
        return {
            "cross_size": False,
            "results_by_size": results_by_size,
            "cross_size_duplicates_count": len(cross_size_duplicates),
            "cross_size_duplicates": cross_size_duplicates,
            "cross_size_details": cross_size_details,
            "sizes": sorted(sizes)
        }
    
    clue_counts = Counter([clue[0] for clue in all_clues])
    duplicates = {clue: count for clue, count in clue_counts.items() if count > 1}
    
    clue_to_word = defaultdict(list)
    for clue, word, puzzle_id, puzzle_path, puzzle_size in all_clues:
        if clue in duplicates:
            clue_to_word[clue].append((word, puzzle_id, puzzle_path, puzzle_size))
    
    clue_to_sizes = defaultdict(set)
    for clue, word, puzzle_id, puzzle_path, puzzle_size in all_clues:
        clue_to_sizes[clue].add(puzzle_size)
    
    cross_size_duplicates = {clue: sizes for clue, sizes in clue_to_sizes.items() if len(sizes) > 1}
    
    return {
        "cross_size": True,
        "total_puzzles": puzzles_count,
        "total_clues": len(all_clues),
        "unique_clues": len(clue_counts),
        "duplicate_clues_count": len(duplicates),
        "duplicates": duplicates,
        "duplicate_details": clue_to_word,
        "cross_size_duplicates_count": len(cross_size_duplicates),
        "cross_size_duplicates": cross_size_duplicates,
        "sizes": sorted(sizes)
    }

def print_results(results, verbose=False):
    print("\n===== Crossword Puzzle Clue Analysis =====")
    
    if results.get("cross_size_only", False):
        print(f"\n--- Cross-Size Clue Analysis ---")
        print(f"Total puzzles analyzed: {results['total_puzzles']}")
        print(f"Total clues found: {results['total_clues']}")
        print(f"Clues appearing in multiple size categories: {results['cross_size_duplicates_count']}")
        
        if results['cross_size_duplicates_count'] > 0:
            print(f"\nSizes analyzed: {', '.join(results['sizes'])}")
            
            sorted_duplicates = sorted(
                results['cross_size_duplicates'].items(), 
                key=lambda x: (len(x[1]), x[0]), 
                reverse=True
            )
            
            print("\nClues that appear across different size puzzles:")
            for clue, sizes in sorted_duplicates[:10]:
                print(f"  \"{clue}\" appears in {len(sizes)} different sizes: {', '.join(sorted(sizes))}")
            
            if verbose:
                print("\nDetailed cross-size duplicate clue information:")
                for clue, instances in results['cross_size_details'].items():
                    sizes = results['cross_size_duplicates'][clue]
                    print(f"\nClue: \"{clue}\" appears in {len(sizes)} sizes ({', '.join(sorted(sizes))}):")
                    
                    by_size = defaultdict(list)
                    for word, puzzle_id, puzzle_path, size in instances:
                        by_size[size].append((word, puzzle_id, puzzle_path))
                    
                    for size in sorted(by_size.keys()):
                        print(f"  Size {size}:")
                        for word, puzzle_id, puzzle_path in by_size[size]:
                            print(f"    - Word: {word}, Puzzle ID: {puzzle_id}, Path: {puzzle_path}")
        else:
            print("No clues found that appear across different size puzzles!")
        return
    
    if not results["cross_size"]:
        for size, size_results in results["results_by_size"].items():
            print(f"\n--- Results for {size} puzzles ---")
            print(f"Total puzzles analyzed: {size_results['total_puzzles']}")
            print(f"Total clues found: {size_results['total_clues']}")
            print(f"Unique clues: {size_results['unique_clues']}")
            print(f"Clues appearing in multiple puzzles: {size_results['duplicate_clues_count']}")
            
            if size_results['duplicate_clues_count'] > 0:
                duplicate_percentage = (size_results['duplicate_clues_count'] / size_results['unique_clues']) * 100
                print(f"Percentage of clues that are duplicates: {duplicate_percentage:.2f}%")
                
                common_duplicates = sorted(size_results['duplicates'].items(), key=lambda x: x[1], reverse=True)
                print("\nMost frequently repeated clues:")
                for clue, count in common_duplicates[:10]: 
                    print(f"  \"{clue}\" appears {count} times")
                
                if verbose:
                    print("\nDetailed duplicate clue information:")
                    for clue, instances in size_results['duplicate_details'].items():
                        print(f"\nClue: \"{clue}\" appears {len(instances)} times:")
                        for word, puzzle_id, puzzle_path in instances:
                            print(f"  - Word: {word}, Puzzle ID: {puzzle_id}, Path: {puzzle_path}")
            else:
                print("No duplicate clues found across puzzles!")
        
        print(f"\n--- Cross-Size Clue Analysis ---")
        print(f"Clues appearing in multiple size categories: {results['cross_size_duplicates_count']}")
        
        if results['cross_size_duplicates_count'] > 0:
            print(f"\nSizes analyzed: {', '.join(results['sizes'])}")
            
            sorted_duplicates = sorted(
                results['cross_size_duplicates'].items(), 
                key=lambda x: (len(x[1]), x[0]), 
                reverse=True
            )
            
            print("\nClues that appear across different size puzzles:")
            for clue, sizes in sorted_duplicates[:10]:
                print(f"  \"{clue}\" appears in {len(sizes)} different sizes: {', '.join(sorted(sizes))}")
            
            if verbose:
                print("\nDetailed cross-size duplicate clue information:")
                for clue, instances in results['cross_size_details'].items():
                    sizes = results['cross_size_duplicates'][clue]
                    print(f"\nClue: \"{clue}\" appears in {len(sizes)} sizes ({', '.join(sorted(sizes))}):")
                    
                    by_size = defaultdict(list)
                    for word, puzzle_id, puzzle_path, size in instances:
                        by_size[size].append((word, puzzle_id, puzzle_path))
                    
                    for size in sorted(by_size.keys()):
                        print(f"  Size {size}:")
                        for word, puzzle_id, puzzle_path in by_size[size]:
                            print(f"    - Word: {word}, Puzzle ID: {puzzle_id}, Path: {puzzle_path}")
    else:
        print(f"Total puzzles analyzed (all sizes): {results['total_puzzles']}")
        print(f"Total clues found: {results['total_clues']}")
        print(f"Unique clues: {results['unique_clues']}")
        print(f"Clues appearing in multiple puzzles: {results['duplicate_clues_count']}")
        
        if results['duplicate_clues_count'] > 0:
            duplicate_percentage = (results['duplicate_clues_count'] / results['unique_clues']) * 100
            print(f"Percentage of clues that are duplicates: {duplicate_percentage:.2f}%")
            
            common_duplicates = sorted(results['duplicates'].items(), key=lambda x: x[1], reverse=True)
            print("\nMost frequently repeated clues:")
            for clue, count in common_duplicates[:10]: 
                print(f"  \"{clue}\" appears {count} times")
            
            if verbose:
                print("\nDetailed duplicate clue information:")
                for clue, instances in results['duplicate_details'].items():
                    print(f"\nClue: \"{clue}\" appears {len(instances)} times:")
                    for word, puzzle_id, puzzle_path, puzzle_size in instances:
                        print(f"  - Word: {word}, Puzzle ID: {puzzle_id}, Size: {puzzle_size}, Path: {puzzle_path}")
        else:
            print("No duplicate clues found across puzzles!")
        
        print(f"\n--- Cross-Size Clue Analysis ---")
        print(f"Clues appearing in multiple size categories: {results['cross_size_duplicates_count']}")
        
        if results['cross_size_duplicates_count'] > 0:
            print(f"\nSizes analyzed: {', '.join(results['sizes'])}")
            
            sorted_duplicates = sorted(
                results['cross_size_duplicates'].items(), 
                key=lambda x: (len(x[1]), x[0]), 
                reverse=True
            )
            
            print("\nClues that appear across different size puzzles:")
            for clue, sizes in sorted_duplicates[:10]:
                print(f"  \"{clue}\" appears in {len(sizes)} different sizes: {', '.join(sorted(sizes))}")

def save_results(results, output_file):
    with open(output_file, 'w') as f:
        f.write("===== Crossword Puzzle Clue Analysis =====\n")
        
        if results.get("cross_size_only", False):
            f.write(f"\n--- Cross-Size Clue Analysis ---\n")
            f.write(f"Total puzzles analyzed: {results['total_puzzles']}\n")
            f.write(f"Total clues found: {results['total_clues']}\n")
            f.write(f"Clues appearing in multiple size categories: {results['cross_size_duplicates_count']}\n")
            
            if results['cross_size_duplicates_count'] > 0:
                f.write(f"\nSizes analyzed: {', '.join(results['sizes'])}\n")
                
                sorted_duplicates = sorted(
                    results['cross_size_duplicates'].items(), 
                    key=lambda x: (len(x[1]), x[0]), 
                    reverse=True
                )
                
                f.write("\nClues that appear across different size puzzles:\n")
                for clue, sizes in sorted_duplicates:
                    f.write(f"  \"{clue}\" appears in {len(sizes)} different sizes: {', '.join(sorted(sizes))}\n")
                
                f.write("\nDetailed cross-size duplicate clue information:\n")
                for clue, instances in results['cross_size_details'].items():
                    sizes = results['cross_size_duplicates'][clue]
                    f.write(f"\nClue: \"{clue}\" appears in {len(sizes)} sizes ({', '.join(sorted(sizes))}):\n")
                    
                    by_size = defaultdict(list)
                    for word, puzzle_id, puzzle_path, size in instances:
                        by_size[size].append((word, puzzle_id, puzzle_path))
                    
                    for size in sorted(by_size.keys()):
                        f.write(f"  Size {size}:\n")
                        for word, puzzle_id, puzzle_path in by_size[size]:
                            f.write(f"    - Word: {word}, Puzzle ID: {puzzle_id}, Path: {puzzle_path}\n")
            else:
                f.write("No clues found that appear across different size puzzles!\n")
            return
        
        if not results["cross_size"]:
            for size, size_results in results["results_by_size"].items():
                f.write(f"\n--- Results for {size} puzzles ---\n")
                f.write(f"Total puzzles analyzed: {size_results['total_puzzles']}\n")
                f.write(f"Total clues found: {size_results['total_clues']}\n")
                f.write(f"Unique clues: {size_results['unique_clues']}\n")
                f.write(f"Clues appearing in multiple puzzles: {size_results['duplicate_clues_count']}\n")
                
                if size_results['duplicate_clues_count'] > 0:
                    duplicate_percentage = (size_results['duplicate_clues_count'] / size_results['unique_clues']) * 100
                    f.write(f"Percentage of clues that are duplicates: {duplicate_percentage:.2f}%\n")
                    
                    common_duplicates = sorted(size_results['duplicates'].items(), key=lambda x: x[1], reverse=True)
                    f.write("\nMost frequently repeated clues:\n")
                    for clue, count in common_duplicates:
                        f.write(f"  \"{clue}\" appears {count} times\n")
                    
                    f.write("\nDetailed duplicate clue information:\n")
                    for clue, instances in size_results['duplicate_details'].items():
                        f.write(f"\nClue: \"{clue}\" appears {len(instances)} times:\n")
                        for word, puzzle_id, puzzle_path in instances:
                            f.write(f"  - Word: {word}, Puzzle ID: {puzzle_id}, Path: {puzzle_path}\n")
                else:
                    f.write("No duplicate clues found across puzzles!\n")
            
            f.write(f"\n--- Cross-Size Clue Analysis ---\n")
            f.write(f"Clues appearing in multiple size categories: {results['cross_size_duplicates_count']}\n")
            
            if results['cross_size_duplicates_count'] > 0:
                f.write(f"\nSizes analyzed: {', '.join(results['sizes'])}\n")
                
                sorted_duplicates = sorted(
                    results['cross_size_duplicates'].items(), 
                    key=lambda x: (len(x[1]), x[0]), 
                    reverse=True
                )
                
                f.write("\nClues that appear across different size puzzles:\n")
                for clue, sizes in sorted_duplicates:
                    f.write(f"  \"{clue}\" appears in {len(sizes)} different sizes: {', '.join(sorted(sizes))}\n")
                
                f.write("\nDetailed cross-size duplicate clue information:\n")
                for clue, instances in results['cross_size_details'].items():
                    sizes = results['cross_size_duplicates'][clue]
                    f.write(f"\nClue: \"{clue}\" appears in {len(sizes)} sizes ({', '.join(sorted(sizes))}):\n")
                    
                    by_size = defaultdict(list)
                    for word, puzzle_id, puzzle_path, size in instances:
                        by_size[size].append((word, puzzle_id, puzzle_path))
                    
                    for size in sorted(by_size.keys()):
                        f.write(f"  Size {size}:\n")
                        for word, puzzle_id, puzzle_path in by_size[size]:
                            f.write(f"    - Word: {word}, Puzzle ID: {puzzle_id}, Path: {puzzle_path}\n")
        else:
            f.write(f"Total puzzles analyzed (all sizes): {results['total_puzzles']}\n")
            f.write(f"Total clues found: {results['total_clues']}\n")
            f.write(f"Unique clues: {results['unique_clues']}\n")
            f.write(f"Clues appearing in multiple puzzles: {results['duplicate_clues_count']}\n")
            
            if results['duplicate_clues_count'] > 0:
                duplicate_percentage = (results['duplicate_clues_count'] / results['unique_clues']) * 100
                f.write(f"Percentage of clues that are duplicates: {duplicate_percentage:.2f}%\n")
                
                common_duplicates = sorted(results['duplicates'].items(), key=lambda x: x[1], reverse=True)
                f.write("\nMost frequently repeated clues:\n")
                for clue, count in common_duplicates:
                    f.write(f"  \"{clue}\" appears {count} times\n")
                
                f.write("\nDetailed duplicate clue information:\n")
                for clue, instances in results['duplicate_details'].items():
                    f.write(f"\nClue: \"{clue}\" appears {len(instances)} times:\n")
                    for word, puzzle_id, puzzle_path, puzzle_size in instances:
                        f.write(f"  - Word: {word}, Puzzle ID: {puzzle_id}, Size: {puzzle_size}, Path: {puzzle_path}\n")
                
                f.write(f"\n--- Cross-Size Clue Analysis ---\n")
                f.write(f"Clues appearing in multiple size categories: {results['cross_size_duplicates_count']}\n")
                
                if results['cross_size_duplicates_count'] > 0:
                    f.write(f"\nSizes analyzed: {', '.join(results['sizes'])}\n")
                    
                    sorted_duplicates = sorted(
                        results['cross_size_duplicates'].items(), 
                        key=lambda x: (len(x[1]), x[0]), 
                        reverse=True
                    )
                    
                    f.write("\nClues that appear across different size puzzles:\n")
                    for clue, sizes in sorted_duplicates:
                        f.write(f"  \"{clue}\" appears in {len(sizes)} different sizes: {', '.join(sorted(sizes))}\n")
            else:
                f.write("No duplicate clues found across puzzles!\n")
    
    print(f"\nResults saved to {output_file}")

def main():
    args = parse_arguments()
    
    puzzle_files = find_puzzle_files(args.data_dirs)
    print(f"Found {len(puzzle_files)} total puzzle files to analyze.")
    
    results = analyze_clues(
        puzzle_files, 
        cross_size=args.cross_size, 
        cross_size_only=args.cross_size_only,
        verbose=args.verbose
    )
    print_results(results, args.verbose)
    
    if args.output:
        save_results(results, args.output)

if __name__ == "__main__":
    main()