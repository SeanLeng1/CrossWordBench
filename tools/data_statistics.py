import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import sys
import statistics
from datetime import datetime

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(description="Analyze crossword puzzle statistics")
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
        help="Print detailed information about puzzles"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Output file to save results (optional)"
    )
    parser.add_argument(
        "--analyze_by_size", 
        action="store_true",
        help="Analyze statistics separately for each puzzle size"
    )
    parser.add_argument(
        "--top_n", 
        type=int,
        default=10,
        help="Number of top items to display in reports (default: 10)"
    )
    return parser.parse_args()

def find_puzzle_files(data_dirs):
    """
    Find all puzzle_state.json files in the given directories.
    
    Args:
        data_dirs: List of directory paths to search
        
    Returns:
        List of Path objects pointing to puzzle_state.json files
    """
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

def extract_puzzle_data(puzzle_file):
    """
    Extract relevant data from a puzzle_state.json file.
    
    Args:
        puzzle_file: Path to a puzzle_state.json file
        
    Returns:
        Dictionary containing processed puzzle data or None if error
    """
    with open(puzzle_file, 'r') as f:
        try:
            puzzle_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {puzzle_file}")
            return None
    
    wordlist = puzzle_data.get('wordlist', [])
    meta_data = puzzle_data.get('meta_data', {})
    grid = puzzle_data.get('grid', [])
    
    puzzle_id = meta_data.get('id', 'unknown')
    created_at = meta_data.get('created_at', None)
    grid_size = meta_data.get('grid_size', None)
    puzzle_path = str(puzzle_file.parent)
    
    # Determine puzzle size from metadata, path, or grid
    size_match = grid_size
    
    if not size_match:
        for part in puzzle_path.split(os.sep):
            if "x" in part and part.replace("x", "").isdigit():
                size_match = part
                break
    
    if not size_match and grid:
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        size_match = f"{width}x{height}"
    
    puzzle_size = size_match if size_match else "unknown_size"
    
    # Extract words and clues
    words_and_clues = []
    for word_entry in wordlist:
        if len(word_entry) > 1:
            # format is: [word, clue, y_pos, x_pos, orientation]
            word = word_entry[0]
            clue = word_entry[1]
            y_pos = word_entry[2] if len(word_entry) > 2 else None
            x_pos = word_entry[3] if len(word_entry) > 3 else None
            orientation = word_entry[4] if len(word_entry) > 4 else "unknown"
            words_and_clues.append({
                "word": word,
                "clue": clue,
                "y_pos": y_pos,
                "x_pos": x_pos,
                "orientation": orientation,
                "length": len(word)
            })
    
    # Count blocked cells and analyze grid if available
    blocked_cells = 0
    total_cells = 0
    filled_cells = 0
    empty_cells = 0
    
    # Dictionary to count letter frequencies in the grid
    letter_frequency = Counter()
    
    if grid:
        for row in grid:
            for cell in row:
                total_cells += 1
                if cell == "-":
                    blocked_cells += 1
                elif cell == "":
                    empty_cells += 1
                else:
                    filled_cells += 1
                    letter_frequency[cell] += 1
    
    # Parse creation date if available
    creation_date = None
    if created_at:
        try:
            creation_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            pass
    
    return {
        "puzzle_id": puzzle_id,
        "puzzle_size": puzzle_size,
        "puzzle_path": puzzle_path,
        "creation_date": creation_date,
        "words_and_clues": words_and_clues,
        "grid_stats": {
            "total_cells": total_cells,
            "blocked_cells": blocked_cells,
            "filled_cells": filled_cells,
            "empty_cells": empty_cells,
            "blocked_percentage": (blocked_cells / total_cells * 100) if total_cells > 0 else 0,
            "letter_frequency": dict(letter_frequency)
        }
    }

def analyze_puzzles(puzzle_files, analyze_by_size=False, top_n=10):
    """
    Analyze puzzle files and extract meaningful statistics.
    
    Args:
        puzzle_files: List of paths to puzzle_state.json files
        analyze_by_size: Whether to analyze puzzles separately by size
        top_n: Number of top items to show in rankings
        
    Returns:
        Dictionary containing comprehensive statistics
    """
    print(f"Analyzing {len(puzzle_files)} puzzles for statistics...")
    
    # Initialize counters and collections
    all_puzzles = []
    all_words = []
    all_clues = []
    word_lengths = []
    words_per_puzzle = []
    puzzles_by_size = defaultdict(list)
    puzzles_by_month = defaultdict(list)
    
    # Process each puzzle file
    for puzzle_file in puzzle_files:
        puzzle_data = extract_puzzle_data(puzzle_file)
        if not puzzle_data:
            continue
        
        all_puzzles.append(puzzle_data)
        puzzles_by_size[puzzle_data["puzzle_size"]].append(puzzle_data)
        
        # Track creation dates by month if available
        if puzzle_data["creation_date"]:
            month_key = puzzle_data["creation_date"].strftime("%Y-%m")
            puzzles_by_month[month_key].append(puzzle_data)
        
        # Add words and clues to global collections
        puzzle_words = []
        for item in puzzle_data["words_and_clues"]:
            all_words.append(item["word"])
            all_clues.append(item["clue"])
            word_lengths.append(item["length"])
            puzzle_words.append(item["word"])
        
        words_per_puzzle.append(len(puzzle_words))
    
    # Generate overall statistics
    overall_stats = {
        "total_puzzles": len(all_puzzles),
        "total_words": len(all_words),
        "total_clues": len(all_clues),
        "unique_words": len(set(all_words)),
        "unique_clues": len(set(all_clues)),
        "puzzle_sizes": dict(Counter([p["puzzle_size"] for p in all_puzzles])),
        "words_per_puzzle": {
            "min": min(words_per_puzzle) if words_per_puzzle else 0,
            "max": max(words_per_puzzle) if words_per_puzzle else 0,
            "mean": statistics.mean(words_per_puzzle) if words_per_puzzle else 0,
            "median": statistics.median(words_per_puzzle) if words_per_puzzle else 0
        },
        "word_length": {
            "min": min(word_lengths) if word_lengths else 0,
            "max": max(word_lengths) if word_lengths else 0,
            "mean": statistics.mean(word_lengths) if word_lengths else 0,
            "median": statistics.median(word_lengths) if word_lengths else 0,
            "distribution": dict(Counter(word_lengths))
        },
        "most_common_words": dict(Counter(all_words).most_common(top_n)),
        "most_common_clues": dict(Counter(all_clues).most_common(top_n)),
        "clue_reuse": {
            "total_reused_clues": len([c for c, count in Counter(all_clues).items() if count > 1]),
            "max_reuse": max(Counter(all_clues).values()) if all_clues else 0
        },
        "word_reuse": {
            "total_reused_words": len([w for w, count in Counter(all_words).items() if count > 1]),
            "max_reuse": max(Counter(all_words).values()) if all_words else 0
        },
        "time_distribution": {
            "puzzles_by_month": {k: len(v) for k, v in sorted(puzzles_by_month.items())}
        },
        "grid_stats": {
            "avg_blocked_cells_percentage": statistics.mean([p["grid_stats"]["blocked_percentage"] for p in all_puzzles if p["grid_stats"]["total_cells"] > 0]) if all_puzzles else 0,
            "letter_frequency": dict(Counter([letter for p in all_puzzles for letter, count in p["grid_stats"]["letter_frequency"].items() for _ in range(count)]))
        }
    }
    
    # If requested, generate statistics by puzzle size
    size_stats = {}
    if analyze_by_size:
        for size, puzzles in puzzles_by_size.items():
            size_words = []
            size_clues = []
            size_word_lengths = []
            size_words_per_puzzle = []
            
            for puzzle in puzzles:
                puzzle_word_count = 0
                for item in puzzle["words_and_clues"]:
                    size_words.append(item["word"])
                    size_clues.append(item["clue"])
                    size_word_lengths.append(item["length"])
                    puzzle_word_count += 1
                
                size_words_per_puzzle.append(puzzle_word_count)
            
            size_stats[size] = {
                "total_puzzles": len(puzzles),
                "total_words": len(size_words),
                "total_clues": len(size_clues),
                "unique_words": len(set(size_words)),
                "unique_clues": len(set(size_clues)),
                "words_per_puzzle": {
                    "min": min(size_words_per_puzzle) if size_words_per_puzzle else 0,
                    "max": max(size_words_per_puzzle) if size_words_per_puzzle else 0,
                    "mean": statistics.mean(size_words_per_puzzle) if size_words_per_puzzle else 0,
                    "median": statistics.median(size_words_per_puzzle) if size_words_per_puzzle else 0
                },
                "word_length": {
                    "min": min(size_word_lengths) if size_word_lengths else 0,
                    "max": max(size_word_lengths) if size_word_lengths else 0,
                    "mean": statistics.mean(size_word_lengths) if size_word_lengths else 0,
                    "median": statistics.median(size_word_lengths) if size_word_lengths else 0,
                    "distribution": dict(Counter(size_word_lengths))
                },
                "most_common_words": dict(Counter(size_words).most_common(top_n)),
                "most_common_clues": dict(Counter(size_clues).most_common(top_n)),
                "clue_reuse": {
                    "total_reused_clues": len([c for c, count in Counter(size_clues).items() if count > 1]),
                    "max_reuse": max(Counter(size_clues).values()) if size_clues else 0
                },
                "word_reuse": {
                    "total_reused_words": len([w for w, count in Counter(size_words).items() if count > 1]),
                    "max_reuse": max(Counter(size_words).values()) if size_words else 0
                },
                "grid_stats": {
                    "avg_blocked_cells_percentage": statistics.mean([p["grid_stats"]["blocked_percentage"] for p in puzzles if p["grid_stats"]["total_cells"] > 0]) if puzzles else 0,
                    "letter_frequency": dict(Counter([letter for p in puzzles for letter, count in p["grid_stats"]["letter_frequency"].items() for _ in range(count)]))
                }
            }
    
    # Find clues used for different words
    clue_to_words = defaultdict(set)
    for puzzle in all_puzzles:
        for item in puzzle["words_and_clues"]:
            clue_to_words[item["clue"]].add(item["word"])
    
    ambiguous_clues = {clue: list(words) for clue, words in clue_to_words.items() if len(words) > 1}
    
    # Find words with multiple clues
    word_to_clues = defaultdict(set)
    for puzzle in all_puzzles:
        for item in puzzle["words_and_clues"]:
            word_to_clues[item["word"]].add(item["clue"])
    
    multi_clued_words = {word: list(clues) for word, clues in word_to_clues.items() if len(clues) > 1}
    
    # Add these to overall stats
    overall_stats["clue_word_relation"] = {
        "ambiguous_clues_count": len(ambiguous_clues),
        "top_ambiguous_clues": dict(sorted(
            {clue: len(words) for clue, words in ambiguous_clues.items()}.items(),
            key=lambda x: x[1], reverse=True
        )[:top_n]),
        "multi_clued_words_count": len(multi_clued_words),
        "top_multi_clued_words": dict(sorted(
            {word: len(clues) for word, clues in multi_clued_words.items()}.items(),
            key=lambda x: x[1], reverse=True
        )[:top_n])
    }
    
    return {
        "overall_stats": overall_stats,
        "size_stats": size_stats if analyze_by_size else {},
        "ambiguous_clues": ambiguous_clues,
        "multi_clued_words": multi_clued_words
    }

def print_results(results, verbose=False, top_n=10):
    """
    Print the analysis results in a readable format.
    
    Args:
        results: Dictionary containing analysis results
        verbose: Whether to print detailed information
        top_n: Number of top items to show in rankings
    """
    overall_stats = results["overall_stats"]
    size_stats = results["size_stats"]
    
    print("\n===== Crossword Puzzle Statistics Analysis =====")
    
    # Print overall statistics
    print("\n--- Overall Statistics ---")
    print(f"Total puzzles analyzed: {overall_stats['total_puzzles']}")
    print(f"Total words: {overall_stats['total_words']}")
    print(f"Unique words: {overall_stats['unique_words']} ({overall_stats['unique_words'] / overall_stats['total_words'] * 100:.2f}% of total)")
    print(f"Total clues: {overall_stats['total_clues']}")
    print(f"Unique clues: {overall_stats['unique_clues']} ({overall_stats['unique_clues'] / overall_stats['total_clues'] * 100:.2f}% of total)")
    
    print(f"\nPuzzle sizes distribution:")
    for size, count in sorted(overall_stats['puzzle_sizes'].items()):
        print(f"  {size}: {count} puzzles ({count / overall_stats['total_puzzles'] * 100:.2f}%)")
    
    print("\nWords per puzzle statistics:")
    print(f"  Min: {overall_stats['words_per_puzzle']['min']}")
    print(f"  Max: {overall_stats['words_per_puzzle']['max']}")
    print(f"  Mean: {overall_stats['words_per_puzzle']['mean']:.2f}")
    print(f"  Median: {overall_stats['words_per_puzzle']['median']}")
    
    print("\nWord length statistics:")
    print(f"  Min length: {overall_stats['word_length']['min']}")
    print(f"  Max length: {overall_stats['word_length']['max']}")
    print(f"  Mean length: {overall_stats['word_length']['mean']:.2f}")
    print(f"  Median length: {overall_stats['word_length']['median']}")
    
    print("\nWord length distribution:")
    for length, count in sorted(overall_stats['word_length']['distribution'].items()):
        print(f"  {length} letters: {count} words ({count / overall_stats['total_words'] * 100:.2f}%)")
    
    print(f"\nGrid statistics:")
    print(f"  Average blocked cells percentage: {overall_stats['grid_stats']['avg_blocked_cells_percentage']:.2f}%")
    
    print(f"\nTop {top_n} most common letters in grids:")
    most_common_letters = dict(Counter(overall_stats['grid_stats']['letter_frequency']).most_common(top_n))
    total_letters = sum(overall_stats['grid_stats']['letter_frequency'].values())
    for letter, count in most_common_letters.items():
        print(f"  '{letter}': {count} times ({count/total_letters*100:.2f}% of all letters)")
    
    print(f"\nClue reuse statistics:")
    print(f"  Clues appearing multiple times: {overall_stats['clue_reuse']['total_reused_clues']} ({overall_stats['clue_reuse']['total_reused_clues'] / overall_stats['unique_clues'] * 100:.2f}% of unique clues)")
    print(f"  Maximum times a clue is reused: {overall_stats['clue_reuse']['max_reuse']}")
    
    print(f"\nWord reuse statistics:")
    print(f"  Words appearing multiple times: {overall_stats['word_reuse']['total_reused_words']} ({overall_stats['word_reuse']['total_reused_words'] / overall_stats['unique_words'] * 100:.2f}% of unique words)")
    print(f"  Maximum times a word is reused: {overall_stats['word_reuse']['max_reuse']}")
    
    print(f"\nClue-word relationship:")
    print(f"  Clues used for multiple different words: {overall_stats['clue_word_relation']['ambiguous_clues_count']}")
    print(f"  Words with multiple different clues: {overall_stats['clue_word_relation']['multi_clued_words_count']}")
    
    print(f"\nTop {top_n} most common words:")
    for word, count in list(overall_stats['most_common_words'].items())[:top_n]:
        print(f"  \"{word}\": {count} times")
    
    print(f"\nTop {top_n} most common clues:")
    for clue, count in list(overall_stats['most_common_clues'].items())[:top_n]:
        print(f"  \"{clue}\": {count} times")

    print(f"\nTop {top_n} most ambiguous clues (used for different words):")
    for clue, count in list(overall_stats['clue_word_relation']['top_ambiguous_clues'].items())[:top_n]:
        print(f"  \"{clue}\": used for {count} different words")
    
    print(f"\nTop {top_n} words with most different clues:")
    for word, count in list(overall_stats['clue_word_relation']['top_multi_clued_words'].items())[:top_n]:
        print(f"  \"{word}\": has {count} different clues")
    
    # Output puzzle creation timeline if available
    if overall_stats['time_distribution']['puzzles_by_month']:
        print("\nPuzzle creation timeline:")
        for month, count in sorted(overall_stats['time_distribution']['puzzles_by_month'].items()):
            print(f"  {month}: {count} puzzles")
    
    # If we have statistics by size, print them
    if size_stats:
        for size, stats in sorted(size_stats.items()):
            print(f"\n\n--- Statistics for {size} puzzles ---")
            print(f"Total puzzles: {stats['total_puzzles']}")
            print(f"Total words: {stats['total_words']}")
            print(f"Unique words: {stats['unique_words']} ({stats['unique_words'] / stats['total_words'] * 100:.2f}% of total)")
            print(f"Total clues: {stats['total_clues']}")
            print(f"Unique clues: {stats['unique_clues']} ({stats['unique_clues'] / stats['total_clues'] * 100:.2f}% of total)")
            
            print("\nWords per puzzle statistics:")
            print(f"  Min: {stats['words_per_puzzle']['min']}")
            print(f"  Max: {stats['words_per_puzzle']['max']}")
            print(f"  Mean: {stats['words_per_puzzle']['mean']:.2f}")
            print(f"  Median: {stats['words_per_puzzle']['median']}")
            
            print("\nWord length statistics:")
            print(f"  Min length: {stats['word_length']['min']}")
            print(f"  Max length: {stats['word_length']['max']}")
            print(f"  Mean length: {stats['word_length']['mean']:.2f}")
            print(f"  Median length: {stats['word_length']['median']}")
            
            print(f"\nGrid statistics:")
            print(f"  Average blocked cells percentage: {stats['grid_stats']['avg_blocked_cells_percentage']:.2f}%")
            
            print(f"\nTop {top_n} most common letters in {size} grids:")
            most_common_letters = dict(Counter(stats['grid_stats']['letter_frequency']).most_common(top_n))
            total_letters = sum(stats['grid_stats']['letter_frequency'].values())
            for letter, count in most_common_letters.items():
                print(f"  '{letter}': {count} times ({count/total_letters*100:.2f}% of all letters)")
            
            print(f"\nClue reuse statistics:")
            print(f"  Clues appearing multiple times: {stats['clue_reuse']['total_reused_clues']} ({stats['clue_reuse']['total_reused_clues'] / stats['unique_clues'] * 100:.2f}% of unique clues)")
            print(f"  Maximum times a clue is reused: {stats['clue_reuse']['max_reuse']}")
            
            print(f"\nWord reuse statistics:")
            print(f"  Words appearing multiple times: {stats['word_reuse']['total_reused_words']} ({stats['word_reuse']['total_reused_words'] / stats['unique_words'] * 100:.2f}% of unique words)")
            print(f"  Maximum times a word is reused: {stats['word_reuse']['max_reuse']}")
            
            print(f"\nTop {top_n} most common words for {size}:")
            for word, count in list(stats['most_common_words'].items())[:top_n]:
                print(f"  \"{word}\": {count} times")
            
            print(f"\nTop {top_n} most common clues for {size}:")
            for clue, count in list(stats['most_common_clues'].items())[:top_n]:
                print(f"  \"{clue}\": {count} times")
    
    # Print detailed information if verbose mode is enabled
    if verbose:
        print("\n\n--- Detailed Ambiguous Clues ---")
        print("Clues that are used for multiple different words:")
        sorted_ambiguous = sorted(results["ambiguous_clues"].items(), key=lambda x: len(x[1]), reverse=True)
        for clue, words in sorted_ambiguous[:20]:  # Limit to top 20 in verbose mode
            print(f"\nClue: \"{clue}\" is used for {len(words)} different words:")
            for word in sorted(words):
                print(f"  - {word}")
        
        print("\n\n--- Detailed Multi-Clued Words ---")
        print("Words that have multiple different clues:")
        sorted_multi_clued = sorted(results["multi_clued_words"].items(), key=lambda x: len(x[1]), reverse=True)
        for word, clues in sorted_multi_clued[:20]:  # Limit to top 20 in verbose mode
            print(f"\nWord: \"{word}\" has {len(clues)} different clues:")
            for clue in sorted(clues):
                print(f"  - {clue}")

def main():
    args = parse_arguments()
    
    puzzle_files = find_puzzle_files(args.data_dirs)
    print(f"Found {len(puzzle_files)} total puzzle files to analyze.")
    
    results = analyze_puzzles(
        puzzle_files, 
        analyze_by_size=args.analyze_by_size,
        top_n=args.top_n
    )
    
    print_results(results, args.verbose, args.top_n)
    
if __name__ == "__main__":
    main()