import json
import os
import re
from typing import List, Tuple, Dict
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

def clean_word(word):
    """Clean a word by removing spaces, punctuation, and special characters"""
    # Remove any non-Chinese/non-English characters (keeping only letters, numbers, and Chinese characters)
    cleaned = re.sub(r'[^\u4e00-\u9fff\w]', '', word)
    # Remove all whitespace
    cleaned = ''.join(cleaned.split())
    return cleaned

def get_chinese_char_count(text):
    """Count actual Chinese/non-Chinese characters in a string"""
    return sum(1 for c in text if not c.isspace())

def parse_crossword_json(json_data):
    answer_grid = json_data["answer"]
    h_clues = json_data["tip"]["h"]
    v_clues = json_data["tip"]["v"]
    word_clue_pairs = []
    
    # For Chinese text, 50 characters is more appropriate than 50 words
    MAX_CLUE_LENGTH = 60

    for row_idx, row in enumerate(answer_grid):
        row_num = row_idx + 1
        current_word = ""
        start_col = None
        
        for col_idx, char in enumerate(row):
            col_num = col_idx + 1
            
            if char != "0":
                if start_col is None:
                    start_col = col_num
                current_word += char
            
            if (char == "0" or col_idx == len(row) - 1) and current_word:
                position_key = f"{row_num}-{start_col}"
                if position_key in h_clues:
                    clue = h_clues[position_key].strip()
                    # Use proper character counting for Chinese
                    char_count = get_chinese_char_count(clue)
                    if char_count <= MAX_CLUE_LENGTH:
                        cleaned_word = clean_word(current_word)
                        if cleaned_word:
                            word_clue_pairs.append((cleaned_word, clue))
                current_word = ""
                start_col = None

    num_cols = len(answer_grid[0])
    for col_idx in range(num_cols):
        current_word = ""
        start_row = None
        
        for row_idx, row in enumerate(answer_grid):
            col_num = col_idx + 1
            char = row[col_idx]
            
            if char != "0":
                if start_row is None:
                    start_row = row_idx + 1
                current_word += char
            
            if (char == "0" or row_idx == len(answer_grid) - 1) and current_word:
                position_key = f"{start_row}-{col_num}"
                if position_key in v_clues:
                    clue = v_clues[position_key].strip()
                    if clue[-1] == '。':
                        clue = clue[:-1]
                    # Skip if clue is too long
                    if len(clue) <= MAX_CLUE_LENGTH:
                        cleaned_word = clean_word(current_word)
                        if cleaned_word:
                            word_clue_pairs.append((cleaned_word, clue))
                current_word = ""
                start_row = None

    return word_clue_pairs

def process_folder(folder_path, output_file = 'chinese_words.txt'):
    all_pairs = set()
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    for filename in tqdm(json_files, desc="Processing JSON files"):
        try:
            with (Path(folder_path) / filename).open('r', encoding='utf-8') as f:
                json_data = json.load(f)
            pairs = parse_crossword_json(json_data)
            all_pairs.update(pairs)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, clue in sorted(all_pairs):
            f.write(f"{word} {clue}\n")
    
    return all_pairs

def analyze_statistics(word_clue_pairs, output_file = 'chinese_meta.json'):
    word_to_clues = defaultdict(list)
    for word, clue in word_clue_pairs:
        word_to_clues[word].append(clue)
    
    word_lengths = [len(word) for word in word_to_clues.keys()]
    clue_lengths = [len(clue) for _, clue in word_clue_pairs]
    
    max_clues_count = max(len(clues) for clues in word_to_clues.values())
    words_with_max_clues = [
        (word, len(clues))
        for word, clues in word_to_clues.items()
        if len(clues) == max_clues_count
    ]
    
    clues_per_word_dist = defaultdict(int)
    for clues in word_to_clues.values():
        clues_per_word_dist[len(clues)] += 1
    
    stats = {
        "total_pairs": len(word_clue_pairs),
        "unique_words": len(word_to_clues),
        "average_clues_per_word": len(word_clue_pairs) / len(word_to_clues),
        "max_clues_for_word": max_clues_count,
        "words_with_max_clues": {
            word: word_to_clues[word]
            for word, _ in words_with_max_clues
        },
        "word_length_stats": {
            "min": min(word_lengths),
            "max": max(word_lengths),
            "average": sum(word_lengths) / len(word_lengths)
        },
        "clue_length_stats": {
            "min": min(clue_lengths),
            "max": max(clue_lengths),
            "average": sum(clue_lengths) / len(clue_lengths)
        },
        "distribution": {
            "single_clue": clues_per_word_dist[1],
            "multiple_clues": sum(count for clues, count in clues_per_word_dist.items() if clues > 1),
            "clues_per_word": dict(sorted(clues_per_word_dist.items()))
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print("\nKey Statistics:")
    print(f"Total word-clue pairs: {stats['total_pairs']}")
    print(f"Unique words: {stats['unique_words']}")
    print(f"Average clues per word: {stats['average_clues_per_word']:.2f}")
    print(f"Words with single clue: {stats['distribution']['single_clue']}")
    print(f"Words with multiple clues: {stats['distribution']['multiple_clues']}")
    print(f"Most clues for a word: {stats['max_clues_for_word']}")
    
    return stats

def main():
    folder_path = "chinese"  
    word_clue_pairs = process_folder(folder_path)
    analyze_statistics(word_clue_pairs)

if __name__ == "__main__":
    main()