import csv
from tqdm import tqdm
from collections import defaultdict
import json
import re
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List
import inflect
import pyinflect
import spacy
from spacy.cli import download
from nltk.corpus import wordnet

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
p = inflect.engine()

def dictionary(words, max_workers=4):
    """
    Get valid words and their definitions using NLTK's WordNet.
    Returns a set of (word, clue) pairs where each definition becomes a separate clue.
    Words are converted to singular form and present tense.
    
    Parameters:
    - words: list of words to look up
    - max_workers: number of concurrent workers for processing
    
    Note: First-time users need to download WordNet:
    >>> import nltk
    >>> nltk.download('wordnet')
    """
    word_clue_pairs = set()
    
    def get_word_definitions(word):
        try:
            # Get synsets (word senses) from WordNet
            synsets = wordnet.synsets(word)
            
            if not synsets:
                # Check for British spelling variants (like "favour")
                if word.endswith('our'):
                    american_spelling = word[:-3] + 'or'
                    synsets = wordnet.synsets(american_spelling)
                
                # Try other common variations if needed
                if not synsets and word.endswith('re'):
                    american_spelling = word[:-2] + 'er'
                    synsets = wordnet.synsets(american_spelling)
            
            if not synsets:
                print(f"No definitions found for word: {word}")
                return word, False, []
            
            # Collect all definitions
            definitions = []
            for synset in synsets:
                def_text = synset.definition().strip()
                if def_text:
                    if def_text.endswith('.'):
                        def_text = def_text[:-1]
                    
                    # Only add if we don't already have this definition
                    if def_text not in definitions:
                        definitions.append(def_text)
            
            return word, True, definitions
            
        except Exception as e:
            print(f"Error getting definition for '{word}': {str(e)}")
            return word, False, []
    
    print(f"Checking {len(words)} words and getting their definitions using WordNet...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_word = {
            executor.submit(get_word_definitions, word): word 
            for word in words
        }
        
        for future in tqdm(as_completed(future_to_word), total=len(words), desc="Dictionary Processing"):
            word, is_valid, definitions = future.result()
            if is_valid and definitions:
                for definition in definitions:
                    word_clue_pairs.add((word.upper(), definition))

    print(f"Found {len(word_clue_pairs)} word-clue pairs")
    return word_clue_pairs

@torch.inference_mode()
def encode_clues_batch(all_clues, batch_size=32):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model = model.to('cuda', dtype=torch.float32)
    embedding_dim = model.get_sentence_embedding_dimension()
    all_embeddings = torch.zeros((len(all_clues), embedding_dim), device='cuda', dtype=torch.float32)
    for i in tqdm(range(0, len(all_clues), batch_size), desc="Encoding batches", total=len(all_clues) // batch_size):
        batch = all_clues[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        all_embeddings[i:i + len(batch)] = batch_embeddings
    del model
    return all_embeddings

def filter_similar_clues_batch(word_clue_pairs, similarity_threshold=0.9, encoding_batch_size=8192):
    """
    Filters out near-duplicate clues *within* each word by computing
    a local NxN similarity matrix on GPU. This avoids building a 
    global FAISS index and searching it repeatedly.
    
    word_clue_pairs : List of (word, clue) pairs
    similarity_threshold : float in [0,1], 
        the threshold above which two clues are considered duplicates
    encoding_batch_size : batch size for encoding clues 
        (not for the similarity part, which is local per word)
    """
    word_groups = defaultdict(list)
    for word, clue in word_clue_pairs:
        word_groups[word].append(clue)
    
    all_clues = []
    word_slice_map = {}
    running_index = 0
    
    for word, clues in word_groups.items():
        word_slice_map[word] = (running_index, running_index + len(clues))
        all_clues.extend(clues)
        running_index += len(clues)
    
    print("Encoding all clues...")
    all_embeddings = encode_clues_batch(all_clues, batch_size=encoding_batch_size)
    
    # filter duplicates within each word locally
    filtered_pairs = set()
    print("Filtering duplicates for each word...")

    for word in tqdm(word_groups.keys(), desc="Per-word filtering"):
        start_idx, end_idx = word_slice_map[word]
        word_embeddings = all_embeddings[start_idx:end_idx]  
        
        if word_embeddings.shape[0] == 1:
            filtered_pairs.add((word, word_groups[word][0]))
            continue
        
        # cosine similarity
        normalized = torch.nn.functional.normalize(word_embeddings, dim=1)
        sim_matrix = torch.matmul(normalized, normalized.T)  
        
        keep_mask = torch.ones(word_embeddings.size(0), dtype=torch.bool, device=word_embeddings.device)

        for i in range(word_embeddings.size(0)):
            if keep_mask[i]:
                duplicates_i = (sim_matrix[i] > similarity_threshold)
                keep_mask[duplicates_i] = False
                keep_mask[i] = True
        
        keep_indices = torch.nonzero(keep_mask, as_tuple=True)[0].cpu().tolist()
        
        for idx in keep_indices:
            filtered_pairs.add((word, word_groups[word][idx]))

    return filtered_pairs

def write_output_file(word_clue_pairs, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for word, clue in sorted(word_clue_pairs):  # sort for consistency
            outfile.write(f'{word} {clue}\n')

def analyze_statistics(word_clue_pairs, output_file = 'eng_meta.json'):
    # group clues by word
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
            "words_with_single_clue": clues_per_word_dist[1],
            "words_with_multiple_clues": sum(count for clues, count in clues_per_word_dist.items() if clues > 1),
            "clues_per_word_distribution": dict(sorted(clues_per_word_dist.items()))
        }
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Statistics saved to {output_file}")
    except Exception as e:
        print(f"Error saving statistics to {output_file}: {str(e)}")
    
    print("\nKey Statistics:")
    print(f"Total word-clue pairs: {stats['total_pairs']}")
    print(f"Unique words: {stats['unique_words']}")
    print(f"Average clues per word: {stats['average_clues_per_word']:.2f}")
    print(f"Maximum clues for a single word: {stats['max_clues_for_word']}")
    print(f"Word length range: {stats['word_length_stats']['min']} to {stats['word_length_stats']['max']} characters")
    print(f"Clue length range: {stats['clue_length_stats']['min']} to {stats['clue_length_stats']['max']} characters")
    print(f"Words with single clue: {stats['distribution']['words_with_single_clue']}")
    print(f"Words with multiple clues: {stats['distribution']['words_with_multiple_clues']}")
    
    return stats

if __name__ == '__main__':
    word_file = 'english/words.txt'
    output_file = 'english_words_simple.txt'
    with open(word_file, 'r', encoding='utf-8') as f:
        words = {line.strip() for line in f}
    valid_pairs = dictionary(words)
    print(f"Total pairs before filtering: {len(valid_pairs)}")
    filtered_pairs = filter_similar_clues_batch(valid_pairs, similarity_threshold=0.9, encoding_batch_size=1024)
    print(f"Total pairs after filtering: {len(filtered_pairs)}")
    write_output_file(filtered_pairs, output_file)
    stats = analyze_statistics(filtered_pairs, 'eng_simple_meta.json')

