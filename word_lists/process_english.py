import csv
from tqdm import tqdm
from collections import defaultdict
import json
import re
from sentence_transformers import SentenceTransformer
import torch

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

def clean_word(word):
    """Clean a word by removing punctuation and whitespace"""
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', word)
    return cleaned.strip()

def filter_clue(clue):
    clue_lower = clue.lower()
    # "1-Across", "4Down", "39-across", "12 down", "5 Across", etc.
    # These are context-specific and not useful for general word lists
    reference_pattern = re.compile(r'\d+[-\s]?(across|down)', re.IGNORECASE)
    punctuation_pattern = re.compile(r'^\W+$')
    note_references = [
        'see note',
        'seenote',
        'see notepad',
        'see notes',
        'see above',
        'see below'
    ]
    if (len(clue) >= 70 or 
        reference_pattern.search(clue) or 
        punctuation_pattern.match(clue) or 
        len(clue) <= 2 or
        any(ref in clue_lower for ref in note_references)):
        return False
    return True

def filter_answer(answer):
    # single letter or pure numbers can be ambiguous
    if len(answer) < 2 or answer.isdigit():
        return False
    return True

def process_csv_file(csv_file):
    word_clue_pairs = set()
    with open(csv_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        total_rows = sum(1 for _ in open(csv_file, encoding='utf-8')) - 1 
        for row in tqdm(reader, desc=f'Processing {csv_file}', total=total_rows):
            answer = clean_word(row.get('answer', ''))
            clue = row.get('clue', '').strip()
            if clue == '':
                clue = row.get('question', '').strip()
            if answer and clue and filter_clue(clue) and filter_answer(answer):
                word_clue_pairs.add((answer, clue))
    return word_clue_pairs

def process_tsv_file(tsv_file):
    word_clue_pairs = set()
    with open(tsv_file, 'r', encoding='utf-8') as file:
        header = file.readline().strip().split('\t')
        answer_index = header.index('answer')
        clue_index = header.index('clue')
        total_rows = sum(1 for _ in open(tsv_file, encoding='utf-8')) - 1 
        for line in tqdm(file, desc=f'Processing {tsv_file}', total=total_rows):
            row = line.strip().split('\t')
            if len(row) > max(answer_index, clue_index):
                answer = clean_word(row[answer_index])
                clue = row[clue_index].strip()
                if answer and clue and filter_clue(clue) and filter_answer(answer):
                    word_clue_pairs.add((answer, clue))

    return word_clue_pairs

def write_output_file(word_clue_pairs, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for word, clue in sorted(word_clue_pairs):  # Sort for consistency
            outfile.write(f'{word} {clue}\n')

def analyze_statistics(word_clue_pairs, output_file = 'eng_meta.json'):
    # Group clues by word
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
    csv_file = 'english/clues.csv'
    csv_file2 = 'english/clues2.csv'
    output_file = 'english_words.txt'
    csv_pairs = process_csv_file(csv_file)
    csv_pairs2 = process_csv_file(csv_file2)
    all_pairs = csv_pairs.union(csv_pairs2)
    print(f"Total pairs before filtering: {len(all_pairs)}")
    filtered_pairs = filter_similar_clues_batch(all_pairs, similarity_threshold=0.9, encoding_batch_size=4096)
    print(f"Total pairs after filtering: {len(filtered_pairs)}")
    write_output_file(filtered_pairs, output_file)
    stats = analyze_statistics(filtered_pairs)

    # this contains more cyptic clues which is more challenging and requires more fine-grained filtering
    # uncomment if you want to use
    # tsv_file = 'english/clues.tsv'
    # tsv_pairs = process_tsv_file(tsv_file)
    # print(f"Total cryptic pairs before filtering: {len(tsv_pairs)}")
    # filtered_tsv_pairs = filter_similar_clues_batch(tsv_pairs, similarity_threshold=0.9, encoding_batch_size=1024)
    # print(f"Total cryptic pairs after filtering: {len(filtered_tsv_pairs)}")
    # write_output_file(filtered_tsv_pairs, 'english_cryptic_words.txt')
    # stats = analyze_statistics(filtered_tsv_pairs, 'eng_cryptic_meta.json')
