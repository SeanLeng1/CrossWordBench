import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import re
import argparse
from pathlib import Path
from transformers import set_seed
from tqdm import tqdm
from datasets import load_dataset

def normalize_string(string):
    string = string.strip().upper()
    if string.endswith('.'):
        string = string[:-1]
    return string

def process_simpleqa(output_folder):
    df = pd.read_csv(
                f"https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
            )
    word_lists = []
    examples = [row.to_dict() for _, row in df.iterrows()]
    print('total examples', len(examples))
    for example in tqdm(examples, desc="Processing examples", total=len(examples)):
        problem = example.get("problem", "")
        answer = example.get("answer", "")
        if problem and answer:
            # answer is not number
            if not re.search(r'\s', answer) and not re.search(r'\d', answer):
                word_lists.append(normalize_string(answer) + " " + problem)
    print('final valid examples', len(word_lists))
    output_file = Path(output_folder) / 'simpleqa_word.txt'
    with open(output_file, 'w') as f:
        for word_list in word_lists:
            f.write(f"{word_list}\n")

def process_commonsenseqa(output_folder):
    data = load_dataset('tau/commonsense_qa', split = 'train')
    word_lists = []
    for example in tqdm(data, desc="Processing examples", total=len(data)):
        question = example['question']
        options = example['choices']  
        answer_key = example['answerKey']
        
        if answer_key in options['label']:
            index = options['label'].index(answer_key)
            answer_text = options['text'][index]
            if not re.search(r'\s', answer_text) and not re.search(r'\d', answer_text):
                word_lists.append(normalize_string(answer_text) + " " + question)
    print('final valid examples', len(word_lists))
    output_file = Path(output_folder) / 'commonsenseqa_word.txt'
    with open(output_file, 'w') as f:
        for word_list in word_lists:
            f.write(f"{word_list}\n")


def process_dbpedia(output_folder):
    data = load_dataset('s-nlp/Llama-3.1-8B-Instruct-DBpedia-HighlyKnown', split = 'full')
    word_lists = []
    word_clue_set = set()
    for example in tqdm(data, desc="Processing examples", total=len(data)):
        question = example['question']
        answers = example['greedy_ans']
        labels = example['p_greed']
        for answer, label in zip(answers, labels):
            try:
                answer = answer.split('Answer: ')[1]
            except:
                pass
            if label == True:
                if not re.search(r'\s', answer) and not re.search(r'\d', answer):
                    if f"{normalize_string(answer)} {question}" not in word_clue_set:
                        word_clue_set.add(f"{normalize_string(answer)} {question}")
                        word_lists.append(normalize_string(answer) + " " + question)
    print('final valid examples', len(word_lists))
    output_file = Path(output_folder) / 'dbpedia_word.txt'
    with open(output_file, 'w') as f:
        for word_list in word_lists:
            f.write(f"{word_list}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process existing bench dataset.")
    parser.add_argument("--output_folder", type=str, default="bench_word_lists", help="Output file name.")
    args = parser.parse_args()
    set_seed(42)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    # process_simpleqa(args.output_folder)
    process_commonsenseqa(args.output_folder)
    # process_dbpedia(args.output_folder)
