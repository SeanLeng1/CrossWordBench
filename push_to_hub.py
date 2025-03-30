import datasets
import os
import json
from argparse import ArgumentParser
from PIL import Image
from pathlib import Path
import random

# https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/tools/make_image_hf_dataset.ipynb

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="data/")
    parser.add_argument("--save_path", type=str, default="full_data")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--subject", type=str, default="english")
    parser.add_argument("--difficulty", type=str, default="7x7")
    return parser.parse_args()

def generator(data_folder, subject, difficulty):
    data_path = Path(data_folder) / subject
    if not data_path.exists():
        raise FileNotFoundError(f"Data folder {data_path} does not exist.")
    difficulty_path = data_path / difficulty
    if not difficulty_path.exists():
        raise FileNotFoundError(f"Data folder {difficulty_path} does not exist.")
    # from_generator is weird, you need to modify the function a bit
    # otherwise it will use cached data...
    # gonna use random num for this behavior
    random_num = random.randint(0, 101)
    print(f'{random_num}')
    for sub_folder in difficulty_path.iterdir():
        # Load JSON files
        puzzle_state = json.dumps(json.loads((sub_folder / 'puzzle_state.json').read_text()))
        reference_answer = json.dumps(json.loads((sub_folder / 'crossword_answers.json').read_text()))
        partial_grid_0_5 = json.dumps(json.loads((sub_folder / 'partial_grid_0.5.json').read_text()))
        partial_grid_0_25 = json.dumps(json.loads((sub_folder / 'partial_grid_0.25.json').read_text()))
        partial_grid_0_75 = json.dumps(json.loads((sub_folder / 'partial_grid_0.75.json').read_text()))
        
        # Load images
        empty_image = Image.open(sub_folder / 'crossword_grid.png')
        key_image = Image.open(sub_folder / 'crossword_key.png')
        grid_only_empty_image = Image.open(sub_folder / 'crossword_grid_only.png')
        data = {
            'id': int(json.loads(puzzle_state)['meta_data']['id']),
            'difficulty': difficulty,
            'grid_image': empty_image,
            'empty_grid_image': grid_only_empty_image,
            'key_image': key_image,
            'partial_0.25': Image.open(sub_folder / 'crossword_partial_0.25.png'),
            'partial_0.5': Image.open(sub_folder / 'crossword_partial_0.5.png'),
            'partial_0.75': Image.open(sub_folder / 'crossword_partial_0.75.png'),
            'partial_grid_0.25': partial_grid_0_25,
            'partial_grid_0.5': partial_grid_0_5,
            'partial_grid_0.75': partial_grid_0_75,
            'puzzle_state': puzzle_state,
            'reference_answer': reference_answer,
        }
        yield data

if __name__ == '__main__':
    args = parse_args()
    features = datasets.Features({
        "grid_image": datasets.Image(),
        "key_image": datasets.Image(),
        "empty_grid_image": datasets.Image(),
        "partial_0.25": datasets.Image(),
        "partial_0.5": datasets.Image(),
        "partial_0.75": datasets.Image(),
        "partial_grid_0.25": datasets.Value("string"),
        "partial_grid_0.5": datasets.Value("string"),
        "partial_grid_0.75": datasets.Value("string"),
        "id": datasets.Value("int32"),  
        "difficulty": datasets.Value("string"),
        "reference_answer": datasets.Value("string"),
        "puzzle_state": datasets.Value("string"),
    })
    num_proc = 32
    data_eval = datasets.Dataset.from_generator(
        generator, 
        # TODO: should make input a list for multiprocessing
        gen_kwargs={'data_folder': args.data_folder, 'subject': args.subject, 'difficulty': args.difficulty},
        num_proc=num_proc,
    )
    if args.push:
        data_eval.push_to_hub(
            repo_id='JixuanLeng/CrossWordBench',
            split=args.difficulty,  # This becomes the split name (like "test")
            config_name=args.subject,  # This becomes the subset name (like "subset1") 
            private=True
        )
    else:
        save_dir = Path(args.save_path) / args.subject
        save_dir.mkdir(parents=True, exist_ok=True)
        data_eval.save_to_disk(str(save_dir / args.difficulty))
        





            