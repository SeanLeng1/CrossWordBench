"""
You probably don't need this, it is used to generate extra grid only image based on already generated data
"""
import json
import os
from utils import (
    gen_empty_grid
)
from tqdm import tqdm
import argparse 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, default='chinese')
    parser.add_argument('--difficulty', type=str, default='21x21')
    args = parser.parse_args()
    data_path = f'data/{args.subject}/{args.difficulty}'
    for folder in tqdm(os.listdir(data_path)):
        with open(f'{data_path}/{folder}/puzzle_state.json', 'r') as f:
            puzzle_state = json.load(f)
        gen_empty_grid(puzzle_state, f'{data_path}/{folder}/')