import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Set, Tuple

from tqdm import tqdm
from transformers import set_seed

from utils import (Genxword, gen_grid, post_process_save_file, setup_logger,
                   validate_save_formats)

PARTIAL_RATIOS = [0.25, 0.5, 0.75]
GRID_SIZES = [(7, 7), (14, 14)]
TARGET_CLUE_COUNTS = {
    (7, 7): 10,
    (14, 14): 20,
    (21, 21): 30,
}

class CrossWordGenerator:
    def __init__(self, args: argparse.Namespace, logger=None):
        self.args = args
        self.logger = logger or logging.getLogger(__name__)
    
    def process_wordlist(self, input_file: Path, ncols: int, nrows: int) -> Tuple[str, List[str]]:
        """Process input wordlist based on grid size and return filtered words."""
        filtered_words = []
        min_grid_size = min(ncols, nrows)
        with open(input_file, 'r') as f:
            for line in f:
                # split into word and clue if format is "word clue"
                parts = line.strip().split(' ', 1)
                word = parts[0] 
                clue = parts[1] if len(parts) > 1 else None  
                if len(word) <= min_grid_size - 2:
                    if clue:
                        filtered_words.append(f"{word} {clue}")
                    else:
                        filtered_words.append(word)
        filtered_words = sorted(filtered_words, key=lambda x: (len(x.split(' ')[0]), x))
        filtered_words = list(dict.fromkeys(filtered_words))
        
        if not filtered_words:
            raise ValueError(
                "No valid words found. Please increase grid size to at least 2 larger than the longest word."
            )

        # write filtered words to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix=".txt")
        with temp_file as tmp:
            tmp.write('\n'.join(filtered_words) + '\n')
            
        return temp_file.name, filtered_words
    
    def filter_used_clues(self, word_lists: List[str], used_clues: Set[str]) -> Tuple[str, List[str]]:
        """Filter out entries with previously used clues."""
        if not used_clues:
            return None, word_lists
            
        self.logger.info(f"Filtering wordlist to remove {len(used_clues)} previously used clues")
        original_count = len(word_lists)
        
        filtered_entries = [
            entry for entry in word_lists 
            if len(entry.split(' ', 1)) == 1 or entry.split(' ', 1)[1] not in used_clues
        ]
        
        # write filtered entries to temp file
        filtered_temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix=".txt")
        with filtered_temp_file as tmp:
            tmp.write('\n'.join(filtered_entries) + '\n')
            
        self.logger.info(
            f"Filtered wordlist contains {len(filtered_entries)} entries "
            f"(removed {original_count - len(filtered_entries)} entries with used clues)"
        )
        
        return filtered_temp_file.name, filtered_entries
    
    def initialize_generator(self, temp_word_file: str, num_words: int) -> Tuple[Genxword, int]:
        """Initialize the crossword generator with wordlist."""
        with open(temp_word_file, 'r') as f:
            word_count = sum(1 for _ in f)
        actual_num_words = min(num_words, word_count)
        if actual_num_words < num_words:
            self.logger.warning(
                f"Number of words to use ({num_words}) exceeds available words ({word_count}), "
                f"using {actual_num_words} words"
            )
        generator = Genxword(True, self.args.mix)
        with open(temp_word_file, 'r') as f:
            generator.wlist(f, actual_num_words)
            
        return generator, actual_num_words
    
    @staticmethod
    def get_target_clue_count(grid_size: Tuple[int, int]) -> int:
        return TARGET_CLUE_COUNTS.get(grid_size, 20)
    
    def setup_generator(self, grid_size: Tuple[int, int], used_clues: Optional[Set[str]] = None) -> Tuple[str, Genxword]:
        """Set up generator for specified grid size."""
        nrows, ncols = grid_size
        temp_word_file, word_lists = self.process_wordlist(self.args.input, ncols, nrows)
        if used_clues:
            new_temp_file, word_lists = self.filter_used_clues(word_lists, used_clues)
            if new_temp_file:
                Path(temp_word_file).unlink(missing_ok=True)
                temp_word_file = new_temp_file
        self.logger.info(f"Loaded {len(word_lists)} valid words for grid size {nrows}x{ncols}")
        generator, _ = self.initialize_generator(temp_word_file, float('inf'))
        generator.grid_size(nrows, ncols)
        return temp_word_file, generator

    @staticmethod
    def load_generated_data(output_dir: Path) -> Tuple[Set[str], Set[str]]:
        """Load previously generated puzzle data."""
        used_clues = set()
        generated_grids = set()
        
        if not output_dir.exists():
            return used_clues, generated_grids
            
        for folder in output_dir.iterdir():
            if not folder.is_dir():
                continue
                
            # Load puzzle state
            puzzle_state_file = folder / "puzzle_state.json"
            if puzzle_state_file.exists():
                try:
                    with open(puzzle_state_file, 'r') as f:
                        puzzle_state = json.load(f)
                        if 'grid' in puzzle_state:
                            generated_grids.add(str(puzzle_state['grid']))
                except Exception as e:
                    logging.warning(f"Could not load puzzle state from {puzzle_state_file}: {e}")
            
            # Load answers
            answer_state_file = folder / "crossword_answers.json"
            if answer_state_file.exists():
                try:
                    with open(answer_state_file, 'r') as f:
                        answers = json.load(f)
                        for answer in answers:
                            used_clues.add(answer['clue'])
                except Exception as e:
                    logging.warning(f"Could not load answers from {answer_state_file}: {e}")
                    
        return used_clues, generated_grids

    @staticmethod
    def count_existing_puzzles(output_dir: Path) -> int:
        """Count number of existing puzzle directories."""
        if not output_dir.exists():
            return 0
            
        return sum(1 for item in output_dir.iterdir() 
                  if item.is_dir() and any(item.iterdir()))
    
    def generate_partial_fills(self, puzzle_dir: Path) -> None:
        """Generate partial-filled and leave-one-out PDFs for a puzzle."""
        try:
            puzzle_state_file = puzzle_dir / "puzzle_state.json"
            if not puzzle_state_file.exists():
                return
            with open(puzzle_state_file, 'r') as f:
                puzzle_state = json.load(f)
            generator = Genxword(True, self.args.mix)
            # generate partial filled puzzles at different ratios
            if self.args.partial_fill is not None:
                partial_ratios = [self.args.partial_fill]
            else:
                partial_ratios = PARTIAL_RATIOS
            for ratio in partial_ratios:
                gen_grid(
                    puzzle_dir,
                    generator, 
                    self.args.formats,
                    prefill_ratio=ratio, 
                    puzzle_state=puzzle_state, 
                    silent=True,
                )
            # generate leave-one-out puzzle
            gen_grid(
                puzzle_dir,
                generator, 
                self.args.formats,
                left_one_out=True,
                puzzle_state=puzzle_state, 
                silent=True,
            )
        except Exception as e:
            self.logger.error(f"Error generating partial fills for {puzzle_dir}: {e}")

    def generate_for_grid_size(self, grid_size: Tuple[int, int], all_used_clues: Set[str]) -> Set[str]:
        """Generate crosswords for a specific grid size."""
        nrows, ncols = grid_size
        output_dir = Path(self.args.output) / f"{nrows}x{ncols}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        existing_count = self.count_existing_puzzles(output_dir)
        required_count = self.args.sample - existing_count
        
        if required_count <= 0:
            self.logger.info(f"Skipping {nrows}x{ncols} grid generation as all samples already exist")
            return all_used_clues
            
        # load previously generated data
        generated_clues, generated_grids = self.load_generated_data(output_dir)
        self.logger.info(f'Found {len(generated_clues)} previously used clues for grid size {nrows}x{ncols}')
        
        all_used_clues.update(generated_clues)
        
        temp_word_file, generator = self.setup_generator(grid_size, all_used_clues)
        
        try:
            target_clue_count = self.get_target_clue_count(grid_size)
            if 'commonsenseqa' in str(self.args.input):
                self.logger.warning('The target clue count is reduced to 3 due to limited data for commonsenseqa')
                target_clue_count = 5
            self.logger.info(f'Target clue count for {nrows}x{ncols} is {target_clue_count}')
            
            results, _, new_used_clues = gen_grid(
                output_dir,
                generator,
                nrows,
                ncols,
                self.args.formats,
                self.args.sample,        # still use args.sample instead of required count
                silent=True,
                greedy=True,
                no_replacement=self.args.no_replacement,
                logger=self.logger,
                target_clue_count=target_clue_count,
                existing_results=generated_grids,
            )
            
            self.logger.info(f"Generated {len(results)} crosswords for grid size {nrows}x{ncols}")
            
            # generate partial filled versions
            puzzle_dirs = [d for d in output_dir.iterdir() if d.is_dir() and any(d.iterdir())]
            for folder in tqdm(puzzle_dirs, desc=f'Generating variants for {nrows}x{ncols}'):
                self.generate_partial_fills(folder)
                
            # clean up empty folders
            for folder in output_dir.iterdir():
                if folder.is_dir() and not any(folder.iterdir()):
                    folder.rmdir()
                    
            all_used_clues.update(new_used_clues)
            
        finally:
            # clean up temp file
            if temp_word_file:
                Path(temp_word_file).unlink(missing_ok=True)
                
        return all_used_clues

    def generate_full_dataset(self) -> None:
        """Generate the full dataset with all grid sizes."""
        all_used_clues = set()
        
        for grid_size in GRID_SIZES:
            print('=' * 100)
            self.logger.info(f"Processing grid size {grid_size[0]}x{grid_size[1]}")
            all_used_clues = self.generate_for_grid_size(grid_size, all_used_clues)
            
            # post-process and clean up
            output_dir = Path(self.args.output) / f"{grid_size[0]}x{grid_size[1]}"
            post_process_save_file(output_dir, remove_pdf=True)

    def generate_single_size(self) -> None:
        """Generate crosswords for a single grid size."""
        grid_size = (self.args.nrow, self.args.ncol)
        output_dir = Path(self.args.output) / f"{grid_size[0]}x{grid_size[1]}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        temp_word_file, word_lists = self.process_wordlist(self.args.input, grid_size[1], grid_size[0])
        
        try:
            generator, actual_num_words = self.initialize_generator(temp_word_file, self.args.num_words)
            generator.grid_size(grid_size[0], grid_size[1])
            
            self.logger.info(f"Generating {self.args.sample} crosswords with {actual_num_words} words")
            
            results, _, _ = gen_grid(
                output_dir,
                generator,
                grid_size[0],
                grid_size[1],
                self.args.formats,
                self.args.sample,
                silent=True,
                no_replacement=self.args.no_replacement,
                logger=self.logger,
            )
            
            self.logger.info(f"Generated {len(results)} crosswords")
            
            # generate partial filled versions
            puzzle_dirs = [d for d in output_dir.iterdir() if d.is_dir() and any(d.iterdir())]
            for folder in tqdm(puzzle_dirs, desc='Generating variants'):
                self.generate_partial_fills(folder)
                
            # post-process and clean up
            post_process_save_file(output_dir, remove_pdf=True)
            
        finally:
            # clean up temp file
            if temp_word_file:
                Path(temp_word_file).unlink(missing_ok=True)

    def run(self) -> None:
        """Run the crossword generation process."""
        if self.args.gen_full:
            self.generate_full_dataset()
        else:
            self.generate_single_size()

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate crossword puzzles from word lists",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s -i words.txt -f pn -o mycrossword  # Generate A4 PDF and PNG\n"
            "  %(prog)s -i words.txt -f s --auto  # Generate SVG non-interactively\n"
            "  %(prog)s -i words.txt -f z -n 30 --mix  # Generate IPUZ with 30 words and anagrams\n"
            "\n"
            "Input File Format:\n"
            "  - One entry per line\n"
            "  - Optional format: word<space>clue\n"
        )
    )
    
    # Input/Output Options
    input_group = parser.add_argument_group('Input/Output Options')
    input_group.add_argument(
        '-i', '--input',
        type=Path,
        required=True,
        help="Path to word list file"
    )
    input_group.add_argument(
        '-f', '--formats',
        type=validate_save_formats,
        required=True,
        help="Output formats (combine): p (A4 PDF), l (Letter PDF), n (PNG), s (SVG), z (IPUZ)"
    )
    input_group.add_argument(
        '-o', '--output',
        default='crossword',
        help="Base name for output folder (default: %(default)s)"
    )
    
    # Generation Options
    generation_group = parser.add_argument_group('Generation Options')
    generation_group.add_argument(
        '-n', '--num-words',
        type=int,
        default=50,
        help="Number of words to use from input (default: %(default)d)"
    )
    generation_group.add_argument(
        '-m', '--mix',
        action='store_true',
        help="Generate anagram clues"
    )
    generation_group.add_argument(
        '--nrow',
        type=int,
        default=7,
        help="Number of rows in the grid"
    )
    generation_group.add_argument(
        '--ncol',
        type=int,
        default=7,
        help="Number of columns in the grid"
    )
    generation_group.add_argument(
        '--sample',
        type=int,
        default=10,
        help="Number of crosswords to generate"
    )
    generation_group.add_argument(
        '--partial-fill',
        type=float,
        default=None,
        help="Percentage of cells to reveal in partial-filled PDF. default: None"
    )
    generation_group.add_argument(
        '--no-replacement',
        action='store_true',
        help="Generate puzzle with no clues replacement"
    )
    generation_group.add_argument(
        '--gen-full',
        action='store_true',
        help="Automatically generate full dataset"
    )
    
    # Debug Options
    debug_group = parser.add_argument_group('Debug Options')
    debug_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Enable verbose debug output"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(1234)
    logger = setup_logger(verbose=args.verbose)
    logger.info("\U0001F61A Starting crossword generation with arguments: %s", vars(args))
    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        raise FileNotFoundError(f"Input file not found: {args.input}")
    generator = CrossWordGenerator(args, logger)
    generator.run()

if __name__ == '__main__':
    main()