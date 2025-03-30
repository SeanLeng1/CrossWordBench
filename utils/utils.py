import argparse
import inspect
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from gi.repository import Pango
from pdf2image import convert_from_path
from PIL import Image, ImageChops
from tqdm import tqdm

from utils.patch import Exportfiles


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',    
        'INFO': '\033[32m',   
        'WARNING': '\033[33m', 
        'ERROR': '\033[31m', 
        'CRITICAL': '\033[41m', 
        'RESET': '\033[0m'     
    }

    def format(self, record: logging.LogRecord) -> str:
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        while caller_frame and 'logging' in caller_frame.f_code.co_filename:
            caller_frame = caller_frame.f_back
        
        if caller_frame:
            file_path = caller_frame.f_code.co_filename
            line_no = caller_frame.f_lineno
            file_name = os.path.basename(file_path)
        else:
            file_name = "unknown"
            line_no = 0

        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        level_color = self.COLORS.get(record.levelname, '')
        level_name = f"{level_color}{record.levelname:<8}{self.COLORS['RESET']}"
        location_info = f"{file_name}:{line_no:<3}"
        location_info = f"{location_info:<20}"
        
        msg = (f"{timestamp} | "
               f"{level_name} | "
               f"{location_info} | "
               f"{record.getMessage()}")
        
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"  
        return msg

def setup_logger(verbose: bool, name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name) if name else logging.getLogger(__name__)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    formatter = ColoredFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger

def validate_save_formats(formats: str) -> str:
    """Validate the save format string contains only valid format specifiers."""
    valid_formats = {'p', 'l', 'n', 's', 'z'}
    formats = formats.lower()
    if not all(f in valid_formats for f in formats):
        raise argparse.ArgumentTypeError(
            "Invalid format(s). Valid formats are: "
            "p (A4 PDF), l (Letter PDF), n (PNG), s (SVG), z (IPUZ)"
        )
    return formats

##############################
# postprocess the save file
# recommend using combined formats
# since we might need to extract ground truth from text file
##############################
def trim_whitespace(image, margin=100):  
    """Remove white borders from image but keep a margin"""
    bg = Image.new(image.mode, image.size, 'white')
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    if bbox:
        left, top, right, bottom = bbox
        width, height = image.size
        left = max(0, left - margin)
        top = max(0, top - margin)
        right = min(width, right + margin)
        bottom = min(height, bottom + margin)
        return image.crop((left, top, right, bottom))
    return image

def process_pdfs(output_folder):
    output_folder = Path(output_folder)
    for file in output_folder.iterdir():
        if file.suffix == '.pdf':
            images = convert_from_path(file, first_page=1, last_page=1, thread_count=4)
            try:
                images[1] 
                print(f"Skipping {file}: contains more than 1 page.")
                continue
            except IndexError:
                image = images[0]
                image = trim_whitespace(image)
                png_path = Path(output_folder) / f"{Path(file).stem}.png"
                image.save(png_path, 'PNG')

def post_process_save_file(output_folder, remove_pdf=True, use_tqdm=True):
    output_folder = Path(output_folder).resolve()
    folders = list(output_folder.iterdir())
    if use_tqdm:
        pbar = tqdm(total=len(folders), desc='Post-processing files')

    for folder in folders:
        if folder.is_dir():  
            temp_folder = folder
            if not any(temp_folder.iterdir()):  
                temp_folder.rmdir() 
                if use_tqdm:
                    pbar.update(1) 
                continue
        else:  
            temp_folder = output_folder

        # for file in temp_folder.iterdir():
        #     if file.suffix == '.png':
        #         file.unlink()  

        process_pdfs(temp_folder)

        if remove_pdf:
            for file in temp_folder.iterdir():
                if file.suffix == '.pdf':
                    file.unlink() 

        if use_tqdm:
            pbar.update(1)

############################################################ 
# Main Crossword Generation Wrapper
############################################################
def _should_process_existing_puzzle(
    prefill_ratio: float, 
    left_one_out: bool, 
    filled_words: Optional[Dict[str, str]]
) -> bool:
    """Determine if we should process an existing puzzle instead of generating a new one."""
    if filled_words is not None and prefill_ratio > 0:
        raise ValueError("Cannot have both filled words and partial fill")
    return prefill_ratio > 0 or left_one_out or filled_words is not None


def _process_existing_puzzle(
    output_dir: Path,
    puzzle_state: Dict[str, Any],
    formats: str,
    lang: List[str],
    message: str,
    prefill_ratio: float = 0,
    left_one_out: bool = False,
    filled_words: Optional[Dict[str, str]] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[None, bool, None]:
    """Process an existing puzzle for partial fill, leave-one-out, or filled words."""
    if puzzle_state is None:
        raise ValueError("puzzle_state must be provided for processing existing puzzles")
    
    # extract puzzle data
    best_grid, best_wordlist = puzzle_state['grid'], puzzle_state['wordlist']
    grid_size = puzzle_state['meta_data']['grid_size']
    nrow, ncol = map(int, grid_size.split('x'))
    
    exp = Exportfiles(nrow, ncol, best_grid, best_wordlist, '-')
    
    if filled_words is not None:
        correctness, filled_data = exp.create_files(
            f'{output_dir}/crossword', 
            formats, 
            lang, 
            message, 
            filled_words=filled_words
        )
        if filled_data:
            with open(f'{output_dir}/filled_words.json', 'w') as f:
                json.dump(filled_data, f, indent=2)
    
    elif prefill_ratio > 0:
        correctness, partial_data = exp.create_files(
            f'{output_dir}/crossword', 
            formats, 
            lang, 
            message, 
            prefill_ratio=prefill_ratio
        )
        if partial_data:
            with open(f'{output_dir}/partial_grid_{prefill_ratio}.json', 'w') as f:
                json.dump(partial_data, f, indent=2)
    
    elif left_one_out:
        correctness, partial_data = exp.create_files(
            f'{output_dir}/crossword', 
            formats, 
            lang, 
            message, 
            left_one_out=left_one_out
        )
        if partial_data:
            with open(f'{output_dir}/left_one_out.json', 'w') as f:
                json.dump(partial_data, f, indent=2)
    
    return None, correctness, None


def _maybe_refresh_word_pool(generator, original_wordlist, used_clues, logger):
    """Check if all clues are used and refresh the word pool if needed."""
    if original_wordlist is None:
        return
    
    available_clues = {word_entry[1] for word_entry in generator.wordlist if len(word_entry) > 1}
    
    if available_clues and all(clue in used_clues for clue in available_clues):
        logger.info("All clues used - refreshing clue pool")
        generator.wordlist = original_wordlist.copy()
        used_clues.clear()


def _update_used_clues(generator, best_wordlist, used_clues):
    """Update the set of used clues and remove them from the generator's wordlist."""
    puzzle_clue = []
    
    for word_entry in best_wordlist:
        if len(word_entry) > 1:
            clue = word_entry[1]
            used_clues.add(clue)
            puzzle_clue.append(clue)
    
    # update generator wordlist
    if hasattr(generator, 'wordlist'):
        new_wordlist = [
            word_entry for word_entry in generator.wordlist
            if len(word_entry) <= 1 or word_entry[1] not in puzzle_clue
        ]
        generator.wordlist = new_wordlist


def _save_puzzle_state(
    puzzle_dir: Path,
    best_grid: List[List[str]],
    best_wordlist: List[List[Any]],
    best_score: float,
    nrow: int,
    ncol: int,
    filled_words: Optional[Dict[str, str]],
    prefill_ratio: float,
    puzzle_id: int
) -> None:
    """Save puzzle state to file."""
    puzzle_state = {
        'grid': best_grid,
        'wordlist': best_wordlist,
        'score': best_score,
        'meta_data': {
            'grid_size': f'{nrow}x{ncol}',
            'filled_words': filled_words,
            'prefill_ratio': prefill_ratio,
            'id': puzzle_id
        },
        'length': len(best_wordlist)
    }
    
    with open(f'{puzzle_dir}/puzzle_state.json', 'w') as f:
        json.dump(puzzle_state, f, indent=2)


def gen_grid(
    output_dir: Union[str, Path],
    generator=None,
    nrow: int = 7,
    ncol: int = 7,
    formats: str = 'np',
    sample: int = 5,
    prefill_ratio: float = 0,
    left_one_out: bool = False,
    filled_words: Optional[Dict[str, str]] = None,
    puzzle_state: Optional[Dict[str, Any]] = None,
    silent: bool = False,
    greedy: bool = False,
    tol: int = 10,
    no_replacement: bool = False,
    time_permitted: float = 30.00,
    logger: Optional[logging.Logger] = None,
    target_clue_count: int = 8,
    existing_results: Optional[Set[str]] = None,
) -> Tuple[Optional[Set[str]], bool, Optional[Set[str]]]:
    """Generate crossword puzzles and save them to output directory.
    
    This function has three main modes of operation:
    1. Generate new puzzles from scratch
    2. Apply partial fill to an existing puzzle
    3. Create a "leave one out" version of an existing puzzle
    
    Args:
        output_dir: Directory to save the generated puzzles
        generator: Genxword instance for puzzle generation
        nrow: Number of rows in the grid
        ncol: Number of columns in the grid
        formats: Output formats (e.g., 'np' for PNG)
        sample: Number of puzzles to generate
        prefill_ratio: Ratio of cells to prefill (0-1)
        left_one_out: Whether to create a "leave one out" version
        filled_words: Dictionary of words to fill in
        puzzle_state: Existing puzzle state for modes 2 and 3
        silent: Whether to suppress progress output
        greedy: Whether to use greedy generation
        tol: Tolerance for consecutive duplicates before stopping
        no_replacement: Whether to avoid reusing clues
        time_permitted: Time limit for each generation attempt
        logger: Logger instance
        target_clue_count: Minimum number of clues required
        existing_results: Set of existing puzzle grids to avoid duplication
        
    Returns:
        Tuple of (results, correctness, used_clues)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    lang = ['Across:', 'Down:']
    message = ''
    used_clues = None
    output_dir = Path(output_dir)
    # mode handling: process existing puzzle (partial fill or leave-one-out)
    if _should_process_existing_puzzle(prefill_ratio, left_one_out, filled_words):
        return _process_existing_puzzle(
            output_dir, 
            puzzle_state,
            formats, 
            lang, 
            message, 
            prefill_ratio, 
            left_one_out, 
            filled_words,
            logger
        )
    results = set(existing_results) if existing_results else set()
    consecutive_duplicates = 0
    total_attempts = 0
    puzzle_id = len(results) + 1
    original_wordlist = None
    correctness = False

    if no_replacement:
        used_clues = set()
        logger.info('Generating without Replacement...')

    progress_bar = None
    if silent:
        desc = 'Generating crosswords (greedy mode)' if greedy else 'Generating crosswords'
        progress_bar = tqdm(total=sample - len(results), desc=desc)
    elif greedy:
        logger.info('Generating crosswords (greedy mode)...')

    # main generation
    while len(results) < sample and consecutive_duplicates < tol:
        if no_replacement and hasattr(generator, 'wordlist'):
            _maybe_refresh_word_pool(generator, original_wordlist, used_clues, logger)
            if original_wordlist is None:
                original_wordlist = generator.wordlist.copy()

        total_attempts += 1
        if not silent:
            status = f'greedy: {total_attempts} attempts' if greedy else f'{len(results) + 1}/{sample}'
            logger.info(f'Generating crossword {status}...')

        puzzle_dir = output_dir / str(puzzle_id)
        puzzle_dir.mkdir(parents=True, exist_ok=True)
        name = f'{puzzle_dir}/crossword'
        best_grid, best_wordlist, best_score = generator.gengrid(
            silent=silent, 
            time_permitted=time_permitted
        )
        if len(best_wordlist) <= target_clue_count:
            continue
        if str(best_grid) in results:
            consecutive_duplicates += 1
            continue
        consecutive_duplicates = 0
        if no_replacement:
            _update_used_clues(generator, best_wordlist, used_clues)
        
        _save_puzzle_state(
            puzzle_dir, 
            best_grid, 
            best_wordlist, 
            best_score, 
            nrow, 
            ncol, 
            filled_words, 
            prefill_ratio, 
            puzzle_id
        )

        if not silent:
            message = 'The following files have been saved to your current working directory:\n'
        exp = Exportfiles(nrow, ncol, best_grid, best_wordlist, '-')
        correctness, _ = exp.create_files(name, formats, lang, message)

        results.add(str(best_grid))
        puzzle_id += 1

        if silent and progress_bar:
            progress_bar.update(1)
            if greedy:
                progress_bar.set_description(f'Generated {len(results)} unique puzzles')
        
    if silent and progress_bar:
        progress_bar.close()

    logger.warning(f'Generated {len(results)} unique puzzles in {total_attempts} attempts')
    if consecutive_duplicates >= tol:
        logger.warning(f'Stopped due to {consecutive_duplicates} consecutive duplicates')
    
    return results, correctness, used_clues


def gen_empty_grid(
    puzzle_state,
    output_dir,
):
    lang = ('Across:/Down:').split('/')
    best_grid, best_wordlist = puzzle_state['grid'], puzzle_state['wordlist']
    grid_size = puzzle_state['meta_data']['grid_size']
    nrow = int(grid_size.split('x')[0])
    ncol = int(grid_size.split('x')[1])
    exp = Exportfiles(nrow, ncol, best_grid, best_wordlist, '-')
    name = f'{output_dir}/crossword'
    os.makedirs(output_dir, exist_ok=True)
    if Pango.find_base_dir(exp.wordlist[0][0], -1) == Pango.Direction.RTL:
        [i.reverse() for i in exp.grid]
        RTL = True
    else:
        RTL = False
    exp.export_pdf(name, '_grid_only.pdf', lang, RTL, include_clues = False)
    post_process_save_file(output_dir, remove_pdf=True, use_tqdm=False)




