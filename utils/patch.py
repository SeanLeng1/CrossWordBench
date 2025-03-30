"""
We are patching https://github.com/riverrun/genxword/blob/master/genxword/control.py

TODO: refactor and create separate file without dependencies
"""
import json
import multiprocessing as mp
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial, wraps
from operator import itemgetter
from typing import Callable, TypeVar

import cairo
import gi
from genxword.calculate import Exportfiles
from genxword.complexstring import ComplexString
from genxword.control import Genxword, _  # genxword is weird :(
from gi.repository import Pango
from tqdm import tqdm

gi.require_version('PangoCairo', '1.0')
gi.require_version('Pango', '1.0')



T = TypeVar('T')



def patch_class_method(target_class: type, method_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        # Patch the method
        setattr(target_class, method_name, wrapper)
        return wrapper
    return decorator

############################################################ 
# For benchmark, we do not need to generate perfect grid (as many intersecting words as possible)
# Instead, we want more variety
# So I rewrite this to add more randomization
############################################################
class Crossword(object):
    def __init__(self, rows, cols, empty=' ', available_words=[]):
        self.rows = rows
        self.cols = cols
        self.empty = empty
        self.available_words = available_words
        self.let_coords = defaultdict(list)
        self.used_first_words = set()
        self.best_score = 0
        self.current_score = 0

    def prep_grid_words(self):
        self.current_wordlist = []
        self.let_coords.clear()
        self.grid = [[self.empty]*self.cols for i in range(self.rows)]

        self.available_words = [word[:2] for word in self.available_words]
        #self.first_word(self.available_words[0])

        # instead of always using the first word, we randomly select from available words that has not been used 
        # as the first word recently
        available_first_words = [w for w in self.available_words if w[0] not in self.used_first_words]
        if not available_first_words:  
            self.used_first_words.clear()
            available_first_words = self.available_words
        first_word = random.choice(available_first_words)
        self.used_first_words.add(first_word[0])
        self.first_word(first_word)

    def compute_crossword(self, time_permitted=5.00):
        """
        Add some randomization factor
        """
        self.best_wordlist = []
        wordlist_length = len(self.available_words)
        time_permitted = float(time_permitted)
        start_full = float(time.time())
        while (float(time.time()) - start_full) < time_permitted:
            self.prep_grid_words()

            # [self.add_words(word) for i in range(2) for word in self.available_words if word not in self.current_wordlist]
            
            # randomly shuffle available words for each attempt
            shuffled_words = list(self.available_words)
            random.shuffle(shuffled_words)
            for i in range(5):
                for word in shuffled_words:
                    # if word not in self.current_wordlist:
                    # even the same word with different clues should not be repeated
                    if not any(w[0] == word[0] for w in self.current_wordlist):
                        coords = self.get_coords(word)
                        if coords:
                            self.add_words(word)
                            self.current_score = coords[3]

            if len(self.current_wordlist) > len(self.best_wordlist):
                self.best_wordlist = list(self.current_wordlist)
                self.best_grid = list(self.grid)
                self.best_score = self.current_score
            if len(self.best_wordlist) == wordlist_length:
                break
        #answer = '\n'.join([''.join(['{} '.format(c) for c in self.best_grid[r]]) for r in range(self.rows)])
        answer = '\n'.join([''.join([u'{} '.format(c) for c in self.best_grid[r]])
                            for r in range(self.rows)])
        return answer + '\n\n' + str(len(self.best_wordlist)) + ' out of ' + str(wordlist_length)

    def get_coords(self, word):
        """Return possible coordinates for each letter."""
        word_length = len(word[0])
        coordlist = []
        temp_list =  [(idx, v) for idx, letter in enumerate(word[0])
                      for k, v in self.let_coords.items() if k == letter]
        for coord in temp_list:
            letc = coord[0]
            for item in coord[1]:
                (rowc, colc, vertc) = item
                if vertc:
                    if colc - letc >= 0 and (colc - letc) + word_length <= self.cols:
                        row, col = (rowc, colc - letc)
                        score = self.check_score_horiz(word, row, col, word_length)
                        if score:
                            coordlist.append([rowc, colc - letc, 0, score])
                else:
                    if rowc - letc >= 0 and (rowc - letc) + word_length <= self.rows:
                        row, col = (rowc - letc, colc)
                        score = self.check_score_vert(word, row, col, word_length)
                        if score:
                            coordlist.append([rowc - letc, colc, 1, score])
        if coordlist:
            return max(coordlist, key=itemgetter(3))
        else:
            return

    def first_word(self, word):
        """Place the first word at a random position in the grid."""
        vertical = random.randrange(0, 2)
        if vertical:
            row = random.randrange(0, self.rows - len(word[0]))
            col = random.randrange(0, self.cols)
        else:
            row = random.randrange(0, self.rows)
            col = random.randrange(0, self.cols - len(word[0]))
        self.set_word(word, row, col, vertical)

    def add_words(self, word):
        """Add the rest of the words to the grid."""
        coordlist = self.get_coords(word)
        if not coordlist:
            return
        row, col, vertical = coordlist[0], coordlist[1], coordlist[2]
        self.set_word(word, row, col, vertical)

    def check_score_horiz(self, word, row, col, word_length, score=1):
        cell_occupied = self.cell_occupied
        if col and cell_occupied(row, col-1) or col + word_length != self.cols and cell_occupied(row, col + word_length):
            return 0
        for letter in word[0]:
            active_cell = self.grid[row][col]
            if active_cell == self.empty:
                if row + 1 != self.rows and cell_occupied(row+1, col) or row and cell_occupied(row-1, col):
                    return 0
            elif active_cell == letter:
                score += 1
            else:
                return 0
            col += 1
        return score

    def check_score_vert(self, word, row, col, word_length, score=1):
        cell_occupied = self.cell_occupied
        if row and cell_occupied(row-1, col) or row + word_length != self.rows and cell_occupied(row + word_length, col):
            return 0
        for letter in word[0]:
            active_cell = self.grid[row][col]
            if active_cell == self.empty:
                if col + 1 != self.cols and cell_occupied(row, col+1) or col and cell_occupied(row, col-1):
                    return 0
            elif active_cell == letter:
                score += 1
            else:
                return 0
            row += 1
        return score

    def set_word(self, word, row, col, vertical):
        """Put words on the grid and add them to the word list."""
        word.extend([row, col, vertical])
        self.current_wordlist.append(word)

        horizontal = not vertical
        for letter in word[0]:
            self.grid[row][col] = letter
            if (row, col, horizontal) not in self.let_coords[letter]:
                self.let_coords[letter].append((row, col, vertical))
            else:
                self.let_coords[letter].remove((row, col, horizontal))
            if vertical:
                row += 1
            else:
                col += 1

    def cell_occupied(self, row, col):
        cell = self.grid[row][col]
        if cell == self.empty:
            return False
        else:
            return True

############################################################ 
# Patching draw_img to support reveal_mask for partial filled puzzle
# It also should support fill in specific words
############################################################
@patch_class_method(Exportfiles, 'draw_img')
def draw_img(
    self, 
    name, 
    context, 
    px, 
    xoffset, 
    yoffset, 
    RTL, 
    reveal_mask=None,
    filled_words=None,
):
    # create cell mask from filled_words
    # wordlist: [word, clue, position y, position x, orientation]
    correctness = True
    word_mask = None
    if filled_words is not None:
        word_mask = [[False] * self.cols for _ in range(self.rows)]
        filled_word_set = set(w.upper() for w in filled_words)
        grid_word_set = set(word[0].upper() for word in self.wordlist)
        for filled_word in filled_word_set:
            if filled_word not in grid_word_set:
                correctness = False
                break

        for word in self.wordlist:
            if word[0].upper() in [w.upper() for w in filled_words]:  # Case-insensitive comparison
                x, y = word[3], word[2]
                length = len(word[0])
                if word[4] == 0:  # horizontal
                    for i in range(length):
                        word_mask[y][x + i] = True
                else:  # vertical
                    for i in range(length):
                        word_mask[y + i][x] = True

        # Create filled grid using the word mask
        filled_grid = [[self.empty]*self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                if word_mask[r][c]:
                    filled_grid[r][c] = self.grid[r][c]
        
        # Store the filled grid and word mask
        self._last_filled_grid = filled_grid
        self._last_word_mask = word_mask

    for r in range(self.rows):
        for i, c in enumerate(self.grid[r]):
            if c != self.empty:
                context.set_line_width(1.0)
                context.set_source_rgb(0.5, 0.5, 0.5)
                context.rectangle(xoffset+(i*px), yoffset+(r*px), px, px)
                context.stroke()
                context.set_line_width(1.0)
                context.set_source_rgb(0, 0, 0)
                context.rectangle(xoffset+1+(i*px), yoffset+1+(r*px), px-2, px-2)
                context.stroke()
                # draw letter if:
                # 1. it's a key file
                # 2. the cell is in revel_mask
                # the cell is in word_mask
                should_reveal = (
                    '_key.' in name or (reveal_mask is not None and reveal_mask[r][i]) or
                    (word_mask is not None and word_mask[r][i])
                )
                if should_reveal:
                    self.draw_letters(c, context, xoffset+(i*px)+10, yoffset+(r*px)+8, 'monospace 11')
        self.order_number_words()
        for word in self.wordlist:
            if RTL:
                x, y = ((self.cols-1)*px)+xoffset-(word[3]*px), yoffset+(word[2]*px)
            else:
                x, y = xoffset+(word[3]*px), yoffset+(word[2]*px)
            self.draw_letters(str(word[5]), context, x+3, y+2, 'monospace 6')
    return correctness

############################################################ 
# Patching export_pdf method to support partial fill and remove the title on pdf
############################################################
@patch_class_method(Exportfiles, 'export_pdf')
def export_pdf(
    self, 
    xwname, 
    filetype, 
    lang, 
    RTL, 
    width=700,
    height=842,
    prefill_ratio=None,
    filled_words=None,
    include_clues=True,
    left_one_out=False,
):
    # First calculate required height for all content
    px, xoffset, yoffset = 28, 36, 72
    name = xwname + filetype

    # Safely generate legend with error checking
    def safe_legend(lang):
        outStrA = lang[0] + '\n'
        outStrD = '\n' + lang[1] + '\n'
        
        for word in self.wordlist:
            try:
                # Check if word has all required elements
                if len(word) < 6:
                    continue
                    
                word_num = word[5]
                clue = word[1]
                orientation = word[4]
                
                if orientation == 0:  # Across
                    outStrA += f'{word_num}. {clue}\n'
                else:  # Down
                    outStrD += f'{word_num}. {clue}\n'
            except (IndexError, TypeError):
                print(f"Warning: Skipping malformed word entry: {word}")
                continue
                
        return outStrA + outStrD

    # Calculate grid height
    sc_ratio = float(width-(xoffset*2))/(px*self.cols)
    if self.cols <= 21:
        sc_ratio, xoffset = 0.8, float(width-(px*self.cols))/2
    grid_height = (self.rows * px * sc_ratio) + yoffset

    # Calculate clues height with standard font size
    clues = self.wrap(safe_legend(lang))
    clue_lines = clues.splitlines()[1:]  # Skip header lines
    standard_line_height = 16
    clues_height = len(clue_lines) * standard_line_height

    total_required_height = grid_height + clues_height + yoffset

    # Adjust page height if needed (max reasonable height, e.g., 2x A4)
    max_height = 1684  # 2x A4 height
    actual_height = min(max(height, total_required_height), max_height)
    if actual_height == max_height:
        width *= 2
    
    # Calculate font scaling if needed
    font_scale = 1.0
    if total_required_height > actual_height:
        available_clues_space = actual_height - grid_height - yoffset
        font_scale = available_clues_space / clues_height
        font_scale = max(0.6, font_scale)  # Don't go smaller than 60% of original size

    # Create surface and context
    surface = cairo.PDFSurface(name, width, actual_height)
    context = cairo.Context(surface)
    
    # White background
    context.set_source_rgb(1, 1, 1)
    context.rectangle(0, 0, width, actual_height)
    context.fill()
    
    # Draw grid
    context.save()
    context.scale(sc_ratio, sc_ratio)
    ################################################################################################################
    # this part is tricky, for few-shot to work, we actually would want the reveal mask to be cumulative
    # meaning higher ratios include all the revealed cells from lower ratios
    # TODO: we should also set constraints that we should not reveal entire words? OW the metric might be tricky
    # reveal_mask = None
    # if prefill_ratio is not None and '_partial' in filetype:
    #     reveal_mask = [[False]*self.cols for _ in range(self.rows)]
    #     filled_cells = [(r, c) for r in range(self.rows) for c in range(self.cols) 
    #                     if self.grid[r][c] != self.empty]
        
    #     # set a fixed seed based on the grid to make it deterministic
    #     grid_str = ''.join(''.join(row) for row in self.grid)
    #     random.seed(hash(grid_str))
        
    #     # shuffle once and use the same order for all ratios
    #     all_cells = filled_cells.copy()
    #     random.shuffle(all_cells)
    #     # calculate how many cells to reveal for current ratio
    #     num_to_reveal = int(len(filled_cells) * prefill_ratio)
    #     # take cells from the beginning up to num_to_reveal
    #     revealed = all_cells[:num_to_reveal]
    #     for r, c in revealed:
    #         reveal_mask[r][c] = True
    #     # reset
    #     random.seed()
    reveal_mask = None
    if (prefill_ratio is not None or left_one_out) and ('_partial' in filetype or '_left_one_out' in filetype):
        reveal_mask = [[False]*self.cols for _ in range(self.rows)]
        cell_to_words = {}  
        word_cells = {}
        for word_idx, word_entry in enumerate(self.wordlist):
            word_text = word_entry[0]
            y_pos = word_entry[2]  # Starting y position
            x_pos = word_entry[3]  # Starting x position
            orientation = word_entry[4]  # 0 for across, 1 for down
            cells = []
            for i in range(len(word_text)):
                if orientation == 0:
                    cell = (y_pos, x_pos + i)
                else:
                    cell = (y_pos + i, x_pos)
                cells.append(cell)
                if cell not in cell_to_words:
                    cell_to_words[cell] = []
                cell_to_words[cell].append(word_idx)
            word_cells[word_idx] = cells

        filled_cells = [(r, c) for r in range(self.rows) for c in range(self.cols) 
                    if self.grid[r][c] != self.empty]
        grid_str = ''.join(''.join(row) for row in self.grid)
        random.seed(hash(grid_str))

        if left_one_out:
            # "left one out" mode - reveal all words except one randomly chosen word
            # randomly select one word to leave out
            for r, c in filled_cells:
                reveal_mask[r][c] = True
            word_indices = list(range(len(self.wordlist)))
            random.shuffle(word_indices)
            hidden_word_idx = word_indices[0]
            for r, c in word_cells[hidden_word_idx]:
                reveal_mask[r][c] = False
        else:
            all_cells = filled_cells.copy()
            random.shuffle(all_cells)
            num_to_reveal = int(len(filled_cells) * prefill_ratio)
            def would_complete_word(cell):
                if cell not in cell_to_words:
                    return False
                    
                # temporarily mark this cell as revealed
                row, col = cell
                reveal_mask[row][col] = True
                
                # check all words this cell belongs to
                for word_idx in cell_to_words[cell]:
                    # count revealed cells in this word
                    revealed = sum(reveal_mask[r][c] for r, c in word_cells[word_idx])
                    total = len(word_cells[word_idx])
                    
                    # check if too many cells would be revealed (leave at least 2 unrevealed)
                    if revealed > total - 1:
                        reveal_mask[row][col] = False
                        return True
                        
                reveal_mask[row][col] = False
                return False
            # reveal cells while respecting constraints
            revealed_count = 0
            for cell in all_cells:
                if revealed_count >= num_to_reveal:
                    break
                    
                # skip if revealing this cell would complete any word
                if would_complete_word(cell):
                    continue
                    
                # reveal the cell
                row, col = cell
                reveal_mask[row][col] = True
                revealed_count += 1
        
        random.seed()

        # create partial grid 
        partial_grid = [[self.empty]*self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                if reveal_mask[r][c]:
                    partial_grid[r][c] = self.grid[r][c]
        
        # Store the partial grid and reveal mask
        self._last_partial_grid = partial_grid
        self._last_reveal_mask = reveal_mask
    ################################################################################################################
    correctness = self.draw_img(name, context, 28, xoffset, 80, RTL, reveal_mask, filled_words)
    context.restore()
    
    if include_clues:
        # Draw clues with adjusted font sizes
        context.set_source_rgb(0, 0, 0)
        x, y = 36, yoffset+5+(self.rows*px*sc_ratio)
        
        # Draw "Across" header
        across_font_size = int(12 * font_scale)
        clue_font_size = int(9 * font_scale)
        line_height = standard_line_height * font_scale
        
        self.draw_letters(lang[0], context, x, y, f'Sans {across_font_size} bold')
        y += line_height * 1.5

        for line in clue_lines:
            if line.strip() == lang[1]:  # "Down" section header
                y += line_height * 0.5
                self.draw_letters(lang[1], context, x, y, f'Sans {across_font_size} bold')
                y += line_height * 1.5
                continue
                
            self.draw_letters(line, context, x, y, f'Serif {clue_font_size}')
            y += line_height

    context.show_page()
    surface.finish()
    return correctness

############################################################
# Patching create_files to support new parameters
############################################################
@patch_class_method(Exportfiles, 'create_files')
def create_files(
    self, 
    name, 
    save_format, 
    lang, 
    message, 
    prefill_ratio=0,
    filled_words=None,
    left_one_out=False,
):
    flags = {}
    partial_data = None
    assert 'p' in save_format, 'PDF format is required for generating data'
    if Pango.find_base_dir(self.wordlist[0][0], -1) == Pango.Direction.RTL:
        [i.reverse() for i in self.grid]
        RTL = True
    else:
        RTL = False
    img_files = ''
    if 'p' in save_format:
        self.export_pdf(name, '_grid.pdf', lang, RTL)
        self.export_pdf(name, '_key.pdf', lang, RTL)
        # only grid without clues
        self.export_pdf(name, '_grid_only.pdf', lang, RTL, include_clues = False)
        # partial filled pdf
        if prefill_ratio > 0:
            self.export_pdf(name, f'_partial_{str(prefill_ratio)}.pdf', lang, RTL, prefill_ratio=prefill_ratio)
            img_files += name + f'_partial_{str(prefill_ratio)}.pdf '
            # store partial grid data if available
            if hasattr(self, '_last_partial_grid') and hasattr(self, '_last_reveal_mask'):
                partial_data = {
                    'grid': self._last_partial_grid,
                    'mask': self._last_reveal_mask
                }
        if left_one_out:
            self.export_pdf(name, '_left_one_out.pdf', lang, RTL, left_one_out=True)
            img_files += name + '_left_one_out.pdf '
            # store partial grid data if available
            if hasattr(self, '_last_partial_grid') and hasattr(self, '_last_reveal_mask'):
                partial_data = {
                    'grid': self._last_partial_grid,
                    'mask': self._last_reveal_mask
                }
        if filled_words is not None:
            correctness = self.export_pdf(name, '_filled.pdf', lang, RTL, filled_words=filled_words)
            flags['filled'] = correctness
            img_files += name + '_filled.pdf '
            # store filled grid data if available
            if hasattr(self, '_last_filled_grid') and hasattr(self, '_last_word_mask'):
                partial_data = {
                    'grid': self._last_filled_grid,
                    'mask': self._last_word_mask
                }
        img_files += name + '_grid.pdf ' + name + '_key.pdf '
    if 'l' in save_format:
        self.export_pdf(name, 'l_grid.pdf', lang, RTL, 612, 792)
        self.export_pdf(name, 'l_key.pdf', lang, RTL, 612, 792)
        img_files += name + 'l_grid.pdf ' + name + 'l_key.pdf '
    if 'n' in save_format:
        self.create_img(name + '_grid.png', RTL)
        self.create_img(name + '_key.png', RTL)
        img_files += name + '_grid.png ' + name + '_key.png '
    if 's' in save_format:
        self.create_img(name + '_grid.svg', RTL)
        self.create_img(name + '_key.svg', RTL)
        img_files += name + '_grid.svg ' + name + '_key.svg '
    if 'n' in save_format or 's' in save_format:
        self.clues_txt(name + '_clues.txt', lang)
        img_files += name + '_clues.txt'
    if 'z' in save_format:
        out = name + '.ipuz'
        self.write_ipuz(name=name, filename=out, lang=lang)
        img_files += out
    if message:
        print(message + img_files)
    return flags, partial_data

############################################################
# Patch clues_txt function
# it should return a json object for easy parsing
############################################################
@patch_class_method(Exportfiles, 'legend')
def legend(self, lang):
    outStrA = u'\nClues\n{}\n'.format(lang[0])
    outStrD = u'{}\n'.format(lang[1])
    across_clues = {}
    down_clues = {}
    for word in self.wordlist:
        clue = word[1]
        number = word[5]
        answer = word[0]
        
        if word[4]:  # Down
            outStrD += u'{:d}. {}\n'.format(number, clue)
            down_clues[str(number)] = {"clue": clue, "answer": answer}
        else:  # Across
            outStrA += u'{:d}. {}\n'.format(number, clue)
            across_clues[str(number)] = {"clue": clue, "answer": answer}
            
    self._across_clues = across_clues
    self._down_clues = down_clues
    
    return outStrA + outStrD

@patch_class_method(Exportfiles, 'clues_txt')
def clues_txt(self, name, lang):
    # Write original text file
    with open(name, 'w') as clues_file:
        clues_file.write(self.word_bank())
        clues_file.write(self.legend(lang))
    
    # Write answers in JSON format
    answers_file = name.replace('clues.txt', 'answers.json')
    entries = []
    # Write Across clues
    if hasattr(self, '_across_clues'):
        for number, data in sorted(self._across_clues.items(), key=lambda x: int(x[0])):
            entry = {
                "direction": f"across {number}",
                "clue": data["clue"],
                "answer": data["answer"]
            }
            entries.append(entry)
    
    # Write Down clues
    if hasattr(self, '_down_clues'):
        for number, data in sorted(self._down_clues.items(), key=lambda x: int(x[0])):
            entry = {
                "direction": f"down {number}",
                "clue": data["clue"],
                "answer": data["answer"]
            }
            entries.append(entry)
    with open(answers_file, 'w') as f:
        json.dump(entries, f, indent=2)

############################################################
# wlist is very slow for large wordlist
# we will use a faster implementation
# TODO: check if this is valid
# org: https://github.com/riverrun/genxword/blob/master/genxword/control.py
############################################################
def apply_word_mixer_chunk(words_chunk, word_mixer):
    results = []
    for word, ignored in words_chunk:  # Ignore original clue
        mixed = word_mixer(word.lower())
        results.append([word, mixed])
    return results

def chunkify(file_object, chunk_size=1000):
    chunk = []
    for line in file_object:
        chunk.append(line)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def process_chunk(chunk):
    results = []
    for line in chunk:
        line = line.strip()
        if not line:
            continue
        parts = line.split(' ', 1)
        if len(parts) < 2:
            continue
        word = ComplexString(parts[0].upper())
        clue = parts[-1]
        results.append([word, clue])
    return results

@patch_class_method(Genxword, 'wlist')
def wlist(self, words, nwords=50):
    """Process words in parallel and create word-clue pairs."""
    num_workers = min(8, mp.cpu_count() * 2)
    chunk_size = 2048
    process_func = partial(process_chunk)
    all_results = []
    futures = []
    chunks = list(chunkify(words, chunk_size))
    total_chunks = len(chunks)
    words.seek(0)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=total_chunks, desc="Submitting chunks for vocab") as pbar:
            for chunk in chunkify(words, chunk_size):
                futures.append(executor.submit(process_func, chunk))
                pbar.update(1)
        
        with tqdm(total=len(futures), desc="Processing chunks for vocab") as pbar:
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    pbar.update(1)
                    continue

    if len(all_results) > nwords:
        print(f"\nSampling {nwords} words from {len(all_results)} processed words")
        all_results = random.sample(all_results, nwords)

    all_results.sort(key=lambda i: len(i[0]), reverse=True)
    
    # word mixing at the end, exactly like the original
    if self.mixmode:
        # Split results into chunks for parallel processing
        result_chunks = [all_results[i:i + chunk_size] 
                        for i in range(0, len(all_results), chunk_size)]
        futures = []
        mixed_results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=len(result_chunks), desc="Mixing words") as pbar:
                for chunk in result_chunks:
                    futures.append(
                        executor.submit(
                            apply_word_mixer_chunk, 
                            chunk,
                            self.word_mixer
                        )
                    )
                    pbar.update(1)
                for future in as_completed(futures):
                    try:
                        mixed_chunk = future.result()
                        mixed_results.extend(mixed_chunk)
                    except Exception as e:
                        print(f"Error in word mixing: {e}")
                        continue         
        all_results = mixed_results

    self.wordlist = all_results
    return self.wordlist

############################################################ 
# Patching the old grid size function
# this allows us to control the grid size (difficulty)
############################################################
@patch_class_method(Genxword, 'grid_size')
def grid_size(
    self,
    nrow=None,
    ncol=None,
    gtkmode=False,
):
    if nrow is None or ncol is None:
        # calculate the default grid size based on number of words
        if len(self.wordlist) <= 20:
            self.nrow = self.ncol = 17
        elif len(self.wordlist) <= 100:
            self.nrow = self.ncol = int((round((len(self.wordlist) - 20) / 8.0) * 2) + 19)
        else:
            self.nrow = self.ncol = 41
        # ensure grid size is at least as large as the longest word + 2
        if min(self.nrow, self.ncol) <= len(self.wordlist[0][0]):
            self.nrow = self.ncol = len(self.wordlist[0][0]) + 2
    else:
        # use provided values
        self.nrow, self.ncol = nrow, ncol
    if not gtkmode:
        self.check_grid_size(f"{self.nrow}, {self.ncol}")
    # print(f"Grid size: {self.nrow} x {self.ncol}")

############################################################
# Patching the old gengrid function
# it will automatically increase the grid size if calc.best_wordlist
# is less than 90% of the total wordlist
# we do not want this behavior
############################################################
@patch_class_method(Genxword, 'gengrid')
def gengrid(self, silent=False, time_permitted=5.00):
    if not silent:
        print('='*40)
        print('Calculating your crossword...')
    calc = Crossword(self.nrow, self.ncol, '-', self.wordlist)
    calc.compute_crossword(time_permitted)
    #print(result)
    if self.auto:
        if not silent:
            print('Auto mode enabled: proceeding with the current grid size.')
            print('[INFO] We removed the 0.9 constraint')
    else:
        # Manual confirmation for the solution
        h = input('Are you happy with this solution? [Y/n] ')
        if h.strip() == _('n'):
            print('Grid generation aborted by the user.')
            return
    #lang = _('Across/Down').split('/')
    #message = _('The following files have been saved to your current working directory:\n')
    #exp = Exportfiles(self.nrow, self.ncol, calc.best_grid, calc.best_wordlist, '-')
    #exp.create_files(name, saveformat, lang, message)
    return calc.best_grid, calc.best_wordlist, calc.best_score



