import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from vllm.multimodal.utils import encode_image_base64

from eval.constants import (ANAGRAM_SUFFIX, IMG_COT_GRID_ONLY_PROMPT,
                            IMG_COT_PROMPT, IMG_COT_PROMPT_PREFILLED,
                            IMG_SHOT_PROMPT, IMG_VOT_PROMPT,
                            INTERACTIVE_FOLLOWUP_PROMPT, INTERACTIVE_PROMPT,
                            OCR_EXTRACT_ALL_PROMPT, REFLECTION,
                            TEXT_COT_PROMPT, TEXT_SHOT_PROMPT, 
                            TEXT_COT_PROMPT_PREFILLED, TEXT_VOT_PROMPT)
from utils import gen_grid, post_process_save_file, setup_logger


class PromptType(Enum):
    SINGLE_TURN_IMG = auto()
    MULTI_TURN_IMG = auto()
    SINGLE_TURN_TEXT = auto()
    TWO_TURN_TEXT = auto()
    TWO_TURN_IMG = auto()
    EXTRACTION = auto()

ImageInput = Union[str, Path, Image.Image]
MessageType = Dict[str, Any]
MessagesType = List[MessageType]


class BaseTemplate(ABC):
    def __init__(self, anagram_mix: bool = False):
        self.anagram_mix = anagram_mix
    @property
    @abstractmethod
    def prompt_type(self) -> PromptType:
        pass

    def _get_prompt_text(self, prompt: str) -> str:
        """Get the prompt text with anagram suffix if enabled."""
        if self.anagram_mix:
            return prompt + '\n\n' + ANAGRAM_SUFFIX
        return prompt
    
    def _encode_image(self, image):
        if isinstance(image, Image.Image):
            return encode_image_base64(image)
        image_path = Path(image)
        with Image.open(image_path) as img:
            return encode_image_base64(img)
        
    def _create_message_content(self, text: str, images: Optional[List[ImageInput]] = None) -> List[Dict[str, Any]]:
        content = [{"type": "text", "text": text}]
        if images:
            for image in images:
                base64_image = self._encode_image(image)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        # we use high for gpt, this will be safely ignored by vllm (only supports auto)
                        "detail": "high",
                    },
                })
        return content
    
    @abstractmethod
    def format_prompt(self, *args, **kwargs) -> MessagesType:
        pass

class SingleTurnTemplate(BaseTemplate):
    """Base class for templates that complete in a single interaction."""
    @property
    def prompt_type(self) -> PromptType:
        return PromptType.SINGLE_TURN_IMG
    
class MultiTurnTemplate(BaseTemplate):
    """Base class for templates that require multiple interactions."""
    @property
    def prompt_type(self) -> PromptType:
        return PromptType.MULTI_TURN_IMG
    
    @abstractmethod
    def format_follow_up(self, *args, **kwargs) -> MessagesType:
        """Format follow-up prompts based on conversation history."""
        pass

class TwoRoundMixin:
    """Mixin to add two-round capability to any template."""
    @property
    def prompt_type(self) -> PromptType:
        # override the prompt type to indicate this is a two-round template
        base_type = super().prompt_type
        if base_type == PromptType.SINGLE_TURN_TEXT:
            return PromptType.TWO_TURN_TEXT
        elif base_type == PromptType.SINGLE_TURN_IMG:
            return PromptType.TWO_TURN_IMG
        return base_type
    
    # def _get_criticism_prompt(
    #     self, 
    #     previous_response: Dict[str, Any],
    #     reference_answer: List[Dict[str, Any]],
    # ) -> str:
    #     reference_answer_str = "\n\n".join(
    #         f"{answer['direction']}: {answer['clue']}\nAnswer: {answer['answer']}" for answer in reference_answer
    #     )
    #     return [{
    #         "role": "user",
    #         "content": CRITICISM.replace("<reference>", reference_answer_str).replace("<model_response>", previous_response['choices'][-1]['message']['content'])
    #     }]

    # def format_follow_up(
    #     self,
    #     conversation_history: MessagesType,
    #     previous_response: Dict[str, Any],
    #     feedback: str,
    # ) -> MessagesType:
    #     updated_messages = conversation_history + [
    #         {
    #             "role": "assistant",
    #             "content": previous_response['choices'][-1]['message']['content']
    #         },
    #         {
    #             "role": "user",
    #             "content": REFLECTION.replace("<feedback>", feedback)
    #         }
    #     ]
    #     return updated_messages
    
    def format_follow_up(
        self,
        conversation_history: MessagesType,
        previous_response: Dict[str, Any],
    ) -> MessagesType:
        # for models like r1-distilled, we do not include the reasoning content
        # this is to be consistent with models like o3 which does not return reasoning content
        previous_assistant_response = previous_response['choices'][-1]['message']['content']
        try:
            previous_assistant_response = previous_assistant_response.split('</think>')[1]
        except (IndexError, AttributeError):
            pass
        # likely repetition, we only keep the first 3000 characters
        # just in case
        if len(previous_assistant_response) >= 32768:
            previous_assistant_response = previous_assistant_response[:3000]

        updated_messages = conversation_history + [
            {
                "role": "assistant",
                "content": previous_assistant_response
            },
            {
                "role": "user",
                "content": REFLECTION
            }
        ]
        return updated_messages
#####################################################################
# Image-based Templates
#####################################################################
class IMGCoTTemplate(SingleTurnTemplate):
    def format_prompt(
        self, 
        grid_image: ImageInput, 
        *args, 
        **kwargs
    ) -> MessagesType:
        prompt = self._get_prompt_text(IMG_COT_PROMPT)
        content = self._create_message_content(prompt, [grid_image])
        return [{"role": "user", "content": content}]
    
class IMGShotTemplate(SingleTurnTemplate):
    def format_prompt(
        self, 
        grid_image: ImageInput, 
        partial_images: List[ImageInput], 
        *args, 
        **kwargs
    ) -> MessagesType:
        images = [grid_image] + partial_images
        prompt = self._get_prompt_text(IMG_SHOT_PROMPT)
        content = self._create_message_content(prompt, images)
        return [{"role": "user", "content": content}]

# only 2 demo images
class IMGCoTPrefilledTemplate(SingleTurnTemplate):
    def format_prompt(
        self, 
        grid_image: ImageInput, 
        partial_images: List[ImageInput], 
        *args, 
        **kwargs
    ) -> MessagesType:
        images = [partial_images[-1]]
        prompt = self._get_prompt_text(IMG_COT_PROMPT_PREFILLED)
        content = self._create_message_content(prompt, images)
        return [{"role": "user", "content": content}]
    
class IMGVoTTemplate(SingleTurnTemplate):
    def format_prompt(
        self, 
        grid_image: ImageInput, 
        *args, 
        **kwargs
    ) -> MessagesType:
        prompt = self._get_prompt_text(IMG_VOT_PROMPT)
        content = self._create_message_content(prompt, [grid_image])
        return [{"role": "user", "content": content}]
    
class IMGCoTGridOnlyTemplate(SingleTurnTemplate):
    def format_prompt(
        self, 
        grid_image: ImageInput, 
        grid_only_image: ImageInput,
        clues: List[str],
        *args, 
        **kwargs
    ) -> MessagesType:
        prompt = self._get_prompt_text(IMG_COT_GRID_ONLY_PROMPT)
        prompt = prompt.replace('<clues>', '\n'.join(clues))
        content = self._create_message_content(prompt, [grid_only_image])
        return [{"role": "user", "content": content}]
    

#####################################################################
# Text-Based Prompt
#####################################################################
class TextBasedTemplate(SingleTurnTemplate):
    @property
    def prompt_type(self) -> PromptType:
        return PromptType.SINGLE_TURN_TEXT

    def _format_empty_grid(self, grid: List[List[str]]) -> str:
        """Format grid for text representation."""
        # formatted_grid = "\n".join(" ".join('-' if cell.isalpha() else '*' for cell in row) for row in grid)
        # return f"{formatted_grid}"
        matrix = []
        for row in grid:
            matrix_row = [0 if cell.isalpha() else 1 for cell in row]  # 0 for empty, 1 for blocked
            matrix.append(matrix_row)
        return 'empty grid = ' + str(matrix)
    
    def _format_grid(
        self, 
        partial_grid: List[List[str]], 
        grid: List[List[str]],
        ratio: float = 0.5,
    ) -> str:
        # formatted_grid = "\n".join(" ".join(partial_grid[i][j] if partial_grid[i][j].isalpha() else '*' 
        #             for j, cell in enumerate(row)) 
        #             for i, row in enumerate(grid))
        # return f"{formatted_grid}"
        matrix = []
        for i, row in enumerate(grid):
            matrix_row = []
            for j, cell in enumerate(row):
                if cell.isalpha():
                    if partial_grid[i][j].isalpha():
                        matrix_row.append(f"{partial_grid[i][j]}")
                    else:
                        matrix_row.append(0)
                else:
                    matrix_row.append(1)
            matrix.append(matrix_row)
        return f'grid ({str(ratio)}) = ' + str(matrix)
    
class TextCoTTemplate(TextBasedTemplate):
    def format_prompt(
        self, 
        puzzle_state: Dict[str, Any],
        clues: List[str],
        partial_grids: List[List[str]],
        *args, 
        **kwargs
    ) -> MessagesType:
        grid_text = self._format_empty_grid(puzzle_state['grid'])
        prompt = self._get_prompt_text(TEXT_COT_PROMPT)
        prompt = prompt.replace("<grid>", grid_text)
        prompt = prompt.replace('<clues>', '\n'.join(clues))
        return [{"role": "user", "content": prompt}]
    
class TextShoTTemplate(TextBasedTemplate):
    def format_prompt(
        self, 
        puzzle_state: Dict[str, Any],
        clues: List[str],
        partial_grids: List[List[str]],
        *args, 
        **kwargs
    ) -> MessagesType:
        grid_text = self._format_empty_grid(puzzle_state['grid'])
        prompt = self._get_prompt_text(TEXT_SHOT_PROMPT)
        prompt = prompt.replace("<grid>", grid_text)
        for i, grid in enumerate(partial_grids):
            prompt = prompt.replace(f"<grid{i}>", self._format_grid(grid, puzzle_state['grid'], ratio = (i + 1) * 0.25))
        prompt = prompt.replace('<clues>', '\n'.join(clues))
        return [{"role": "user", "content": prompt}]

class TextCoTPrefilledTemplate(TextBasedTemplate):
    def format_prompt(
        self, 
        puzzle_state: Dict[str, Any],
        clues: List[str],
        partial_grids: List[List[str]],
        *args, 
        **kwargs
    ) -> MessagesType:
        prompt = self._get_prompt_text(TEXT_COT_PROMPT_PREFILLED)
        grid = partial_grids[-1]
        prompt = prompt.replace("<grid>", self._format_grid(grid, puzzle_state['grid'], ratio = 0.75))
        prompt = prompt.replace('<clues>', '\n'.join(clues))
        return [{"role": "user", "content": prompt}]

# https://arxiv.org/pdf/2404.03622
class TextVoTTemplate(TextBasedTemplate):
    def format_prompt(
        self, 
        puzzle_state: Dict[str, Any], 
        clues: List[str],
        partial_grids: List[List[str]], # not used, consistency
        *args, 
        **kwargs
    ) -> MessagesType:
        grid_text = self._format_empty_grid(puzzle_state['grid'])
        prompt = self._get_prompt_text(TEXT_VOT_PROMPT)
        prompt = prompt.replace("<grid>", grid_text)
        prompt = prompt.replace('<clues>', '\n'.join(clues))
        return [{"role": "user", "content": prompt}]


#####################################################################
# Unlike the two-round templates, this template is designed for interactive solving
# It is similar to VoT, instead, we will fill the grid and generate images for the model
#####################################################################
class InteractiveTemplate(MultiTurnTemplate):
    def format_prompt(
        self, 
        grid_image: ImageInput, 
        *args, 
        **kwargs
    ) -> MessagesType:
        prompt = self._get_prompt_text(INTERACTIVE_PROMPT)
        content = self._create_message_content(prompt, [grid_image])
        return [{"role": "user", "content": content}]

    def _update_conversation_history(
            self,
            conversation_history: MessagesType,
            previous_response: Dict[str, Any],
            new_content: str,
            mm_limit: int
        ) -> MessagesType:
            """
            Update conversation history while maintaining image limit.
            Last message is always from user and kept separate from pairs.
            """
            updated_messages = conversation_history + [
                {
                    "role": previous_response.choices[0].message.role,
                    "content": previous_response.choices[0].message.content
                },
                {
                    "role": "user", 
                    "content": new_content,
                }
            ]
            image_count = 0
            final_messages = []
            # always include the last user message
            last_message = updated_messages[-1]
            if isinstance(last_message["content"], list):
                image_count = sum(
                    1 for item in last_message["content"]
                    if isinstance(item, dict) and item.get("type") == "image_url"
                )
            final_messages = [last_message]
            # process remaining messages in assistant/user pairs from newest to oldest
            # system / user / assistant / user / assistant / user ... (backward)
            for i in range(len(updated_messages) - 2, 0, -2):
                if i < 1:  
                    break
                assistant_message = updated_messages[i]
                user_message = updated_messages[i-1]
                if assistant_message["role"] != "assistant" or user_message["role"] != "user":
                    continue        
                # only user msg can have images (at least for these VLMs)
                msg_images = 0
                if isinstance(user_message["content"], list):
                    msg_images = sum(
                        1 for item in user_message["content"]
                        if isinstance(item, dict) and item.get("type") == "image_url"
                    )
                if image_count + msg_images > mm_limit:
                    break
                image_count += msg_images
                final_messages.insert(0, user_message)
                final_messages.insert(1, assistant_message)

            # check, since it might be system prompt
            initial_message = updated_messages[0]
            initial_msg_images = 0
            if isinstance(initial_message["content"], list):
                initial_msg_images = sum(
                    1 for item in initial_message["content"]
                    if isinstance(item, dict) and item.get("type") == "image_url"
                )
            if image_count + initial_msg_images <= mm_limit:
                final_messages.insert(0, initial_message)
            # keep at least 4: system / user / assistant / user
            if len(final_messages) < 4:
                return updated_messages[-4:]
            return final_messages
    
    # we need to use gen_grid to generate the grid based on the answer
    # TODO: make code more clean and efficient, also content window might be short
    # TODO: decide if we need history or every turn is a fresh start
    def format_follow_up(
        self,
        conversation_history: MessagesType,
        previous_response: Dict[str, Any],
        temp_dir: Union[str, Path],
        filled_words: List[str],
        puzzle_state: Dict[str, Any],
        turn: int = 0,
        mm_limit: int = 4, 
    ) -> tuple[MessagesType, int]:
        """Format follow-up prompts for interactive solving."""
        temp_path = Path(temp_dir)
        temp_path.mkdir(exist_ok=True)
        
        _, correctness, _ = gen_grid(
            str(temp_path),
            None,       # generator can be none here
            formats='np',
            filled_words=filled_words,
            puzzle_state=puzzle_state,
            silent=True
        )
        filled_correctness = correctness['filled']
        post_process_save_file(temp_path, remove_pdf=True, use_tqdm=False)
        
        # handle file operations
        filled_image = temp_path / f'crossword_filled_{turn}.png'
        orig_image = temp_path / 'crossword_filled.png'
        orig_image.rename(filled_image)

        # TODO: this might be redundant with VoT
        filled_grid = temp_path / f'filled_words_{turn}.json'
        orig_grid = temp_path / 'filled_words.json'
        orig_grid.rename(filled_grid)

        new_message_content = self._create_message_content(
            INTERACTIVE_FOLLOWUP_PROMPT,
            [str(filled_image)]
        )
        messages = self._update_conversation_history(
            conversation_history,
            previous_response,
            new_message_content,
            mm_limit
        )
        return messages, filled_correctness
    
    
#####################################################################
# Extraction OCR
#####################################################################
class ExtractionTemplate(SingleTurnTemplate):
    @property
    def prompt_type(self) -> PromptType:
        return PromptType.EXTRACTION
    
    def format_prompt(
        self, 
        key_image: ImageInput, 
        *args, 
        **kwargs
    ) -> MessagesType:
        # extraction does not require any anagram suffix
        content = self._create_message_content(OCR_EXTRACT_ALL_PROMPT, [key_image])
        return [{"role": "user", "content": content}]
    

class TemplateFactory:
    """Factory class to create different templates based on the prompt type."""
    _logger = setup_logger(verbose = False)
    _prompts: Dict[str, str] = {
        "img_cot": IMG_COT_PROMPT,
        "img_shot": IMG_SHOT_PROMPT,
        "img_cot_prefilled": IMG_COT_PROMPT_PREFILLED,
        "img_vot": IMG_VOT_PROMPT,
        "img_cot_grid_only": IMG_COT_GRID_ONLY_PROMPT,
        "text_vot": TEXT_VOT_PROMPT,
        "text_shot": TEXT_SHOT_PROMPT,
        "text_cot_prefilled": TEXT_COT_PROMPT_PREFILLED,
        "text_cot": TEXT_COT_PROMPT,
        "interactive": INTERACTIVE_PROMPT,
        "extraction": OCR_EXTRACT_ALL_PROMPT,
        "simpleqa": "",
    }
    _templates: Dict[str, type[BaseTemplate]] = {
        "img_cot": IMGCoTTemplate,
        "img_shot": IMGShotTemplate,
        "img_cot_prefilled": IMGCoTPrefilledTemplate,
        "img_vot": IMGVoTTemplate,
        "img_cot_grid_only": IMGCoTGridOnlyTemplate,
        "interactive": InteractiveTemplate,
        "text_shot": TextShoTTemplate,
        "text_cot_prefilled": TextCoTPrefilledTemplate,
        "text_cot": TextCoTTemplate,
        "text_vot": TextVoTTemplate,
        "extraction": ExtractionTemplate,
    }
    @classmethod
    def _create_two_round_class(cls, base_class: type[BaseTemplate]) -> type[BaseTemplate]:
        """Dynamically create a two-round variant of a template class."""
        class_name = f"TwoRound{base_class.__name__}"
        # create a new class
        return type(class_name, (TwoRoundMixin, base_class), {})

    @classmethod
    def create_template(cls, template_type: str, anagram_mix: bool = False) -> BaseTemplate:
        """Create a template instance based on the template type."""
        is_two_round = template_type.endswith("_two_round")
        if is_two_round:
            base_type = template_type.replace("_two_round", "")
        else:
            base_type = template_type
        base_class = cls._templates.get(base_type)
        if base_class is None:
            available_types = list(cls._templates.keys()) + [
                f"{k}_two_round" for k in cls._templates.keys()
            ]
            raise ValueError(
                f"Unknown template type: {template_type}. "
                f"Available types: {available_types}"
            )
        if is_two_round:
            template_class = cls._create_two_round_class(base_class)
        else:
            template_class = base_class

        template_instance = template_class(anagram_mix)
        if not is_two_round:
            default_prompt = template_instance._get_prompt_text(cls._prompts[base_type])
        else:
            default_prompt = REFLECTION
        separator = '-' * 80
        cls._logger.info("Default prompt for template '%s':\n%s\n%s\n%s", template_type, separator, default_prompt, separator)
        return template_instance

if __name__ == '__main__':
    import json

    from datasets import load_dataset
    data = load_dataset('JixuanLeng/CrossWordBench', 'english', split='7x7')
    for sample in data:
        if sample['id'] == 1:
            break
    puzzle_state = json.loads(sample['puzzle_state'])
    answers = json.loads(sample['reference_answer'])
    partial_grids = [
        json.loads(sample['partial_grid_0.25'])['grid'],
        json.loads(sample['partial_grid_0.5'])['grid'],
        json.loads(sample['partial_grid_0.75'])['grid'],
    ]
    template = TemplateFactory.create_template("text_shot", anagram_mix=False)
    clues = []
    for answer in answers:
        gt_anwer = answer['answer']
        # wordlist: [word, clue, position y, position x, orientation]
        for word in puzzle_state['wordlist']:
            if word[0] == gt_anwer:
                position = str(word[2]) + ', ' + str(word[3])
        direction = answer['direction']
        direction = direction[0].upper() + direction[1:]
        clues.append(f'{direction} ({position}): {answer["clue"]}')
    messages = template.format_prompt(puzzle_state, clues, partial_grids)
    print(messages[0]['content'])

    # reference_answer_str = "\n\n".join(
    #     f"{answer['direction']}: {answer['clue']}\nAnswer: {answer['answer']}" for answer in answers
    # )
    # print(reference_answer_str)