import os
import re
import sys
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm
import openai
import requests
from pydantic import BaseModel, Field, create_model, validator

from eval.constants import CLAUDE_SYSTEM_PROMPT, PARSING_PROMPT
from eval.template import ImageInput, PromptType, TemplateFactory
from utils import setup_logger

MAX_RETRIES = 3
DEFAULT_PARSING_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 12000
MAX_STAGNANT_TURNS = 3

# litellm._turn_on_debug()
litellm.drop_params = True

class ClueAnswer(BaseModel):
    """Model for a crossword clue answer."""
    length: Optional[int] = Field(
        None,
        description="Optional. The declared length of the crossword answer as an integer, if provided."
    )
    answer: str = Field(
        ...,
        description="The crossword answer in uppercase without spaces. Only combine multiple words into a single word if necessary."
    )

    @validator('answer')
    def answer_no_spaces(cls, v):
        if re.search(r'\s', v):
            raise ValueError("answer must not contain any spaces or whitespace characters")
        return v
    

@dataclass
class ModelConfig:
    """Configuration for a VLM model."""
    model_name: str
    use_vllm: bool = False
    lora_module: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    parsing_model_name: str = 'gpt-4-turbo'
    template_type: str = "direct"
    output_dir: Optional[Path] = None


class Model(ABC):
    """Abstract base class for all models."""
    @abstractmethod
    def __call__(self, request: dict[str, Any]) -> str:
        raise NotImplementedError
    

class EvalModel(Model):
    def __init__(self, config: ModelConfig, anagram_mix=False):
        self.logger = setup_logger(verbose=False)
        self._initialize_model_settings(config, anagram_mix)

    def _initialize_model_settings(self, config: ModelConfig, anagram_mix: bool) -> None:
        """Initialize model-specific settings."""
        # Set model name with vllm prefix if needed
        self.model_name = f"hosted_vllm/{config.model_name}" if config.use_vllm else config.model_name
        self.use_vllm = config.use_vllm
        self.lora_module = config.lora_module
        self.api_base = config.api_base
        self.api_key = config.api_key
        self.parsing_model_name = config.parsing_model_name or 'gpt-4-turbo'
        self.template = TemplateFactory.create_template(config.template_type, anagram_mix=anagram_mix)
        self.output_dir = config.output_dir
        self._claude_warning = False
        self.parsing_prompt = None
        
        # Set model-specific stop tokens
        self.stop = self._get_model_stop_tokens()
        
        # Check server health if using vLLM
        if self.use_vllm:
            self.logger.info("Using vLLM model, verifying API base and key")
            self._wait_till_healthy()
        else:
            self.logger.info("Using an API model - be mindful of potential costs")

    def _get_model_stop_tokens(self) -> Optional[List[str]]:
        """Get model-specific stop tokens."""
        if 'Aria' in self.model_name:
            return ['<|im_end|>']
        return None
    
    def _wait_till_healthy(self) -> bool:
        base_url = self.api_base.split("/v1")[0] if self.api_base is not None else None
        # wait for server to be ready
        if base_url is None:
            self.logger.warning("No base url is provided, will not check server health")
            return True
        health_endpoint = f"{base_url}/health"
        timeout = 300
        t0 = time.time()
        self.logger.info(f"Waiting for vLLM server to come online at {health_endpoint} ...")
        self.logger.info(f"Timeout is {timeout}s")
        while time.time() - t0 < timeout:
            if int(time.time() - t0) % 5 == 0:
                self.logger.info(f"Waiting for server ({int(time.time() - t0)}s) ...")
            try:
                req = requests.get(health_endpoint)
                self.logger.info("Server is up! \U0001F92A")
            except Exception:
                pass
            else:
                if (
                    req.status_code == 200
                    and req.content == b""
                    or req.json() == {"status": "OK"}
                ):
                    return True
            time.sleep(1)
        raise RuntimeError(
            f"Server not up in {int(timeout / 60)} minutes, something is wrong"
        )
    
    @staticmethod
    def create_dynamic_puzzle_model(reference_answer: List[Dict[str, Any]]) -> BaseModel:
        """
        Create a Pydantic model dynamically based on reference answers.
        Each dictionary in reference_answer should have a 'direction' key 
        with format like "across 1" or "down 1".
        """
        fields = {}
        pattern = re.compile(r'^(across|down)\s+\d+$', re.IGNORECASE)
        
        # convert tuple back to list of dicts
        for answer in reference_answer:
            key = answer['direction']
            if not pattern.match(key):
                raise ValueError(f"Reference key '{key}' does not match expected format ('across X' or 'down X')")
                
            fields[key] = (Optional[ClueAnswer], Field(
                None,
                description=f"Answer for clue at '{key}'."
            ))
            
        # create the model
        DynamicPuzzleModel = create_model(
            'DynamicPuzzleModel',
            **fields
        )
        
        return DynamicPuzzleModel
    
    # https://docs.litellm.ai/docs/completion/output
    # we should format the response to a more readable json format
    def _format_response(self, response: Any, add_correctness: bool = False) -> Dict[str, Any]:
        if not isinstance(response, dict):
            json_response = response.json()
        else:
            json_response = response
        json_response['json_response'] = None
        if 'cost' not in json_response:
            if not self.use_vllm:
                try:
                    json_response['cost'] = litellm.completion_cost(completion_response=response)
                except Exception:
                    json_response['cost'] = 0
            else:
                json_response['cost'] = 0
    
        json_response['parsing_model'] = self.parsing_model_name
        
        if add_correctness:
            json_response['interactive_correctness'] = []
            
        return json_response

    # we should let the model to produce freeform text
    # then use another parser to extract the answer in json format
    # we should move this to metric module, however, we have multi-turn and interactive mode
    # which requires intermediate parsing
    def _parse_response(
        self, 
        response_text: str, 
        id: str, 
        dynamic_puzzle_model: Optional[BaseModel] = None
    ) -> Dict[str, Any]:
        # R1
        try:
            response_text = response_text.split('</think>')[1]
        except (IndexError, AttributeError):
            pass
        # reka
        try:
            response_text = response_text.split('</reasoning>')[1]
        except (IndexError, AttributeError):
            pass
        
        # check for repetition first
        # usually these are long and meaninglessful, feed to gpt just waste money
        # if self._has_consecutive_repetitions(response_text):
        #     self.logger.warning(f"Detected repetitive patterns in model response {id}, skipping parsing")
        #     return {}
        
        for _ in range(MAX_RETRIES):
            try:
                # Parse with GPT
                parsed_response = self._gpt_parse(response_text, dynamic_puzzle_model)
                
                # Process and validate the response
                validated_data = parsed_response.model_dump()
                validated_data = {
                    key.lower(): value 
                    for key, value in validated_data.items() 
                    if value is not None
                }
                
                # Post-process: set defaults and standardize format
                for key, value in validated_data.items():
                    if value.get('length') is None:
                        validated_data[key]['length'] = -100
                    if 'answer' in value and value['answer'] is not None:
                        validated_data[key]['answer'] = value['answer'].upper()
                        
                return validated_data
                
            except Exception as e:
                self.logger.error(f"ID {id}: Error parsing response: {e}")
                self.logger.error("Full error:", exc_info=True)
                
        return {}  # return empty dict if all parsing attempts fail
    
    def _has_consecutive_repetitions(
        self,
        text: str,
        ngram_size: int = 10,
        consecutive_threshold: int = 3,
        frequency_threshold: float = 0.15,
        min_text_length: int = 50
    ) -> bool:
        """
        Detects repetitive patterns in the text using two methods:
        1. Consecutive identical n-grams.
        2. Overall frequency of any n-gram relative to total n-grams.
        
        Args:
            text (str): The input text to check.
            ngram_size (int): Size of each n-gram (default is 1 for single words).
            consecutive_threshold (int): Number of consecutive identical n-grams required to flag repetition.
            frequency_threshold (float): Ratio threshold for overall frequency to flag repetition.
            min_text_length (int): Minimum length (in characters) for text to be checked.
            
        Returns:
            bool: True if repetition is detected, False otherwise.
        """
        cleaned_text = re.sub(r'\s+', ' ', text).strip().lower()
        if len(cleaned_text) < min_text_length:
            return False
        
        words = re.findall(r'\w+', cleaned_text)
        if len(words) < ngram_size:
            return False

        previous_ngram = tuple(words[:ngram_size])
        consecutive_count = 1
        for i in range(1, len(words) - ngram_size + 1):
            current_ngram = tuple(words[i:i+ngram_size])
            if current_ngram == previous_ngram:
                consecutive_count += 1
                if consecutive_count >= consecutive_threshold:
                    return True
            else:
                consecutive_count = 1
                previous_ngram = current_ngram

        ngrams = [' '.join(words[i:i+ngram_size]) for i in range(len(words) - ngram_size + 1)]
        if len(ngrams) < 10:
            return False  
        
        counts = Counter(ngrams)
        total = len(ngrams)
        for ngram, count in counts.items():
            if (count - 1) / total > frequency_threshold:
                return True
                
        return False
    
    # gpt beta feature
    def _gpt_parse(self, inputs: str, response_schema: BaseModel) -> Any:
        api_key = os.environ["OPENAI_API_KEY"]
        engine = openai.OpenAI(api_key=api_key)
        
        # default reasoning effort is medium
        # which should be enough for parsing
        response = engine.beta.chat.completions.parse(
            model=self.parsing_model_name,
            messages=[
                {"role": "system", "content": self.parsing_prompt},
                {"role": "user", "content": inputs}
            ],
            response_format=response_schema,
            timeout=DEFAULT_TIMEOUT,
        )
        # total is not long
        # so should be safe to use o3
        # print(response)
        return response.choices[0].message.parsed
    
    def _make_model_call(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Optional[Any]:
        kwargs.pop('mm_limit', None)
        kwargs.pop('id', None)
        if self.stop:
            kwargs['stop'] = self.stop
        for retry in range(MAX_RETRIES):
            try:
                if self.lora_module is not None:
                    headers = {
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {self.api_key}'
                    }
                    data = {
                        "model": self.lora_module,
                        "messages": messages,
                        "temperature": kwargs.get('temperature', 0.0),
                        "top_p": kwargs.get('top_p', 1.0),
                        "max_completion_tokens": kwargs.get('max_completion_tokens', 8192),
                    }
                    response = requests.post(self.api_base, json=data, headers=headers)
                    response = response.json()
                else:
                    response = litellm.completion(
                        model=self.model_name,
                        messages=messages,
                        num_retries=MAX_RETRIES,
                        timeout=DEFAULT_TIMEOUT,
                        api_base=self.api_base,
                        api_key=self.api_key,
                        # stream=True,            # this is for debug purposes, sometimes the generation hangs, you can use this check if it is generating
                        **kwargs
                    )
                    # chunks = []
                    # for chunk in response: 
                    #     chunks.append(chunk)
                    #     print(chunk.choices[0].delta.content, end="")
                    # response = litellm.stream_chunk_builder(chunks, messages=messages)

                if response is not None:
                    return response
                    
            except litellm.NotFoundError as e:
                self.logger.error(f"LiteLLM NotFoundError: {str(e)}")
                sys.exit(1)
                
            except Exception as e:
                self.logger.warning(f"Attempt {retry + 1}/{MAX_RETRIES} failed: {str(e)}")
                self.logger.error("Full error:", exc_info=True)  
                
        self.logger.error(f"All {MAX_RETRIES} attempts failed")
        return None

    def _update_response(self, base_response: Dict[str, Any], new_response: Dict[str, Any]) -> Dict[str, Any]:
        base_usage = base_response.get('usage', {})
        new_usage = new_response.get('usage', {})
        
        for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
            base_usage[key] = (base_usage.get(key, 0) or 0) + (new_usage.get(key, 0) or 0)
        
        base_response['cost'] = (base_response.get('cost', 0) or 0) + (new_response.get('cost', 0) or 0)
        
        if 'json_response' not in base_response:
            base_response['json_response'] = {}
        
        if new_response.get('json_response'):
            if base_response['json_response'] is None:
                base_response['json_response'] = new_response['json_response']
            else:
                base_response['json_response'].update(new_response['json_response'])
        
        base_response['choices'].extend(new_response['choices'])
        
        return base_response

    def _process_single_turn(
        self,
        messages: List[Dict[str, str]],
        parse_json: bool,
        dynamic_puzzle_model: Optional[BaseModel] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Process a single-turn interaction with the model."""
        id = kwargs.pop('id', '0')
        
        response = self._make_model_call(messages, **kwargs)
        if response is None:
            return None
            
        formatted_response = self._format_response(response)
        
        if parse_json and dynamic_puzzle_model:
            json_response = self._parse_response(
                response['choices'][0]['message']['content'],
                id, 
                dynamic_puzzle_model
            )
            formatted_response['json_response'] = json_response
            
        return formatted_response

    def _process_two_turn(
        self,
        messages: List[Dict[str, str]],
        previous_response: Dict[str, Any],
        reference_answer: List[Dict[str, Any]],
        parse_json: bool,
        dynamic_puzzle_model: Optional[BaseModel] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if reference_answer is None:
            raise ValueError("Reference answer is required for two-turn prompt")
            
        follow_up_message = self.template.format_follow_up(
            messages,                   # user
            previous_response,          # assistant
        )
        # second round
        response = self._process_single_turn(
            follow_up_message, 
            parse_json, 
            dynamic_puzzle_model, 
            **kwargs
        )
        
        accumulated_response = self._update_response(previous_response, response)
        
        return accumulated_response
    
    def _process_multi_turn(
        self,
        messages: List[Dict[str, str]],
        parse_json: bool,
        puzzle_state: Dict[str, Any],
        dynamic_puzzle_model: Optional[BaseModel] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Interative Mode"""
        filled_words = set()
        accumulated_response = None
        puzzle_len = puzzle_state.get('length', 100)
        turn = 1
        id = kwargs.pop('id', '0')
        stagnant_turns = 0
        prev_filled_words_len = 0
        
        # temporary workspace directory
        temp_dir = (Path(self.output_dir) / 'workspace' / str(puzzle_state['meta_data'].get('id', 'temp')))
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        while True:
            response = self._make_model_call(messages, **kwargs)
            if response is None:
                return None
                
            current_response = self._format_response(response, add_correctness=True)
            
            if parse_json and dynamic_puzzle_model:
                json_response = self._parse_response(
                    response['choices'][0]['message']['content'],
                    id, 
                    dynamic_puzzle_model
                )
                current_response['json_response'] = json_response
                
                if json_response:
                    filled_words.update(
                        value_dict['answer'] for value_dict in json_response.values()
                    )
            
            if len(filled_words) == prev_filled_words_len:
                stagnant_turns += 1
                if stagnant_turns >= MAX_STAGNANT_TURNS:
                    self.logger.warning(
                        f"ID {id}: No new words added for {MAX_STAGNANT_TURNS} turns, stopping."
                    )
                    return accumulated_response
            else:
                stagnant_turns = 0
                prev_filled_words_len = len(filled_words)
            
            if accumulated_response is None:
                accumulated_response = current_response
            else:
                accumulated_response = self._update_response(accumulated_response, current_response)
            
            if len(set(filled_words)) >= puzzle_len:
                self.logger.warning(
                    f"ID {id}: Filled answers ({len(filled_words)}) reached/exceeded "
                    f"puzzle length ({puzzle_len}). This is unlikely, check the model response."
                )
                return accumulated_response
            
            # prepare next round
            messages, is_correct = self.template.format_follow_up(
                conversation_history=messages,
                previous_response=response,
                temp_dir=temp_dir,
                filled_words=filled_words,
                puzzle_state=puzzle_state,
                turn=turn,
                mm_limit=kwargs.get('mm_limit', 10),
            )
            
            accumulated_response['interactive_correctness'].append(is_correct)
            if not is_correct:
                self.logger.warning(f"ID {id}: Found incorrect filled words, stopping.")
                return accumulated_response
                
            turn += 1

    def _maybe_add_claude_system_prompt(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if 'claude' in self.model_name:
            if not self._claude_warning:
                self.logger.warning(
                    "Claude may terminate early on larger puzzles. "
                    "A system prompt is added to mitigate this issue."
                )
                self._claude_warning = True
            
            return [{"role": "system", "content": CLAUDE_SYSTEM_PROMPT}] + messages

        return messages
    
    def __call__(
        self,
        grid_image: Optional[ImageInput] = None,
        grid_only_image: Optional[ImageInput] = None,
        partial_images: List[ImageInput] = [],
        partial_grids: List[List[str]] = [],
        parse_json: bool = False,
        puzzle_state: Optional[Dict[str, Any]] = None,
        clues: Optional[List[str]] = None,
        key_image: Optional[ImageInput] = None,
        previous_response: Optional[Dict[str, Any]] = None,
        reference_answer: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        
        dynamic_puzzle_model = None
        assert reference_answer is not None, "Reference answer is required for parsing"
        assert clues is not None, "Clues are required for parsing and IMG_COT_GRID_ONLY Mode"
        if reference_answer:
            # ref_tuple = tuple(item['direction'] for item in reference_answer)
            dynamic_puzzle_model = self.create_dynamic_puzzle_model(reference_answer)
        #self.parsing_prompt = PARSING_PROMPT.format(clues = '\n'.join(clues))
        self.parsing_prompt = PARSING_PROMPT.format(clues = '\n'.join(clues))

        if self.template.prompt_type == PromptType.SINGLE_TURN_IMG:
            # image-based single turn
            messages = self.template.format_prompt(
                grid_image=grid_image, 
                partial_images=partial_images, 
                grid_only_image=grid_only_image, 
                clues=clues
            )
            messages = self._maybe_add_claude_system_prompt(messages)
            return self._process_single_turn(messages, parse_json, dynamic_puzzle_model, **kwargs)
        
        elif self.template.prompt_type == PromptType.MULTI_TURN_IMG:
            # image-based multi-turn interactive
            messages = self.template.format_prompt(grid_image)
            return self._process_multi_turn(messages, parse_json, puzzle_state, dynamic_puzzle_model, **kwargs)
        
        elif self.template.prompt_type == PromptType.SINGLE_TURN_TEXT:
            # text-based single turn
            messages = self.template.format_prompt(puzzle_state, clues, partial_grids)
            messages = self._maybe_add_claude_system_prompt(messages)
            return self._process_single_turn(messages, parse_json, dynamic_puzzle_model, **kwargs)
        
        elif self.template.prompt_type == PromptType.TWO_TURN_TEXT:
            # text-based two-turn
            messages = self.template.format_prompt(puzzle_state, clues, partial_grids)
            messages = self._maybe_add_claude_system_prompt(messages)
            return self._process_two_turn(
                messages, 
                previous_response, 
                reference_answer, 
                parse_json, 
                dynamic_puzzle_model, 
                **kwargs
            )
        
        elif self.template.prompt_type == PromptType.TWO_TURN_IMG:
            # image-based two-turn
            messages = self.template.format_prompt(grid_image, partial_images)
            return self._process_two_turn(
                messages, 
                previous_response, 
                reference_answer, 
                parse_json, 
                dynamic_puzzle_model, 
                **kwargs
            )
        
        elif self.template.prompt_type == PromptType.EXTRACTION:
            # OCR
            messages = self.template.format_prompt(key_image)
            return self._process_single_turn(messages, parse_json, dynamic_puzzle_model, **kwargs)
        
        raise ValueError(f"Invalid prompt type: {self.template.prompt_type}")

    
