import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional



MAX_RETRIES = 5
MAX_TOKENS = 4096
JUDGE_MODEL = 'deepseek/deepseek-chat'
JUDGE_PARSING_MODEL = 'deepseek/deepseek-chat'

def normalize_string(s: str | None) -> str:
    if s is None:
        return ""
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1].lower()
    return s.lower()

class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def is_count_metric(self) -> bool:
        """Whether this metric counts occurrences rather than computing a score."""
        return False

    @abstractmethod
    def score(
        self,
        model_answer: Dict[str, Any],
        reference_answer: List[Dict[str, Any]],
        puzzle_state: Dict[str, Any],
        interactive_correctness: Optional[List[bool]] = None,
        **kwargs
    ) -> float:
        pass

class CoverageMetric(Metric):
    def _validate_inputs(
        self,
        model_answer: Dict[str, Any],
        reference_answer: List[Dict[str, Any]],
    ) -> bool:
        if not isinstance(reference_answer, list):
            raise ValueError(f"{self.name} requires reference answer as list of correct answers")
        if model_answer:
            return True
        return False
    
    def _get_answer_pairs_by_direction(
        self,
        model_answer: Dict[str, Any],
        reference_answer: List[Dict[str, Any]],
    ) -> List[tuple[str, str]]:
        pairs = {'across': [], 'down': []}
        for ref_item in reference_answer:
            direction = ref_item["direction"].lower()
            ref_word = normalize_string(str(ref_item["answer"]))
            model_word = normalize_string(str(model_answer.get(direction, {}).get('answer', '')))
            if 'across' in direction:
                pairs['across'].append((ref_word, model_word))
            else:
                pairs['down'].append((ref_word, model_word))
        return pairs

    def _get_answer_pairs(
        self,
        model_answer: Dict[str, Any],
        reference_answer: List[Dict[str, Any]],
    ) -> List[tuple[str, str]]:
        pairs = []
        for ref_item in reference_answer:
            direction = ref_item["direction"].lower()
            ref_word = normalize_string(str(ref_item["answer"]))
            model_word = normalize_string(str(model_answer.get(direction, {}).get('answer', '')))
            pairs.append((ref_word, model_word))
        return pairs

class WordCoverageRate(CoverageMetric):
    """Calculate percentage of correctly answered words."""
    @property
    def name(self) -> str:
        return "Word Coverage Rate"
    def score(
        self,
        model_answer,
        reference_answer,
        puzzle_state,
        interactive_correctness=None,
        **kwargs
    ) -> float:
        if not self._validate_inputs(model_answer, reference_answer):
            return {'across': 0, 'down': 0, 'all': 0}
        pairs_by_direction = self._get_answer_pairs_by_direction(model_answer, reference_answer)
        scores = {}
        total_correct = total_words = 0
        for direction in ["across", "down"]:
            pairs = pairs_by_direction[direction]
            correct_words = sum(1 for ref_word, model_word in pairs if ref_word == model_word)
            total = len(pairs)
            scores[direction] = correct_words / total if total else 0
            total_correct += correct_words
            total_words += total
        scores["all"] = total_correct / total_words if total_words else 0
        return scores
    
class LetterCoverageRate(CoverageMetric):
    """Calculate percentage of correctly placed letters."""
    @property
    def name(self) -> str:
        return "Letter Coverage Rate"

    def score(
        self,
        model_answer,
        reference_answer,
        puzzle_state,
        interactive_correctness=None,
        **kwargs
    ) -> Dict[str, float]:
        if not self._validate_inputs(model_answer, reference_answer):
            return {"across": 0, "down": 0, "all": 0}
        
        pairs_by_direction = self._get_answer_pairs_by_direction(model_answer, reference_answer)
        scores = {}
        total_correct_all = total_positions_all = 0
        for direction in ["across", "down"]:
            total_positions = correct_positions = 0
            pairs = pairs_by_direction[direction]
            for ref_word, model_word in pairs:
                min_length = min(len(ref_word), len(model_word))
                word_total = max(len(ref_word), len(model_word))
                total_positions += word_total
                correct_positions += sum(
                    1 for i in range(min_length)
                    if ref_word[i] == model_word[i]
                )
                
            scores[direction] = correct_positions / total_positions if total_positions else 0
            total_correct_all += correct_positions
            total_positions_all += total_positions
            
        scores["all"] = total_correct_all / total_positions_all if total_positions_all else 0
        return scores

class CompletionRate(CoverageMetric):
    """Calculate whether puzzle is fully completed correctly."""
    @property
    def name(self) -> str:
        return "Completion Rate"

    def score(
        self, 
        model_answer, 
        reference_answer, 
        puzzle_state, 
        interactive_correctness=None,
        **kwargs
    ) -> float:
        if not self._validate_inputs(model_answer, reference_answer):
            return 0
        ref_map = {
            str(item['direction']).lower(): normalize_string(item['answer'])
            for item in reference_answer
        }
        if len(model_answer) != len(ref_map):
            return 0

        try:
            for pos, answer_dict in model_answer.items():
                if pos not in ref_map or normalize_string(answer_dict['answer']) != ref_map[pos]:
                    return 0
            return 1
        except KeyError:
            return 0

class CrosswordGridAnalyzer:
    @staticmethod
    def get_direction(
        reference_answer: List[Dict[str, Any]],
        word: str
    ):
        for entry in reference_answer:
            if entry["answer"] == word:
                return entry["direction"]
        return None
    @staticmethod
    def find_intersections(
        wordlist: List[List[Any]],
        grid: List[List[str]],
        reference_answer: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Find all intersections between across and down words using wordlist."""
        # Separate words into across vs. down
        across_words = []
        down_words = []
        for entry in wordlist:
            word, clue, start_row, start_col, orientation = entry
            if orientation == 0:  # across
                across_words.append({
                    "word": word,
                    "clue": clue,
                    "row": start_row,
                    "col": start_col,
                })
            else:  # down
                down_words.append({
                    "word": word,
                    "clue": clue,
                    "row": start_row,
                    "col": start_col,
                })

        intersections = []

        # for each across word, compute its row, start_col, end_col
        for across in across_words:
            a_word = across["word"]
            a_row = across["row"]
            a_col_start = across["col"]
            a_col_end = a_col_start + len(a_word) - 1
            # compare against each down word
            for down in down_words:
                d_word = down["word"]
                d_row_start = down["row"]
                d_col = down["col"]
                d_row_end = d_row_start + len(d_word) - 1
                """
                Words intersect if the across row is within the down word's vertical span
                AND the down col is within the across word's horizontal span.
                
                Intersection row = a_row
                Intersection col = d_col
                
                across_index = intersection_col - a_col_start
                down_index   = intersection_row - d_row_start
                """
                if (d_col >= a_col_start and d_col <= a_col_end and
                    a_row >= d_row_start and a_row <= d_row_end):
                    
                    # Indices of the intersecting letters in each word
                    across_index = d_col - a_col_start
                    down_index = a_row - d_row_start

                    # The letter on the actual grid at that intersection
                    grid_letter = grid[a_row][d_col]

                    intersections.append({
                        "across_word": a_word,
                        "down_word": d_word,
                        "across_index": across_index,
                        "down_index": down_index,
                        "row": a_row,
                        "col": d_col,
                        "grid_letter": grid_letter,
                        "across_key": CrosswordGridAnalyzer.get_direction(reference_answer, a_word),
                        "down_key": CrosswordGridAnalyzer.get_direction(reference_answer, d_word),
                    })

        return intersections

class IntersectionRate(Metric):
    """Calculate satisfaction of overlapping constraints at grid intersections."""

    @property
    def name(self) -> str:
        return "Intersection Rate"

    def score(
        self,
        model_answer,
        reference_answer,
        puzzle_state,
        interactive_correctness=None,
        **kwargs
    ) -> float:
        """Calculate satisfaction of overlapping constraints at grid intersections."""
        if model_answer is None:
            return 0.0

        intersections = CrosswordGridAnalyzer.find_intersections(
            puzzle_state['wordlist'],
            puzzle_state['grid'],
            reference_answer
        )
        if not intersections:
            return 0.0

        valid_count = 0

        for inter in intersections:
            across_key = inter["across_key"]  # e.g. "across 4"
            down_key = inter["down_key"]      # e.g. "down 7"
            across_idx = inter["across_index"]
            down_idx = inter["down_index"]
            across_info = model_answer.get(across_key, {})
            down_info = model_answer.get(down_key, {})
            if not across_info or not down_info:
                continue
            across_model_answer = across_info.get("answer", "")
            down_model_answer = down_info.get("answer", "")
            # either answer is missing or too short, skip
            if (across_idx >= len(across_model_answer) or down_idx >= len(down_model_answer)):
                continue
            # compare the letters at the intersection
            if across_model_answer[across_idx] == down_model_answer[down_idx]:
                valid_count += 1
        return valid_count / len(intersections)


class InteractiveSuccessStep(Metric):
    @property
    def name(self) -> str:
        return "Interactive Success Step"

    def score(
        self, 
        model_answer, 
        reference_answer, 
        puzzle_state, 
        interactive_correctness=None,
        **kwargs
    ) -> float:
        return sum(interactive_correctness) if interactive_correctness else 0

class TokenUsage(Metric):
    @property
    def name(self) -> str:
        return "Effective Token Usage"
    
    @property
    def is_count_metric(self) -> bool:
        """Mark as a count metric to handle special aggregation"""
        return True

    def score(
        self, 
        model_answer, 
        reference_answer, 
        puzzle_state, 
        interactive_correctness=None,
        **kwargs
    ) -> Dict[str, float]:
        is_valid = True
        if model_answer is None:
            is_valid = False
        elif isinstance(model_answer, (dict, list, str)) and not model_answer:
            is_valid = False
        elif isinstance(model_answer, dict) and model_answer.get("error") == "default answer after max retries":
            is_valid = False
            
        token_count = 0
        meta_data = kwargs.get('meta_data', None)
        if meta_data and 'usage' in meta_data:
            token_count = int(meta_data['usage']['completion_tokens'])
        
        return {
            "tokens": token_count,
            "valid_response": 1 if is_valid else 0,
            "effective_tokens": token_count if is_valid else 0
        }

# We ask judge to determine if the model's response cannot be parsed to a valid anwer
# or the model concludes that it cannot provide a valid answer
# Try not adding this during debugging (You can remove it from tasks/crossword.py)
# Only add this during final testing
# class Judge(Metric):
#     @property
#     def name(self) -> str:
#         return "Judge"

#     def score(
#         self,
#         model_answer,
#         reference_answer,
#         puzzle_state,
#         interactive_correctness=None,
#         **kwargs
#     ) -> float:
#         meta_data = kwargs.get('meta_data', None)
#         if meta_data:
#             raw_model_response = meta_data['choice'][-1]['message']['content']
#             prompt = Judge_Prompt.replace('<response>', raw_model_response)
#             messages = [{
#                 'role': 'user',
#                 'content': prompt
#             }]
#             judgement = litellm.completion(
#                 model = JUDGE_MODEL,
#                 messages = messages,
#                 num_retries = MAX_RETRIES,
#                 max_completion_tokens = MAX_TOKENS,
#                 temperature = 0,
#                 top_p = 1.0,
#             )
#             messages = Judge_Parsing_Prompt.replace('<response>', judgement)
#             messages = [{
#                 'role': 'user',
#                 'content': messages
#             }]
#             for _ in range(MAX_RETRIES):
#                 try:
#                     parsed = litellm.completion(
#                         model = JUDGE_PARSING_MODEL,
#                         messages = messages,
#                         num_retries = MAX_RETRIES,
#                         max_completion_tokens = MAX_TOKENS,
#                         temperature = 0,
#                         top_p = 1.0,
#                     )
#                     if parsed is None:
#                         continue
#                     parsed = json.loads(parsed.choices[0].message.content)
#                     if parsed:
#                         return parsed
#         return {}


############################################################################################################################################################################################################################################################
# error metric will not be computed for mean, it will sum up the error count
class ErrorMetric(Metric):
    @property
    def is_count_metric(self) -> bool:
        return True

class LocalLengthError(ErrorMetric):
    @property
    def name(self) -> str:
        return "Local Length Error"

    def score(
        self, 
        model_answer, 
        reference_answer, 
        puzzle_state, 
        interactive_correctness=None,
        **kwargs
    ) -> float:
        if model_answer is None:
            return {
                'local_length_error': len(reference_answer),
                'longer_than_reference': 0,
                'shorter_than_reference': 0
            }
            
        length_errors = 0
        longer_count = 0
        shorter_count = 0
        
        for ref in reference_answer:
            model_len = len(model_answer.get(ref['direction'].lower(), {}).get('answer', ''))
            ref_len = len(ref['answer'])
            
            if model_len != ref_len:
                length_errors += 1
                if model_len > ref_len:
                    longer_count += 1
                elif model_len < ref_len:
                    shorter_count += 1
                    
        return {
            'local_length_error': length_errors,
            'longer_than_reference': longer_count,
            'shorter_than_reference': shorter_count
        }

class GlobalLengthError(ErrorMetric):
    @property
    def name(self) -> str:
        return "Global Length Error"

    def score(
        self, 
        model_answer, 
        reference_answer, 
        puzzle_state, 
        interactive_correctness=None,
        **kwargs
    ) -> float:
        if model_answer is None:
            return {
                'global_length_error': len(reference_answer),
                'longer_than_reference': 0,
                'shorter_than_reference': 0
            }
        lenth_diff = len(model_answer) - len(reference_answer)
        longer_count = 1 if lenth_diff > 0 else 0
        shorter_count = 1 if lenth_diff < 0 else 0
        return {
            'global_length_error': 1 if lenth_diff != 0 else 0,
            'longer_than_reference': longer_count,
            'shorter_than_reference': shorter_count
        }

class EmptyAnswerError(ErrorMetric):
    @property
    def name(self) -> str:
        return "Empty Answer Error"

    def score(
        self, 
        model_answer, 
        reference_answer, 
        puzzle_state, 
        interactive_correctness=None,
        **kwargs
    ) -> float:
        if model_answer is None:
            return 1
        if isinstance(model_answer, (dict, list, str)) and not model_answer:
            return 1
        if isinstance(model_answer, dict) and model_answer.get("error") == "default answer after max retries":
            return 1
        return 0
        
class PredictedLengthError(ErrorMetric):
    """Count cases where model's predicted length differs from reference answer length."""
    @property
    def name(self) -> str:
        return "Predicted Length Error"

    def score(
        self, 
        model_answer, 
        reference_answer, 
        puzzle_state, 
        interactive_correctness=None,
        **kwargs
    ) -> float:
        if model_answer is None:
            return {
                'predicted_length_error': len(reference_answer),
                'longer_than_reference': 0,
                'shorter_than_reference': 0,
                'missing_predicted_length': len(reference_answer)
            }
        longer_count, shorter_count, total_errors = 0, 0, 0
        missing_predicted_length = 0
        for ref in reference_answer:
            direction = ref['direction'].lower()
            if direction not in model_answer:
                total_errors += 1
                missing_predicted_length += 1
                continue
            predicted_length = int(model_answer.get(direction, {}).get('length', 0))
            actual_length = len(ref['answer'])
            if predicted_length != actual_length:
                total_errors += 1
                if predicted_length > actual_length:
                    longer_count += 1
                else:
                    shorter_count += 1
        return {
            'predicted_length_error': total_errors,
            'longer_than_reference': longer_count,
            'shorter_than_reference': shorter_count,
            'missing_predicted_length': missing_predicted_length
        }
   
class AnswerLengthConsistencyError(ErrorMetric):
    """Count cases where model's answer length doesn't match its predicted length."""
    @property
    def name(self) -> str:
        return "Answer Length Consistency Error"

    def score(
        self, 
        model_answer, 
        reference_answer, 
        puzzle_state, 
        interactive_correctness=None,
        **kwargs
    ) -> float:
        if model_answer is None:
            return {
                'answer_length_consistency_error': len(model_answer),
                'longer_than_predicted': 0,
                'shorter_than_predicted': 0,
                'missing_predicted_length': len(model_answer)
            }

        longer_count, shorter_count, total_errors, missing_predicted_length = 0, 0, 0, 0
        for direction, ans in model_answer.items():
            answer_length = len(str(ans.get('answer', '')))
            if 'length' not in ans:
                total_errors += 1
                missing_predicted_length += 1
                continue
            predicted_length = int(ans.get('length', 0))
            if answer_length != predicted_length:
                total_errors += 1
                if answer_length > predicted_length:
                    longer_count += 1
                else:
                    shorter_count += 1
        return {
            'answer_length_consistency_error': total_errors,
            'longer_than_predicted': longer_count,
            'shorter_than_predicted': shorter_count,
            'missing_predicted_length': missing_predicted_length
        }

class SelfConsistencyLengthError(ErrorMetric):
    """Count cases where model's answer matches its predicted length but both are wrong."""
    @property
    def name(self) -> str:
        return "Self Consistency Length Error"

    def score(
        self, 
        model_answer, 
        reference_answer, 
        puzzle_state, 
        interactive_correctness=None,
        **kwargs
    ) -> float:
        if model_answer is None:
            return {
                'self_consistency_length_error': len(reference_answer),
                'longer_than_reference': 0,
                'shorter_than_reference': 0,
                'missing_predicted_length': len(reference_answer)
            }
        longer_count, shorter_count, total_errors, missing_predicted_length = 0, 0, 0, 0
        for ref in reference_answer:
            direction = ref['direction'].lower()
            if direction not in model_answer:
                total_errors += 1
                missing_predicted_length += 1
                continue
            if direction in model_answer:
                model_ans = model_answer[direction]
                model_length = len(model_ans.get('answer', ''))
                if 'length' not in model_ans:
                    total_errors += 1
                    missing_predicted_length += 1
                    continue
                predicted_length = int(model_ans.get('length', 0))
                reference_length = len(ref['answer'])
                # check if model answer length equals its predicted length but both are wrong
                if (model_length == predicted_length) and (model_length != reference_length):
                    total_errors += 1
                    if model_length > reference_length:
                        longer_count += 1
                    elif model_length < reference_length:
                        shorter_count += 1
        return {
            'self_consistency_error': total_errors,
            'longer_than_reference': longer_count,
            'shorter_than_reference': shorter_count,
            'missing_predicted_length': missing_predicted_length
        }
    

class WordCount(ErrorMetric):
    @property
    def name(self) -> str:
        return "Word Count"

    def score(
        self, 
        model_answer, 
        reference_answer, 
        puzzle_state, 
        interactive_correctness=None,
        **kwargs
    ) -> float:
        return len(reference_answer)

class CostUsage(ErrorMetric):
    @property
    def name(self) -> str:
        return "Cost Usage"

    def score(
        self, 
        model_answer, 
        reference_answer, 
        puzzle_state, 
        interactive_correctness=None,
        **kwargs
    ) -> float:
        cost = kwargs.get('meta_data', None)
        if cost:
            cost = cost.get('cost', 0.0)
            return float(cost)
        return 0.0


if __name__ == '__main__':
    import json

    from datasets import load_dataset
    data = load_dataset('HINT-lab/CrossWordBench', 'english', split='7x7')
    puzzle_state = json.loads(data[6]['puzzle_state'])
    print(puzzle_state['meta_data']['id'])
    answers = json.loads(data[6]['reference_answer'])
    metric = IntersectionRate()
    grid = puzzle_state['grid']
    intersections = CrosswordGridAnalyzer.find_intersections(puzzle_state['wordlist'], grid, answers)
    for intersec in intersections:
        print(intersec)

