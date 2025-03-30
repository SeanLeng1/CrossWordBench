import json
from pathlib import Path

from datasets import load_dataset

from eval.metrics import (  # PredictedLengthError,; AnswerLengthConsistencyError,; SelfConsistencyLengthError,; GridError,; Judge,
    CompletionRate, CostUsage, EmptyAnswerError, GlobalLengthError,
    InteractiveSuccessStep, IntersectionRate, LetterCoverageRate,
    LocalLengthError, Metric, TokenUsage, WordCount, WordCoverageRate)
from eval.task import BaseEval, Interaction


class PuzzleEval(BaseEval):
    """Evaluation system specifically for puzzle-solving tasks."""

    @property
    def metric_fns(self) -> list[Metric]:
        return [
            WordCoverageRate(), 
            LetterCoverageRate(), 
            CompletionRate(), 
            IntersectionRate(), 
            InteractiveSuccessStep(),
            LocalLengthError(),
            GlobalLengthError(),
            EmptyAnswerError(),
            WordCount(),
            TokenUsage(),
            CostUsage(),
            # GridError(),
            # PredictedLengthError(),
            # AnswerLengthConsistencyError(),
            # SelfConsistencyLengthError(),
            # Judge(),              # uncomment this for final evaluation
        ]

    def form_text_clues(self, subject, reference_answer, puzzle_state):
        clues = []
        for answer in reference_answer:
            position = ""
            gt_anwer = answer['answer']
            # wordlist: [word, clue, position y, position x, orientation]
            for word in puzzle_state['wordlist']:
                if word[0] == gt_anwer:
                    position = str(word[2]) + ', ' + str(word[3])
            if 'anagram' in subject:
                clue_text = ""
                # for text anagrams, we separate the letters for tokenization
                for word in answer['clue']:
                    clue_text += f"{word} "
                clue_text = clue_text.strip()
                direction = answer['direction']
                # upper case the first letter of the direction
                direction = direction[0].upper() + direction[1:]
                clues.append(f'{direction} ({position}): {clue_text}')
            else:
                direction = answer['direction']
                # upper case the first letter of the direction
                direction = direction[0].upper() + direction[1:]
                clues.append(f'{direction} ({position}): {answer["clue"]}')
        return clues

    def load_data(
        self, 
        dataset_name: str, 
        subject: str, 
        difficulty:str, 
        first_round_results: Path = None
    ):
        """Load puzzle data into interactions.
        features = datasets.Features({
            "grid_image": datasets.Image(),
            "empty_grid_image": datasets.Image(),
            "key_image": datasets.Image(),
            "partial_0.25": datasets.Image(),
            "partial_0.5": datasets.Image(),
            "partial_0.75": datasets.Image(),
            "id": datasets.Value("int32"),  
            "difficulty": datasets.Value("string"),
            "reference_answer": datasets.Value("string"),
            "puzzle_state": datasets.Value("string"),
        })
        """
        puzzles = load_dataset(dataset_name, subject)[difficulty]
        if 'anagram' in subject:
            self.logger.info("Automatically separating clue with space for tokenization.")
        if first_round_results and first_round_results != 'none':
            self.logger.info(f"Loading first-round results from {first_round_results}. Make sure they are consistent with the evaluated model.")
            first_round_results = Path(first_round_results)
        loaded_interactions = []
        for puzzle in puzzles:
            id = puzzle['id']
            response = None
            # load saved result if there is one
            file_name = f"{id}.json"
            if first_round_results and first_round_results != 'none':
                file_path = first_round_results / file_name
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        response = json.load(f)
                        response = response['metadata']
                else:
                    raise FileNotFoundError(f"You are using saved results but {file_name} is not found. Double check your arguments.")
            reference_answer = json.loads(puzzle['reference_answer'])
            clues = []
            filtered_answers = []
            for answer in reference_answer:
                clues = self.form_text_clues(
                    subject, 
                    reference_answer, 
                    json.loads(puzzle['puzzle_state'])
                )
                filtered_answers.append(answer)
            interaction = Interaction(
                puzzle_id=str(id),
                request = {
                    "grid_image": puzzle['grid_image'],
                    "grid_only_image": puzzle['empty_grid_image'],
                    "partial_images": [
                        puzzle['partial_0.25'],
                        puzzle['partial_0.5'],
                        puzzle['partial_0.75'],
                    ],
                    "partial_grids": [
                        json.loads(puzzle['partial_grid_0.25'])['grid'],
                        json.loads(puzzle['partial_grid_0.5'])['grid'],
                        json.loads(puzzle['partial_grid_0.75'])['grid'],
                    ],
                    "puzzle_state": json.loads(puzzle['puzzle_state']),
                    "clues": clues,
                    "key_image": puzzle['key_image'],
                    "id": str(id),
                    "previous_response": response,
                    "reference_answer": filtered_answers,
                },
                reference_answer = filtered_answers,
                puzzle_state = json.loads(puzzle['puzzle_state']),
                difficulty=puzzle['difficulty']
            )
            loaded_interactions.append(interaction)
        self.interactions = loaded_interactions


