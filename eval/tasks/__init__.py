from eval.tasks.crossword import PuzzleEval
from datasets import load_dataset

TASK_REGISTRY = {
    "crossword": PuzzleEval
}


def get_task(task_name):
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Did not recognize task name {task_name}")

    return TASK_REGISTRY[task_name]()