import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
import copy
import json
import time
from argparse import ArgumentParser
from pathlib import Path

from transformers import set_seed

from eval.model import EvalModel, ModelConfig
from eval.tasks import get_task
from utils import setup_logger


def parse_args():
    parser = ArgumentParser()
    # seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    # evaluation config
    parser.add_argument(
        "--eval_name",
        type=str,
        default='crossword',
        help="Name of the evaluation task to run (currently only supports crossword)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to evaluate",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name or path of the model to evaluate (e.g., 'gpt-4o')",
    )
    parser.add_argument(
        "--parsing_model",
        type=str,
        default='gpt-4-turbo',
        help="Model to use for parsing responses. Default: gpt-4-turbo",
    )
    parser.add_argument(
        "--template_type",
        type=str,
        default='direct',
        help="Type of prompt template to use for generation (e.g., 'image-cot', 'image-shot')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory path where evaluation results will be saved",
    )
    parser.add_argument(
        "--anagram_mix",
        action='store_true',
        help="Indicate if the puzzle is anagram mix",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default='english',
        help="Subject of the dataset (e.g., 'english', 'chinese')",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default='7x7',
        help="Difficulty level or grid size of puzzles (e.g., '7x7', '14x14', '21x21')",
    )
    parser.add_argument(
        "--first_round_results",
        type=str,
        default=None,
        help="Path to first-round results for reflection template",
    )
    # generation config
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation",
    )
    parser.add_argument(
        "--top_p",
        type=int,
        default=1.0,
        help="Nucleus sampling probability threshold",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default='medium',
        help="Reasoning effort level for generation (e.g., 'low', 'medium', 'high')",
    )
    parser.add_argument(
        "--budget_tokens",
        type=int,
        default=0,
        help="budget_token for claude thinking model"
    )
    # parser.add_argument(
    #     "--repetition_penalty",
    #     type=int,
    #     default=1.1,               
    #     help="Repetition penalty for generation",
    # )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens to generate per completion (adhere to o1)",
    )
    # vllm config
    parser.add_argument(
        "--use_vllm",
        action='store_true',
        help="Indicate if the model is hosted using vllm",
    )
    parser.add_argument(
        "--lora_module",
        type=str,
        default=None,
        help="Name of the LoRA module to use for vLLM API access",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=None,
        help="Base URL for vLLM API endpoint (e.g., 'http://localhost:8000')",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Authentication key for vLLM API access",
    )
    # general config
    parser.add_argument(
        "--parallel",
        type=int,
        default=16,
        help="Number of parallel API requests to make",
    )
    parser.add_argument(
        "--mm_limit",
        type=int,
        default=4,
        help="Limit of images to send to the model",
    )
    return parser.parse_args()


def evaluate(args):
    logger = setup_logger(verbose = False)
    args.api_base = None if isinstance(args.api_base, str) and args.api_base.lower() == 'none' else args.api_base
    args.api_key = None if isinstance(args.api_key, str) and args.api_key.lower() == 'none' else args.api_key
    logger.info("\U0001F61A Starting evaluation with arguments: %s", vars(args))

    if 'Aria' in args.model and args.template_type == 'interactive':
        logger.error("Aria currently has some issues with interactive template. Please use other template types.")
        exit(1)
    if ('deepseek' in args.model or 'llava' in args.model) and (args.template_type == 'img_shot' or args.template_type == 'interactive'):
        logger.error("Deepseek / llava currently has a very short context window that is not enough for image-shot template. Please use other template types.")
        exit(1)
    if ('Molmo' in args.model) and (args.template_type == 'img_shot' or args.template_type == 'interactive'):
        logger.error("Molmo only supports single image. Refer to https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-multimodal-language-models for more info.")
        exit(1)

    clean_model_name = args.model.split('/')[-1]
    args.output_dir = Path(args.output_dir) / clean_model_name / args.subject / args.difficulty / args.template_type

    if 'two_round' in args.template_type:
        assert args.first_round_results and Path(args.first_round_results).exists(), "You must provide saved results for two-round evaluation (to ensure direct comparison)"
    logger.info("Saving results to %s", args.output_dir)
    
    model_config = ModelConfig(
        model_name=args.model,
        use_vllm=args.use_vllm,
        lora_module=args.lora_module,
        api_base=args.api_base,
        api_key=args.api_key,
        template_type=args.template_type,
        parsing_model_name=args.parsing_model,
        output_dir=args.output_dir,
    )

    if args.anagram_mix:
        logger.warning("Anagram Prompt is enabled, you must make sure that the puzzle is anagram mix")

    model = EvalModel(model_config, args.anagram_mix)
    eval_task = get_task(args.eval_name)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_task.load_data(
        args.dataset_name, 
        args.subject, 
        args.difficulty, 
        args.first_round_results, 
    )

    extra_args = {}
    # this is only used by new claude models
    if "claude-3-7-sonnet-20250219" in args.model:
        if args.budget_tokens > 0:
            logger.warning("Claude thinking model is enabled with budget_tokens: %d", args.budget_tokens)
            # "type":"invalid_request_error","message":"`temperature` may only be set to 1 when thinking is enabled. Please consult our documentation at https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#important-considerations-when-using-extended-thinking"
            logger.warning("Temperature is fixed to 1.0 and top_p will be unset due to claude 3.7 constraints")
            time.sleep(3)
            extra_args["thinking"] = {"type": "enabled", "budget_tokens": args.budget_tokens}
            args.temperature = 1.0
            args.top_p = None
        else:
            logger.info("Seems like you are not using thinking mode of claude 3.7, will default max_tokens to 8192")
            time.sleep(3)
            args.max_completion_tokens = 8192
            # we can increase a little bit
            args.parallel = 8
    
    if args.lora_module is not None:
        logger.info("LoRA module is enabled with module: %s", args.lora_module)
        time.sleep(3)
    
    eval_task.get_responses(
        output_dir, 
        model, 
        temperature=args.temperature, 
        top_p=args.top_p,
        # repetition_penalty=args.repetition_penalty,
        # not all models support seed, but it can help with reproducibility sometimes
        seed=args.seed,             # https://docs.litellm.ai/docs/completion/input
        max_completion_tokens=args.max_completion_tokens,
        parallel=args.parallel,
        mm_limit=args.mm_limit,
        reasoning_effort=args.reasoning_effort,
        **extra_args,
    )

    # for template other than interactive, we skip success rate for clarity
    aggregate_metrics, aggregate_metrics_str, raw_values = eval_task.compute_metrics()
    (output_dir / 'metrics').mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'metrics' / 'metrics.json', 'w') as f:
        save_metrics = copy.deepcopy(aggregate_metrics)
        save_metrics['cost'] = sum([interaction.metadata.get('cost', 0) or 0 for interaction in eval_task.interactions])
        save_metrics['usage'] = {
            'prompt_tokens': sum([interaction.metadata['usage']['prompt_tokens'] for interaction in eval_task.interactions]),
            'completion_tokens': sum([interaction.metadata['usage']['completion_tokens'] for interaction in eval_task.interactions]),
            'total_tokens': sum([interaction.metadata['usage']['total_tokens'] for interaction in eval_task.interactions]),
        }
        save_metrics['model'] = eval_task.interactions[0].metadata['model']
        save_metrics['parsing_model'] = eval_task.interactions[0].metadata['parsing_model']
        save_metrics['temperature'] = args.temperature
        save_metrics['top_p'] = args.top_p
        json.dump(save_metrics, f, indent=2)

    with open(output_dir / 'metrics' / 'raw_metrics.json', 'w') as f:
        json.dump(raw_values, f, indent=2)

    logger.info('The cost estimation is not guaranteed to be accurate. Please keep tracking of your usage!! \U0001F643')
    logger.info("\U0001F44D Evaluation completed successfully! \U0001F44D")
    logger.info("Evaluation results:\n%s", aggregate_metrics_str)

    with open(output_dir / 'metrics' / 'metrics.txt', 'w') as f:
        f.write(aggregate_metrics_str)

    logger.info("All evaluation metrics are computed, but be aware that different metrics may not be applicable to all tasks.")


# in case we create cli for this
def main():
    args = parse_args()
    set_seed(args.seed)
    evaluate(args)

if __name__ == '__main__':
    main()
    
