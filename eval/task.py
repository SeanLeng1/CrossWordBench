import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (Any, Dict, List, Optional, Sequence, Tuple,
                    TypeVar, Union)

import numpy as np
from scipy import stats
from tabulate import tabulate
from tqdm import tqdm

from eval.metrics import Metric
from eval.model import Model
from utils import setup_logger

MAX_RETRIES = 5
T = TypeVar('T')
MetricResult = Dict[str, float]

@dataclass
class EvaluationMetaData:
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_info: Dict[str, Any] = field(default_factory=dict)
    evaluation_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Interaction:
    puzzle_id: str
    request: Dict[str, Any]
    reference_answer: Union[Dict[str, str], List[Dict[str, str]]]
    # we need puzzle state for interactive template
    puzzle_state: Optional[Dict[str, Any]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    model_answer: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None
    interactive_correctness: Optional[List[bool]] = None
    difficulty: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "puzzle_id": self.puzzle_id,
            "reference_answer": self.reference_answer,
            "model_answer": self.model_answer,
            "puzzle_state": self.puzzle_state,
            "metrics": self.metrics,
            "interactive_correctness": self.interactive_correctness,
            "metadata": self.metadata
        }
    
# TODO: make this more advanced, add more stats
class MetricAggregator:
    def __init__(self, metrics: Sequence[Metric]):
        self.metric_fns = metrics
        self.metric_names = [metric.name for metric in metrics]
        self.reset()
    
    def reset(self) -> None:
        self.values = {metric_name: [] for metric_name in self.metric_names}
        self.raw_values = {metric_name: [] for metric_name in self.metric_names}

    def add_interaction(self, interaction: Interaction) -> None:
        """
        Add metrics from an interaction to the aggregator.
        Properly tracks length errors and empty answers.
        """
        if not hasattr(self, '_processed_interactions'):
            self._processed_interactions = set()
        
        if interaction.puzzle_id not in self._processed_interactions:
            self._processed_interactions.add(interaction.puzzle_id)
        
        for metric in self.metric_fns:
            name = metric.name
            if name not in interaction.metrics:
                continue
            value = interaction.metrics[name]
            # sometimes it can be a dict
            if isinstance(value, dict):
                for key, val in value.items():
                    if f"{name}_{key}" not in self.values:
                        self.values[f"{name}_{key}"] = []
                        self.raw_values[f"{name}_{key}"] = []
                    self.values[f"{name}_{key}"].append(val)
                    self.raw_values[f"{name}_{key}"].append(val)
            else:
                self.values[name].append(value)
                self.raw_values[name].append(value)
    
    def compute_confidence_interval(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
        """95% confidence interval for the mean."""
        if not values:
            return 0.0, 0.0, 0.0
        
        arr = np.array(values)
        mean = float(np.mean(arr))
        
        if len(values) < 2:
            return mean, mean, mean
        
        std = np.std(arr, ddof=1)
        if std == 0:
            return mean, mean, mean
        # confidence interval
        sem = stats.sem(arr)
        t_value = stats.t.ppf((1 + confidence) / 2., len(values) - 1)
        margin = t_value * sem
        ci_lower = mean - margin
        ci_upper = mean + margin

        return mean, float(ci_lower), float(ci_upper)
    
    def compute_aggregates(self) -> Dict[str, float]:
        results = {}
        all_metric_keys = list(self.values.keys())
        
        for metric in self.metric_fns:
            metric_name = metric.name
            related_keys = [k for k in all_metric_keys if k.startswith(metric_name)]
            
            # Skip counting metrics
            if hasattr(metric, 'is_count_metric') and metric.is_count_metric:
                for key in related_keys:
                    values = self.values[key]
                    results[f"{key}_total"] = sum(values)
                    results[f"{key}_count"] = len(values)
                continue

            # Check if this metric returns dict values
            is_dict_metric = len(related_keys) > 1 and metric_name in related_keys
            if is_dict_metric:
                related_keys.remove(metric_name)
                
            for key in related_keys:
                values = self.values[key]
                if not values:
                    continue

                arr = np.array(values)
                mean, ci_lower, ci_upper = self.compute_confidence_interval(values)
                margin = (ci_upper - ci_lower) / 2
                try:
                    metric_stats = {
                        'mean': mean,
                        'std': float(np.std(arr, ddof=1)),
                        'std_err': float(stats.sem(arr)),
                        'margin': margin,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'median': float(np.median(arr)),
                        'min': float(np.min(arr)),
                        'max': float(np.max(arr)),
                        'total': float(np.sum(arr)),
                        'count': len(values)
                    }
                except Exception:
                    metric_stats = {
                        'mean': mean, 'std': 0.0, 'std_err': 0.0, 'margin': margin,
                        'ci_lower': ci_lower, 'ci_upper': ci_upper,
                        'median': mean, 'min': mean, 'max': mean,
                        'total': mean * len(values), 'count': len(values)
                    }
                
                for stat_name, stat_value in metric_stats.items():
                    results[f"{key}_{stat_name}"] = stat_value
                    
        return results

    def get_summary_string(self) -> str:
        if not self.metric_names:
            return "No metrics defined"
        stats = self.compute_aggregates()
        all_metric_keys = list(self.values.keys())
        stats_tables = []
        count_tables = []
        for metric in self.metric_fns:
            metric_name = metric.name
            related_keys = [k for k in all_metric_keys if k.startswith(metric_name)]
            if hasattr(metric, 'is_count_metric') and metric.is_count_metric:
                count_data = []
                for key in related_keys:
                    total = sum(self.values[key])
                    count = len(self.values[key])
                    if count > 0:
                        suffix = key[len(metric_name):].lstrip('_') if key != metric_name else ''
                        display_name = f"{metric_name} {suffix}".strip()
                        count_data.append([display_name, total, count])
                if count_data:
                    table = tabulate(
                        count_data,
                        headers=['Metric', 'Total', 'Puzzles'],
                        tablefmt='fancy_grid',
                        floatfmt='.3f'
                    )
                    count_tables.append(table)
                continue
            stats_data = []
            for key in related_keys:
                try:
                    count = stats[f"{key}_count"]
                    if count == 0:
                        continue
                    suffix = key[len(metric_name):].lstrip('_') if key != metric_name else ''
                    display_name = f"{metric_name} {suffix}".strip()
                    stats_data.append([
                        display_name,
                        f"{stats[f'{key}_mean']:.3f} ± {stats[f'{key}_std_err']:.3f}",
                        f"{stats[f'{key}_mean']:.3f} ± {stats[f'{key}_margin']:.3f}",
                        f"[{stats[f'{key}_ci_lower']:.3f}, {stats[f'{key}_ci_upper']:.3f}]",
                        stats[f'{key}_median'],
                        stats[f'{key}_min'],
                        stats[f'{key}_max'],
                        count
                    ])
                except (KeyError, ValueError):
                    continue
            if stats_data:
                table = tabulate(
                    stats_data,
                    headers=['Metric', 'Mean ± std_err', 'Mean ± margin', '95% CI', 'Median', 'Min', 'Max', 'N'],
                    tablefmt='fancy_grid',
                    floatfmt='.3f'
                )
                stats_tables.append(table)
        parts = []
        if stats_tables:
            parts.extend([
                "=" * 80,
                "STATISTICS SUMMARY",
                "=" * 80,
                *stats_tables
            ])
        if count_tables:
            parts.extend([
                "\n" + "=" * 80,
                "ERROR COUNTS",
                "=" * 80,
                *count_tables
            ])
        return "\n\n".join(parts)
            
        

class BaseEval(ABC):
    def __init__(self):
        self.interactions: List[Interaction] = []
        self.meta_data = EvaluationMetaData()
        self.logger = setup_logger(verbose=False)

    @property
    @abstractmethod
    def metric_fns(self) -> list[Metric]:
        """A list of metrics to compute for request-response pairs."""
        pass

    @abstractmethod
    def load_data(self, data: List[Dict[str, Any]], output_dir: Union[str, Path]) -> None:
        """Load evaluation data into interactions."""
        pass

    def _process_model_response(
        self,
        response,
        interaction: Interaction,
    ):
        if response is None:
            return interaction
        interaction.model_answer = response.pop('json_response', None)
        interaction.interactive_correctness = response.pop('interactive_correctness', None)
        interaction.metadata = response
        return interaction
    
    def _save_interaction(
        self,
        interaction: Interaction,
        output_dir: Path
    ):
        # if interaction.model_answer is not None:
        save_path = output_dir / f"{interaction.puzzle_id}.json"
        with save_path.open('w') as f:
            json.dump(interaction.to_dict(), f, indent=2)

    def _check_existing_interaction(
        self,
        output_dir: Path,
        interaction_map: Dict[str, Interaction]
    ):
        processed_ids = set()
        # load existing responses from output directory
        for file in output_dir.glob("[!metric]*.json"):
            try:
                data = json.loads(file.read_text(encoding='utf-8'))
                puzzle_id = data.get('puzzle_id')
                if puzzle_id and puzzle_id in interaction_map:
                    interaction = interaction_map[puzzle_id]
                    if data.get('model_answer'):
                        interaction.model_answer = data['model_answer']
                        processed_ids.add(puzzle_id)
                    # this gets accumulated for two_round template, not loading for now
                    if data.get('metadata'):
                        interaction.metadata = data['metadata']
                        if 'choices' in interaction.metadata:
                            choices = interaction.metadata['choices']
                            # if we have multiple choices, we should only read the first one
                            # since it can be accumulated in the second round
                            if len(choices) > 1:
                                interaction.metadata['choices'] = [choices[0]]
                    if data.get('metrics'):
                        interaction.metrics = data['metrics']
                    if data.get('puzzle_state'):
                        interaction.puzzle_state = data['puzzle_state']
                    if data.get('reference_answer'):
                        interaction.reference_answer = data['reference_answer']
                    if data.get('interactive_correctness'):
                        interaction.interactive_correctness = data['interactive_correctness']
            except Exception as e:
                self.logger.error(f"Error loading {file}: {e}")
                self.logger.error("Full error:", exc_info=True)   
                continue
        pending_interactions = [
            interaction for interaction in self.interactions 
            if interaction.puzzle_id not in processed_ids
        ]
        return pending_interactions

    def get_responses(
        self,
        output_dir: Union[str, Path],
        model: Model,
        parallel: int = 16,
        **kwargs
    ):
        """Get model responses for all interactions in parallel."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # create a mapping of puzzle_id to interaction for easy updating
        interaction_map = {interaction.puzzle_id: interaction for interaction in self.interactions}
        retry_count = 0
        while retry_count < MAX_RETRIES:
            # identify which interactions need processing
            pending_interactions = self._check_existing_interaction(output_dir, interaction_map)
            if not pending_interactions:
                self.logger.warning("No new responses needed - all puzzles have answers")
                break
            self.logger.info(f"Attempt {retry_count + 1}/{MAX_RETRIES}: Processing {len(pending_interactions)} interactions")
            if len(pending_interactions) < 10:
                self.logger.info(f"Missing answers for: {[i.puzzle_id for i in pending_interactions]}")
            futures: Dict[Future, Interaction] = {}
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                for interaction in pending_interactions:
                    request = interaction.request.copy()
                    futures[executor.submit(
                        model,
                        **request,
                        parse_json=True,
                        **kwargs
                    )] = interaction
                for future in tqdm(
                    as_completed(futures),
                    total=len(pending_interactions),
                    desc="Getting model responses"
                ):
                    #time.sleep(5)
                    interaction = futures[future]
                    try:
                        response = future.result()
                        interaction = self._process_model_response(response, interaction)
                        self._save_interaction(interaction, output_dir)
                        interaction_map[interaction.puzzle_id] = interaction
                    except Exception as e:
                        self.logger.error(f"Error processing {interaction.puzzle_id}: {e}")
                        self.logger.error("Full error:", exc_info=True)   
                        continue
            retry_count += 1

        # update self.interactions with all interactions, including both
        # successfully processed ones and those that weren't processed
        self.interactions = list(interaction_map.values())
        # final check
        final_pending = [
            i for i in self.interactions
            if i.model_answer is None or not i.model_answer
        ]
        if final_pending:
            if retry_count >= MAX_RETRIES:
                self.logger.error(
                    f"Maximum retries ({MAX_RETRIES}) reached. Still missing answers for: "
                    f"{[i.puzzle_id for i in final_pending]}"
                )
                self.logger.error("Saving default answers for missing puzzles")
                for interaction in final_pending:
                    interaction.model_answer = {"error": "default answer after max retries"}
                    self._save_interaction(interaction, output_dir)
            else:
                self.logger.warning(
                    f"Finished with missing answers for: "
                    f"{[i.puzzle_id for i in final_pending]}"
                )
        else:
            self.logger.info("All puzzles answered successfully!")

    def compute_metrics(self):
        aggregator = MetricAggregator(self.metric_fns)
        for interaction in tqdm(self.interactions, desc="Computing metrics", total=len(self.interactions)):
            for metric in self.metric_fns:
                interaction.metrics[metric.name] = metric.score(
                    interaction.model_answer,
                    interaction.reference_answer,
                    interaction.puzzle_state,
                    interaction.interactive_correctness,
                    meta_data = interaction.metadata,
                )
            aggregator.add_interaction(interaction)
        return aggregator.compute_aggregates(), aggregator.get_summary_string(), aggregator.raw_values


    
