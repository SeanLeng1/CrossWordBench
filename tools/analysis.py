import json
import os

from sklearn.metrics import cohen_kappa_score
from tabulate import tabulate
from tqdm import tqdm


def normalize_string(s: str | None) -> str:
    if s is None:
        return ""
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1].lower()
    return s.lower()

def process_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: {e} in {json_path}")
        return {}
    word_list = data.get('puzzle_state', {}).get('wordlist', [])
    ref_answers = data.get('reference_answer', [])
    model_answers = data.get('model_answer', [])
    ref_index = {}
    for ref in ref_answers:
        clue = ref.get('clue', '')
        if clue:
            ref_index[clue] = ref
    results = {}
    for entry in word_list:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        word, clue = entry[0], entry[1]
        ref = ref_index.get(clue)
        if not ref:
            continue
        direction = ref.get('direction')
        ref_answer = ref.get('answer')
        model_info = model_answers.get(direction, {})
        model_answer = model_info.get("answer")
        if model_answer is None or ref_answer is None:
            continue
        is_correct = normalize_string(model_answer) == normalize_string(ref_answer)
        if clue not in results:
            results[clue] = {"correct": 0, "total": 0}
        results[clue]["total"] += 1
        if is_correct:
            results[clue]["correct"] += 1
    return results

def process_folder(folder_path):
    clues_results = {}
    for file in os.listdir(folder_path):
        if not file.endswith(".json"):
            continue
        file_path = os.path.join(folder_path, file)
        json_result = process_json(file_path)
        for clue, stats in json_result.items():
            if clue not in clues_results:
                clues_results[clue] = {"correct": 0, "total": 0}
            clues_results[clue]["correct"] += stats["correct"]
            clues_results[clue]["total"] += stats["total"]
    # calculate accuracy for each clue
    aggregated_results = {}
    for clue, stats in clues_results.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        aggregated_results[clue] = {"accuracy": accuracy, "count": stats["total"]}
    return aggregated_results

def main(base_path):
    model_stats = {}
    for model_name in tqdm(os.listdir(base_path)):
        print(f"Processing {model_name}")
        model_dir = os.path.join(base_path, model_name)
        if not os.path.isdir(model_dir):
            continue
        seven_dir = os.path.join(model_dir, "english", "7x7", "text_cot")
        fourteen_dir = os.path.join(model_dir, "english", "14x14", "text_cot")
        seven_results = {}
        fourteen_results = {}
        if os.path.exists(seven_dir) and os.path.isdir(seven_dir):
            seven_results = process_folder(seven_dir)
        if os.path.exists(fourteen_dir) and os.path.isdir(fourteen_dir):
            fourteen_results = process_folder(fourteen_dir)
        if not seven_results or not fourteen_results:
            continue
        
        # find common clues
        common_clues = set(seven_results.keys()) & set(fourteen_results.keys())
        total_common = len(common_clues)
        clues_info = {}
        for clue in common_clues:
            data7 = seven_results[clue]
            data14 = fourteen_results[clue]
            accuracy_7 = data7["accuracy"]
            accuracy_14 = data14["accuracy"]
            count_7 = data7["count"]
            count_14 = data14["count"]
            avg_accuracy = (accuracy_7 + accuracy_14) / 2
            clues_info[clue] = {
                "7x7_accuracy": accuracy_7,
                "7x7_count": count_7,
                "14x14_accuracy": accuracy_14,
                "14x14_count": count_14,
                "avg_accuracy": avg_accuracy,
            }
        
        model_stats[model_name] = {
            "total_common_clues": total_common,
            "clues_info": clues_info
        }

    for model, stats in model_stats.items():
        print(f"Model: {model}")
        print(f"Repeated clue number: {stats['total_common_clues']}")
        overall_avg = 0
        if stats["total_common_clues"] > 0:
            overall_avg = sum(info["avg_accuracy"] for info in stats["clues_info"].values()) / stats["total_common_clues"]
        print(f"Overall average accuracy: {overall_avg:.2%}")
        
        table = []
        headers = ["Clue", "7x7 Accuracy", "7x7 Count", "14x14 Accuracy", "14x14 Count", "Avg Accuracy"]
        for clue, info in stats["clues_info"].items():
            table.append([
                clue,
                f"{info['7x7_accuracy']:.2%}",
                info["7x7_count"],
                f"{info['14x14_accuracy']:.2%}",
                info["14x14_count"],
                f"{info['avg_accuracy']:.2%}"
            ])
        print(tabulate(table, headers=headers, tablefmt="grid"))
        print("-" * 40)

if __name__ == "__main__":
    base_path = '../eval_results'
    main(base_path)
