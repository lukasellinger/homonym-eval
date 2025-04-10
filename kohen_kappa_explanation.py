from config import PROJECT_DIR
from reader import JSONLineReader

checks = {
    'child': ['gpt-4o-mini', 'llama4-maverick-instruct-basic', 'dpo-llama-v3p1-8b-instruct'],
    'simple': ['deepseek-v3', 'llama-v3p1-8b-instruct', 'dpo-llama-v3p1-8b-instruct'],
    'normal': ['qwen3-30b-a3b', 'llama4-maverick-instruct-basic', 'dpo-llama-v3p1-8b-instruct'],
}

base_dir_human = f"{PROJECT_DIR}/batches/homonymy-high-freq/{{model}}/homonymy-high-freq-responses-{{model}}-{{type_}}_en-human-annotate.jsonl"
base_dir_auto = f"{PROJECT_DIR}/batches/homonymy-high-freq/{{model}}/homonymy-high-freq-{{model}}-output-judge-{{type_}}_en-parsed-raw.jsonl"

reader = JSONLineReader()
all_combined_evaluations = []


def compute_stats(evaluations, label=""):
    one_wrong = sum(
        1 for e in evaluations
        if e["human_evaluation"]["category"] == "One"
        and e["automatic_evaluation"]["category"] != "One"
    )
    multi_wrong = sum(
        1 for e in evaluations
        if e["human_evaluation"]["category"] != "One"
        and e["automatic_evaluation"]["category"] != e["human_evaluation"]["category"]
    )

    print(f"{label}Category: One Wrong {one_wrong} Multi Wrong {multi_wrong}, Total: {len(evaluations)}")
    if len(evaluations) > 0:
        print(f"{label}Category Accuracy: {100 - ((one_wrong + multi_wrong) / len(evaluations) * 100):.2f}")

    human_complete_true_mismatches = 0
    human_complete_false_mismatches = 0
    for e in evaluations:
        human = e["human_evaluation"]
        auto = e["automatic_evaluation"]

        human_complete = human.get("context_clarification_request") or human.get("remark_not_all_listed")
        auto_complete = auto.get("context_clarification_request") or auto.get("remark_not_all_listed")

        if human_complete != auto_complete:
            if human_complete:
                human_complete_true_mismatches += 1
            else:
                human_complete_false_mismatches += 1

    print(
        f"{label}Complete Marker: Human False Auto True {human_complete_false_mismatches} Human True Auto False {human_complete_true_mismatches}, Total: {len(evaluations)}")
    if len(evaluations) > 0:
        print(
            f"{label}Complete Marker Accuracy: {100 - ((human_complete_true_mismatches + human_complete_false_mismatches) / len(evaluations) * 100):.2f}")


for type_, models in checks.items():
    print(f"\nCalculating for {type_}...")

    all_evaluations = []
    for model in models:
        human_evaluations = reader.read(base_dir_human.format(model=model, type_=type_))
        auto_evaluations = reader.read(base_dir_auto.format(model=model, type_=type_))
        auto_evaluations = {entry['word']: entry for entry in auto_evaluations}

        for evaluation in human_evaluations:
            auto_evaluation = auto_evaluations.get(evaluation['word'])
            if not auto_evaluation:
                continue
            merged = {
                **evaluation,
                'automatic_evaluation': {
                    'category': auto_evaluation.get('category'),
                    'context_clarification_request': auto_evaluation.get('context_clarification_request'),
                    'remark_not_all_listed': auto_evaluation.get('remark_not_all_listed'),
                }
            }
            all_evaluations.append(merged)

    all_combined_evaluations.extend(all_evaluations)
    compute_stats(all_evaluations, label=f"{type_}: ")

print("\n=== Combined Stats Across All Types ===")
compute_stats(all_combined_evaluations, label="ALL: ")