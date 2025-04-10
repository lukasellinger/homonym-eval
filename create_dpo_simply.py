from collections import defaultdict

from datasets import load_dataset

from analysis import Analysis
from config import Credentials
from reader import JSONLineReader

reader = JSONLineReader()
analysis = Analysis()
results = analysis.get_hown_results()


dpos = []
train_words = defaultdict(set)
for type_ in ['simple']:
    word_to_entry = defaultdict(list)
    for model, model_results in results[type_].items():
        for entry in model_results:
            word_to_entry[entry['word']].append((model, entry))

    for result in results[type_]['Llama 3.1 8B']:
        if result['complete_marker'] or result['coarse_synsets_covered'] == 1:
            continue

        word = result['word']
        matching_entries = [
            (model, entry)
            for model, entry in word_to_entry[word]
            if model != 'Llama 3.1 8B'
        ]

        if type_ == 'simple':
            prompt = f"What is the definition of '{result['word']}' in simple language?"
        else:
            prompt = f"Explain me '{result['word']}' like I am 5 years old."

        for _, entry in matching_entries:
            if entry['complete_marker'] or entry['coarse_synsets_covered'] == 1:
                train_words[type_].add(entry['word'])
                dpo = {
                    "word": entry['word'],
                    "input": {
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    },
                    "preferred_output": [
                        {
                            "role": "assistant",
                            "content": entry["model_response"]
                        }
                    ],
                    "non_preferred_output": [
                        {
                            "role": "assistant",
                            "content": result["model_response"]
                        }
                    ]
                }
                dpos.append(dpo)

reader.write('llama-v3p1-8b-instruct-simple-dpo.jsonl', dpos)

def preprocess_dpo(example):
    return {
        "word": example['word'],
        "prompt": example["input"]["messages"][0]["content"],
        "chosen": example["preferred_output"][0]["content"],
        "rejected": example["non_preferred_output"][0]["content"]
    }

dataset = load_dataset("json", data_files='llama-v3p1-8b-instruct-simple-dpo.jsonl')["train"]
dataset = dataset.map(preprocess_dpo)
dataset.push_to_hub(
    repo_id="lukasellinger/homonymy-dpo",
    private=True,
    token=Credentials.hf_api_key
)