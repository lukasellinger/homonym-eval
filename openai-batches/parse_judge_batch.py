import json

from datasets import load_dataset

from config import Config, Credentials, PROJECT_DIR
from evaluation_parser import EvaluationParser
from reader import JSONLineReader

TYPE = 'simple_w_context'
DATASET = 'homonymy-high-freq'
RESPONSES_FILE = f'{PROJECT_DIR}/batches/{DATASET}/{DATASET}-responses-gpt-4o-mini-{TYPE}.jsonl'
RAW_JUDGES_FILE = f'{PROJECT_DIR}/batches/{DATASET}/{DATASET}-raw-output-judge-{TYPE}.jsonl'
OUTPUT_FILE = f'{PROJECT_DIR}/batches/{DATASET}/{DATASET}-output-judge-{TYPE}.jsonl'
PARSED_OUTPUT_FILE = f'{PROJECT_DIR}/batches/{DATASET}/{DATASET}-output-judge-{TYPE}-parsed.jsonl'

dataset = load_dataset(Config.DATASETS[DATASET], token=Credentials.hf_api_key)['train'].to_list()

judge_outputs = JSONLineReader().read(RAW_JUDGES_FILE)
responses = JSONLineReader().read(RESPONSES_FILE)

intermediate_results = []
for result in judge_outputs:
    c_id = result['custom_id'].split('task-')[1]
    model_response = None
    for response in responses:
        if response['custom_id'].split('task-')[1] == c_id:
            model_response = response['response']['body']['choices'][0]['message']['content']
            break
    assert model_response is not None, f'Could not find model response for id {c_id}'

    entry = dataset[int(c_id)]

    try:
        message = result['response']['body']['output'][0]['content'][0]['text']
        evaluation = json.loads(message)
    except Exception:
        print(f'Could not parse response for id {c_id} - {message}')
        continue

    intermediate_results.append(
        {
            "word": entry['word'],
            "model_response": model_response,
            "evaluation": evaluation,
            "avg_google_ngrams_frequency": entry["avg_google_ngrams_frequency"]
        }
    )

JSONLineReader().write(OUTPUT_FILE, intermediate_results)

EvaluationParser().parse_evaluation(OUTPUT_FILE, PARSED_OUTPUT_FILE)
