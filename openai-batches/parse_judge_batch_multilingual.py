import json

from datasets import load_dataset
from tqdm import tqdm

from config import Config, Credentials, PROJECT_DIR
from evaluation_parser import EvaluationParser
from reader import JSONLineReader

DATASET = 'mcl-wic'
LANGUAGES = ['en']

datadict = load_dataset(Config.DATASETS[DATASET], token=Credentials.hf_api_key)
for lang in tqdm(LANGUAGES):
    RAW_JUDGES_FILE = f'{PROJECT_DIR}/batches/{DATASET}/{DATASET}-raw-output-judge-{lang}.jsonl'
    judge_outputs = JSONLineReader().read(RAW_JUDGES_FILE)
    dataset = datadict[lang].to_list()

    for TYPE in ['child', 'simple', 'normal']:
        RESPONSES_FILE = f'{PROJECT_DIR}/batches/{DATASET}/{DATASET}-responses-gpt-4o-mini-{TYPE}-{lang}.jsonl'
        OUTPUT_FILE = f'{PROJECT_DIR}/batches/{DATASET}/{DATASET}-output-judge-{TYPE}-{lang}.jsonl'
        PARSED_OUTPUT_FILE = f'{PROJECT_DIR}/batches/{DATASET}/{DATASET}-output-judge-{TYPE}-{lang}-parsed.jsonl'

        responses = JSONLineReader().read(RESPONSES_FILE)

        intermediate_results = []
        for response in responses:
            c_id = response['custom_id'].split(f'task-{lang}-{TYPE}-')[1]
            model_response = response['response']['body']['choices'][0]['message']['content']

            found_result = None
            for result in judge_outputs:
                if TYPE in result['custom_id'] and result['custom_id'].split(f'task-{lang}-{TYPE}-')[1] == c_id:
                    found_result = result
                    break
            assert found_result is not None, f'Could not find judge result for id {c_id} - {lang} - {TYPE}'

            entry = dataset[int(c_id)]

            try:
                message = found_result['response']['body']['output'][0]['content'][0]['text']
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

        EvaluationParser().parse_evaluation(OUTPUT_FILE, PARSED_OUTPUT_FILE, add_wordnet=False)
