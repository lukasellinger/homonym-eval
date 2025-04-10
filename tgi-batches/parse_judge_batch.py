import json
import re

from datasets import load_dataset
from tqdm import tqdm

from LLMJudge import LLMJudge
from config import Config, Credentials, PROJECT_DIR
from evaluation_parser import EvaluationParser
from reader import JSONLineReader

DATASET = 'homonymy-high-freq'
TYPES = ['simple', 'child', 'normal']
LANGUAGES = ['en']
CONTEXTS = [False] #[False]
MODEL = 'dpo-llama-v3p1-8b-instruct'
DIR = f'{PROJECT_DIR}/batches/{DATASET}/{MODEL}/'

RAW_JUDGES_FILE = f'{PROJECT_DIR}/batches/{DATASET}/{MODEL}/{DATASET}-{MODEL}-raw-output-judge.jsonl'
judge_outputs = JSONLineReader().read(RAW_JUDGES_FILE)

datadict = load_dataset(Config.DATASETS[DATASET], token=Credentials.hf_api_key)

for lang in tqdm(LANGUAGES):
    if DATASET == 'homonymy-high-freq':
        dataset = datadict['train'].to_list()
    else:
        dataset = datadict[lang].to_list()

    for context in CONTEXTS:
        for TYPE in TYPES:
            suffix = f"w_context_{lang}" if context else lang
            prompt_key = f"{TYPE}_{suffix}"

            RESPONSES_FILE = f'{DIR}/{DATASET}-responses-{MODEL}-{prompt_key}.jsonl'
            OUTPUT_FILE = f'{DIR}/{DATASET}-{MODEL}-output-judge-{prompt_key}.jsonl'
            PARSED_OUTPUT_FILE = f'{DIR}/{DATASET}-{MODEL}-output-judge-{prompt_key}-parsed-raw.jsonl'

            # responses = JSONLineReader().read(RESPONSES_FILE)
            #
            # intermediate_results = []
            # for idx, response in enumerate(responses):
            #     raw_model_response = response.get('model_response')
            #     if not raw_model_response: # gpt-4o-mini
            #         raw_model_response = response['response']['body']['choices'][0]['message']['content']
            #     model_response = re.sub(r"<think>.*?</think>", "", raw_model_response, flags=re.DOTALL).strip()
            #
            #     marker_task_id = f"task-marker-{lang}-{context}-{TYPE}-{idx}"
            #     definition_task_id = f"task-definition-{lang}-{context}-{TYPE}-{idx}"
            #
            #     def_result, marker_result = None, None
            #     for result in judge_outputs:
            #         if result['custom_id'] == marker_task_id:
            #             marker_result = result
            #         if result['custom_id'] == definition_task_id:
            #             def_result = result
            #         if def_result and marker_result:
            #             break
            #
            #     assert def_result is not None, f'Could not find judge result for id {definition_task_id}'
            #     assert marker_result is not None, f'Could not find judge result for id {marker_task_id}'
            #
            #     entry = dataset[idx]
            #
            #     try:
            #         marker_message = marker_result['response']['body']['output'][0]['content'][0]['text']
            #         marker_eval = json.loads(marker_message)
            #
            #         def_message = def_result['response']['body']['output'][0]['content'][0]['text']
            #         def_eval = json.loads(def_message)
            #     except Exception:
            #         print(f'Could not parse response for id {definition_task_id} - {marker_task_id}')
            #         continue
            #
            #     evaluation = LLMJudge.combine_judgments(marker_eval, def_eval)
            #
            #     intermediate_results.append(
            #         {
            #             "word": entry['word'],
            #             "model_response": model_response,
            #             "evaluation": evaluation,
            #             "avg_google_ngrams_frequency": entry["avg_google_ngrams_frequency"]
            #         }
            #     )
            #
            # JSONLineReader().write(OUTPUT_FILE, intermediate_results)

            add_wordnet = DATASET == 'homonymy-high-freq'
            EvaluationParser().parse_evaluation(OUTPUT_FILE, PARSED_OUTPUT_FILE, add_wordnet=add_wordnet)
