from datasets import load_dataset
from config import Config, PROJECT_DIR, Credentials
from reader import JSONLineReader

TYPES = ['child', 'simple', 'normal']


for TYPE in TYPES:
    OUTPUT_FILE = f'{PROJECT_DIR}/batches/homonymy-high-freq/gpt-4o-mini/homonymy-high-freq-responses-gpt-4o-mini-{TYPE}_en.jsonl'
    INPUT_FILE = f'{PROJECT_DIR}/batches/homonymy/homonymy-output-judge-{TYPE}-parsed.jsonl'

    results = JSONLineReader().read(INPUT_FILE)
    filtered_data = load_dataset(Config.DATASETS['homonymy-high-freq'], token=Credentials.hf_api_key)['train'].to_list()

    filtered_results = []
    for entry in filtered_data:
        for result in results:
            if result['word'] == entry['word']:
                filtered_results.append({'word': result['word'], 'model_response': result['model_response']})
                break

    JSONLineReader().write(OUTPUT_FILE, filtered_results)