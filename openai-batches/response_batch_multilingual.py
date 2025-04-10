from datasets import load_dataset

from config import LLMConfig, Config, Credentials, PROJECT_DIR
from evaluation import PROMPT_TEMPLATES
from llm_client import OpenAIClient
from reader import JSONLineReader

DATASET = 'mcl-wic'
MODEL = 'gpt-4o-mini-2024-07-18'
LANGUAGES = ['en'] #, 'ar'] #, 'ru', 'zh']
OUTPUT_FILE = f'{PROJECT_DIR}/batches/{DATASET}/{DATASET}-input-gpt-4o-mini.jsonl'

client = OpenAIClient(LLMConfig(
            model=MODEL,
            client_class="OpenAIClient",
))

data_dict = load_dataset(Config.DATASETS[DATASET], token=Credentials.hf_api_key)
tasks = []
for lang in LANGUAGES:
    dataset = data_dict[lang].to_list()
    for TYPE in ['simple', 'normal', 'child']:
        for idx, entry in enumerate(dataset):
            task = {
                "custom_id": f"task-{lang}-{TYPE}-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": PROMPT_TEMPLATES.get(f'{TYPE}_{lang}').format(word=entry['word'])
                        }
                    ],
                }
            }
            tasks.append(task)

JSONLineReader().write(OUTPUT_FILE, tasks)
client.create_batch_job(OUTPUT_FILE)
