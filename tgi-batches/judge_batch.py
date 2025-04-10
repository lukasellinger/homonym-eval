import re

from datasets import load_dataset

from LLMJudge import JUDGE_PROMPT_TEMPLATE, MARKER_SYSTEM_PROMPT, DEFINITION_SYSTEM_PROMPT, \
    DEFINITION_SCHEMA, MARKER_SCHEMA, CHILD_DEFINITION_SYSTEM_PROMPT
from config import LLMConfig, Config, Credentials, PROJECT_DIR
from evaluation import PROMPT_TEMPLATES
from llm_client import OpenAIClient
from reader import JSONLineReader

DATASET = 'mcl-wic' #'homonymy-high-freq'
TYPES = ['simple', 'child', 'normal']
LANGUAGES = ['en', 'fr', 'ar', 'ru', 'zh']
CONTEXTS = [False]
MODEL = 'dpo-llama-v3p1-8b-instruct'
OUTPUT_FILE = f'{PROJECT_DIR}/batches/{DATASET}/{MODEL}/{DATASET}-{MODEL}-input-judge.jsonl'

client = OpenAIClient(LLMConfig(
            model="gpt-4o-mini-2024-07-18", #"gpt-4.1-mini-2025-04-14",
            client_class="OpenAIClient",
))

datadict = load_dataset(Config.DATASETS[DATASET], token=Credentials.hf_api_key)

tasks = []
for sub_judgement in ['marker', 'definition']:
    for lang in LANGUAGES:
        if DATASET == 'homonymy-high-freq':
            dataset = datadict['train'].to_list()
        else:
            dataset = datadict[lang].to_list()

        for context in CONTEXTS:
            for TYPE in TYPES:
                if sub_judgement == "marker":
                    system_prompt = MARKER_SYSTEM_PROMPT
                    schema = MARKER_SCHEMA
                else:
                    system_prompt = (
                        CHILD_DEFINITION_SYSTEM_PROMPT if TYPE == "child"
                        else DEFINITION_SYSTEM_PROMPT
                    )
                    schema = DEFINITION_SCHEMA

                suffix = f"w_context_{lang}" if context else lang
                prompt_key = f"{TYPE}_{suffix}"
                prompt = PROMPT_TEMPLATES.get(prompt_key)

                RESPONSE_FILE = f'{PROJECT_DIR}/batches/{DATASET}/{MODEL}/{DATASET}-responses-{MODEL}-{prompt_key}.jsonl'

                dataset_response = JSONLineReader().read(RESPONSE_FILE)
                for idx, entry in enumerate(dataset_response):
                    word = dataset[idx]['word']
                    response = entry.get('model_response')
                    if not response: # gpt-4o-mini
                        response = entry['response']['body']['choices'][0]['message']['content']
                    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

                    task = {
                        "custom_id": f"task-{sub_judgement}-{lang}-{context}-{TYPE}-{idx}",
                        "method": "POST",
                        "url": '/v1/responses',
                        "body": {
                            "model": "gpt-4o-mini-2024-07-18",
                            "temperature": 0,
                            "input": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user",
                                 "content": JUDGE_PROMPT_TEMPLATE.format(query=prompt.format(word=word), model_response=response)},
                            ],
                            "text": schema
                        }
                    }
                    tasks.append(task)

JSONLineReader().write(OUTPUT_FILE, tasks)
client.create_batch_job(OUTPUT_FILE, endpoint='/v1/responses')
