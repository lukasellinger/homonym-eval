from datasets import load_dataset

from LLMJudge import JUDGE_PROMPT_TEMPLATE
from config import LLMConfig, Config, Credentials, PROJECT_DIR
from evaluation import PROMPT_TEMPLATES
from llm_client import OpenAIClient
from reader import JSONLineReader

TYPE = 'simple_w_context'
DATASET = 'homonymy-high-freq'
RESPONSE_FILE = f'{PROJECT_DIR}/batches/{DATASET}/{DATASET}-responses-gpt-4o-mini-{TYPE}.jsonl'
OUTPUT_FILE = f'{PROJECT_DIR}/batches/{DATASET}/{DATASET}-input-judge-{TYPE}.jsonl'

client = OpenAIClient(LLMConfig(
            model="gpt-4.1-mini-2025-04-14", #gpt-4o-mini-2024-07-18",
            client_class="OpenAIClient",
))

dataset = load_dataset(Config.DATASETS[DATASET], token=Credentials.hf_api_key)['train'].to_list()

dataset_response = JSONLineReader().read(RESPONSE_FILE)

tasks = []
for idx, entry in enumerate(dataset_response):
    c_id = int(entry.get('custom_id').split('task-')[1])
    word = dataset[c_id]['word']
    response = entry['response']['body']['choices'][0]['message']['content']
    task = {
        "custom_id": f"task-{c_id}",
        "method": "POST",
        "url": '/v1/responses',
        "body": {
            "model": "gpt-4.1-mini-2025-04-14", #gpt-4o-mini-2024-07-18",
            "temperature": 0,
            "input": [
                {"role": "user", "content": JUDGE_PROMPT_TEMPLATE.format(query=PROMPT_TEMPLATES.get(f'{TYPE}_en').format(word=word), model_response=response)},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "judge_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "number_of_definitions": {
                                "type": "string",
                                "enum": ["0", "1", "Multiple"]
                            },
                            "context_requested": {
                                "type": "string",
                                "enum": ["Yes", "No"]
                            },
                            "definitions": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "explanation": {
                                "type": "string"
                            }
                        },
                        "required": ["number_of_definitions", "context_requested", "definitions", "explanation"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        }
    }
    tasks.append(task)

JSONLineReader().write(OUTPUT_FILE, tasks)
client.create_batch_job(OUTPUT_FILE, endpoint='/v1/responses')
