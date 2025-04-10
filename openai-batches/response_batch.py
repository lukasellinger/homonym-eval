from datasets import load_dataset

from config import LLMConfig, Config, Credentials, PROJECT_DIR
from evaluation import PROMPT_TEMPLATES
from llm_client import OpenAIClient

TYPE = 'child_w_context'
DATASET = 'homonymy-high-freq'
OUTPUT_FILE = f'{PROJECT_DIR}/batches/{DATASET}/{DATASET}-input-gpt-4o-mini-{TYPE}.jsonl'

client = OpenAIClient(LLMConfig(
            model="gpt-4o-mini-2024-07-18",
            client_class="OpenAIClient",
))

dataset = load_dataset(Config.DATASETS[DATASET], token=Credentials.hf_api_key)['train'].to_list()
client.create_tasks(dataset,"gpt-4o-mini-2024-07-18", 0, OUTPUT_FILE, lambda entry: PROMPT_TEMPLATES.get(f'{TYPE}_en').format(word=entry['word']))

client.create_batch_job(OUTPUT_FILE)
