from datasets import load_dataset
from tqdm import tqdm

from config import LLMConfig, Config, Credentials, PROJECT_DIR
from evaluation import PROMPT_TEMPLATES
from llm_client import OpenAIClient
from reader import JSONLineReader

DATASET = 'mcl-wic'
MODEL = 'lukasellinger/uncertain-dpo-llama-v3p1-8b-instruct'
MODEL_NAME = 'uncertain-dpo'
LANGUAGES = ['en', 'fr', 'ar', 'ru', 'zh']
OUTPUT_FILE = f'{PROJECT_DIR}/batches/{DATASET}/{MODEL_NAME}/{DATASET}-responses-{MODEL_NAME}_{{type_}}.jsonl'

client = OpenAIClient(LLMConfig(
            model=MODEL,
            client_class="OpenAIClient",
            base_url='https://api.runpod.ai/v2/j8erq8xjlg68rh/openai/v1',
            api_key=Credentials.runpod_api_key
))

data_dict = load_dataset(Config.DATASETS[DATASET], token=Credentials.hf_api_key)

for lang in tqdm(LANGUAGES):
    dataset = data_dict[lang].to_list()
    for TYPE in ['simple', 'normal', 'child']:
        for idx, entry in tqdm(enumerate(dataset)):
            type_lang = f'{TYPE}_{lang}'
            response = client.define_term(PROMPT_TEMPLATES.get(type_lang).format(word=entry['word']))
            output_response = {'word': entry['word'], 'model_response': response}
            JSONLineReader().write(OUTPUT_FILE.format(type_=type_lang), [output_response])
