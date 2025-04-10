import argparse
from datasets import load_dataset
from tqdm import tqdm

from config import Config, Credentials, LLMConfig
from evaluation import PROMPT_TEMPLATES
from llm_client import OpenAIClient
from reader import JSONLineReader

TYPES = ['simple', 'child', 'normal']
LANGUAGES = ['en', 'fr', 'ru', 'ar', 'zh']
OUTPUT_FILE = "batches/{dataset}/{model}/{dataset}-responses-{model}-{prompt_type}.jsonl"
TGI_BASE_URL = "https://api.fireworks.ai/inference/v1"
MODEL = "llama-v3p1-8b-instruct" #"deepseek-v3" #"llama4-maverick-instruct-basic" #"qwen3-30b-a3b"

# LLM client
CLIENT = OpenAIClient(LLMConfig(
    model=f"accounts/fireworks/models/{MODEL}",
    client_class="OpenAIClient",
    base_url=TGI_BASE_URL,
    api_key=Credentials.fw_api_key,
))

def generate_model_response(homonyms: list[dict], prompt_template: str, output_file: str):
    for entry in homonyms:
        word = entry["word"]
        prompt = prompt_template.format(word=word)
        model_response = CLIENT.define_term(prompt)
        result = {
            "word": word,
            "model_response": model_response,
        }
        JSONLineReader().write(output_file, [result])

def evaluate_dataset(dataset: list[dict], lang: str, prompt_type: str, context: bool, dataset_name: str):
    suffix = f"w_context_{lang}" if context else lang
    prompt_key = f"{prompt_type}_{suffix}"
    prompt = PROMPT_TEMPLATES.get(prompt_key)

    if prompt is None:
        print(f"Prompt not found for key: {prompt_key}")
        return

    output_path = OUTPUT_FILE.format(dataset=dataset_name, model=MODEL, prompt_type=prompt_key)
    generate_model_response(dataset, prompt, output_path)

def main():
    parser = argparse.ArgumentParser(description="Evaluate model prompts over datasets.")
    parser.add_argument('--types', type=str, nargs='+', default=TYPES,
                        help='Prompt types to evaluate. Default: %(default)s')
    parser.add_argument('--languages', type=str, nargs='+', default=LANGUAGES,
                        help='Languages to evaluate. Default: %(default)s')
    parser.add_argument('--dataset-multi', type=str, default='mcl-wic',
                        help='Multilingual dataset name. Default: %(default)s')
    parser.add_argument('--without-multilingual', action='store_true',
                        help='Skip multilingual dataset evaluation.')
    parser.add_argument('--without-context', action='store_true',
                        help='Evaluate only without context.')
    parser.add_argument('--without-hown', action='store_true',
                        help='Skip HoWN (homonymy-high-freq) evaluation.')

    args = parser.parse_args()
    contexts = [False] if args.without_context else [True, False]

    # Evaluate multilingual dataset
    if not args.without_multilingual:
        dataset_name = args.dataset_multi
        dataset_multi = load_dataset(Config.DATASETS[dataset_name], token=Credentials.hf_api_key)

        for context in tqdm(contexts, desc="Contexts"):
            for lang in tqdm(args.languages, desc="Languages", leave=False):
                dataset = dataset_multi[lang].to_list()
                for prompt_type in tqdm(args.types, desc="Prompt Types", leave=False):
                    evaluate_dataset(dataset, lang, prompt_type, context, dataset_name)

    # Evaluate HoWN
    if not args.without_hown:
        dataset_name = 'homonymy-high-freq'
        dataset_hown = load_dataset(Config.DATASETS[dataset_name], token=Credentials.hf_api_key)
        dataset = dataset_hown['train'].to_list()

        for context in tqdm(contexts, desc="HoWN Contexts"):
            for prompt_type in tqdm(args.types, desc="HoWN Prompt Types", leave=False):
                evaluate_dataset(dataset, lang='en', prompt_type=prompt_type, context=context, dataset_name=dataset_name)

if __name__ == "__main__":
    main()