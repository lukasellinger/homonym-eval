import argparse

from datasets import load_dataset
from tqdm import tqdm

from config import Config, Credentials, LLMConfig
from evaluation import PROMPT_TEMPLATES
from llm_client import LLamaCPPClient
from reader import JSONLineReader

TYPES = ['simple', 'child', 'normal']
LANGUAGES = ['en', 'fr', 'ru', 'ar', 'zh']
OUTPUT_FILE = "mistral/{dataset}-responses-{model}-{prompt_type}.jsonl"
TGI_BASE_URL = "social-large.soc.in.tum.de"
MODEL = 'mistral-7b-instruct'

CLIENT = LLamaCPPClient(LLMConfig(model="mistral-7b-instruct-v0.2.Q4_K_M.gguf"))

def generate_model_response(homonyms: list[dict], prompt_template: str, output_file: str):
    for h in homonyms:
        word = h["word"]
        prompt = prompt_template.format(word=word)
        model_response = CLIENT.define_term(prompt)
        result = {
            "word": word,
            "model_response": model_response,
        }
        JSONLineReader().write(output_file, [result])


def main():
    parser = argparse.ArgumentParser(description="Example of list input")
    parser.add_argument('--types', type=str, nargs='+', default=TYPES,
                        help='A list of prompts to evaluate. Default: ["simple", "child", "normal"]')
    parser.add_argument('--languages', type=str, nargs='+', default=LANGUAGES,
                        help='A list of languages to evaluate. Default: ["en", "fr", "ru", "ar", "zh"]')
    parser.add_argument('--dataset-multi', type=str, default='mcl-wic', help='Dataset to evaluate. Default: "mcl-wic"')
    parser.add_argument('--without-context', action='store_true', default=True, help='Analyze without context')
    parser.add_argument('--without-hown', action='store_true', default=False,
                        help='Do not analyze HoWN (homonymy-high-freq) dataset')

    args = parser.parse_args()

    dataset_name = args.dataset_multi
    dataset_multi = load_dataset(Config.DATASETS[dataset_name], token=Credentials.hf_api_key)
    contexts = [False] if args.without_context else [True, False]

    for context in tqdm(contexts, desc="Processing contexts"):
        for lang in tqdm(args.languages, desc="Processing languages", leave=False):
            for prompt_type in tqdm(args.types, desc="Processing prompt types", leave=False):
                dataset = dataset_multi[lang].to_list()
                prompt_key = f'{prompt_type}_{'w_context_' if context else ''}{lang}'
                prompt = PROMPT_TEMPLATES.get(prompt_key)
                generate_model_response(dataset, prompt,
                                        OUTPUT_FILE.format(dataset=dataset_name, model=MODEL, prompt_type=prompt_key))

    if args.without_hown:
        return

    dataset_name = 'homonymy-high-freq'
    dataset_hown = load_dataset(Config.DATASETS['homonymy-high-freq'], token=Credentials.hf_api_key)
    dataset = dataset_hown['train'].to_list()

    for context in tqdm(contexts, desc="Processing HoWN contexts"):
        for prompt_type in tqdm(args.types, desc="Processing HoWN prompt types", leave=False):
            prompt_key = f'{prompt_type}_{'w_context_' if context else ''}en'
            prompt = PROMPT_TEMPLATES.get(prompt_key)
            generate_model_response(dataset, prompt,
                                    OUTPUT_FILE.format(dataset=dataset_name, model=MODEL, prompt_type=prompt_key))

if __name__ == "__main__":
    main()