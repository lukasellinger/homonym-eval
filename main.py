from argparse import ArgumentParser

from datasets import load_dataset

from config import Config, Credentials
from evaluation import HomonymEvaluator
from analysis import ResultAnalyzer
from evaluation_parser import EvaluationParser


def validate_llm(llm: str) -> bool:
    return llm in Config.SUPPORTED_LLMS

def main(response_llm: str = Config.DEFAULT_RESPONSE_LLM):
    parser = ArgumentParser()
    parser.add_argument("--prompt_type", default='simple', type=str,
                        help="The type of prompt to use. Default 'simple'. Choose between 'child', 'simple', 'normal'")
    args = parser.parse_args()

    if not validate_llm(response_llm):
        print(f"Invalid LLM selected. Supported LLMs: {list(Config.SUPPORTED_LLMS.keys())}")
        return

    homonyms = load_dataset(Config.DATASET, token=Credentials.hf_api_key)['train'].to_list()
    print(f"Loaded {len(homonyms)} homonyms from HF")

    results_file = Config.RESULTS_FILE.replace('prompt_type', args.prompt_type)
    parsed_results_file = Config.PARSED_RESULTS_FILE.replace('prompt_type', args.prompt_type)

    evaluator = HomonymEvaluator(response_llm=response_llm, prompt_type=args.prompt_type)
    evaluator.evaluate_homonyms(homonyms, results_file)
    print(f"Results saved to {results_file}")

    EvaluationParser().parse_evaluation(results_file, parsed_results_file)
    ResultAnalyzer().analyze(parsed_results_file)

if __name__ == "__main__":
    main()