import warnings

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
from tqdm import tqdm

from config import Config, Credentials
from reader import JSONLineReader


class EvaluationParser:
    def __init__(self):
        self.sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        dataset = load_dataset(Config.DATASETS['homonymy'], token=Credentials.hf_api_key)['train']
        self.word_to_result = {entry['word']: entry for entry in dataset}

    def parse_evaluation(self, file_in: str, file_out: str, add_wordnet: bool = True):
        results = JSONLineReader().read(file_in)
        parsed_results = []
        for result in tqdm(results):
            parsed_result = result
            try:
                parsed_result = self.parse_definitions(parsed_result)
            except AssertionError:
                print(f'{parsed_result['word']} - amount of definitions does not match selected category')
                continue

            parsed_result = self.parse_category(parsed_result)
            parsed_result = self.parse_context(parsed_result)
            if add_wordnet:
                parsed_result = self.add_wordnet_ranking(result)
            del parsed_result['evaluation']
            parsed_results.append(parsed_result)
        JSONLineReader().write(file_out, parsed_results)
        return parsed_results

    @staticmethod
    def parse_context(result: dict) -> dict:
        remark = result.get('evaluation', {}).get('remark_not_all_listed')
        context_clarification = result.get('evaluation', {}).get('context_clarification_request')

        result['remark_not_all_listed'] = remark
        result['context_clarification_request'] = context_clarification
        result['complete_marker'] = remark or context_clarification
        return result

    @staticmethod
    def parse_definitions(result: dict) -> dict:
        definitions_raw = result.get('evaluation', {}).get('definitions', [])
        category = result.get('evaluation').get('category')

        if category == "One":
            assert len(definitions_raw) == 1, f"Expected 1 definition, got {len(definitions_raw)}"
        elif category == "Multiple":
            assert len(definitions_raw) > 1, f"Expected >1 definitions, got {len(definitions_raw)}"
        elif category == "None":
            assert len(definitions_raw) == 0, f"Expected 0 definitions, got {len(definitions_raw)}"

        result['definitions'] = definitions_raw
        return result

    @staticmethod
    def parse_category(result: dict) -> dict:
        CATEGORIES = {"None", "One", "Multiple"}

        category = result.get('evaluation', {}).get('category')
        if category not in CATEGORIES:
            print(f'Unknown category: {category}')
        else:
            result['category'] = category
        return result

    def add_wordnet_ranking(self, result: dict) -> dict:
        word = result.get('word')
        definitions = result.get('definitions')
        wn_definitions = [{'name': synset.name(), 'definition': synset.definition()} for synset in wn.synsets(word.replace(' ', '_'), pos='n')]

        wn_rankings = []
        all_similarities = []  # to store similarities for each definition

        for definition in definitions:
            all_definitions = [definition] + [wn_def.get('definition') for wn_def in wn_definitions]
            embeddings = self.sim_model.encode(all_definitions)
            def_embedding = embeddings[0]
            wn_embeddings = embeddings[1:]
            similarities = cosine_similarity([def_embedding], wn_embeddings)[0]

            max_sim = np.max(similarities)
            wn_rankings.append(np.argmax(similarities).item() if max_sim > 0.4 else -1)
            all_similarities.append(similarities)

        result['wordnet_rankings'] = wn_rankings

        # # Step 1: group matched definitions by wn_idx
        # wn_to_defs = {}
        # for i, wn_idx in enumerate(wn_rankings):
        #     if wn_idx == -1:
        #         continue
        #     sim = all_similarities[i][wn_idx]
        #     wn_to_defs.setdefault(wn_idx, []).append((i, sim))
        #
        # # Step 2: keep only best-matching definition per wn sense
        # kept_indices = set()
        # for wn_idx, matches in wn_to_defs.items():
        #     # Find the def index with the highest similarity for this wn_idx
        #     best_match = max(matches, key=lambda x: x[1])
        #     best_i = best_match[0]
        #     kept_indices.add(best_i)
        #
        # # Step 3: add unmatched definitions (wn_idx == -1)
        # for i, wn_idx in enumerate(wn_rankings):
        #     if wn_idx == -1:
        #         kept_indices.add(i)
        #
        # # Step 4: build cleaned_definitions
        # cleaned_definitions = [definitions[i] for i in sorted(kept_indices)]
        # result['definitions'] = cleaned_definitions
        #
        # # Step 5: update category if only one cleaned definition remains
        # if len(cleaned_definitions) == 1:
        #     result['category'] = 'One'


        coarse_synsets = self.word_to_result.get(word, {}).get('coarse_synsets')
        covered = set()
        for wn_ranking in wn_rankings:
            if wn_ranking == -1:
                continue

            matched_synset = wn_definitions[wn_ranking]
            for coarse_synset in coarse_synsets:
                if coarse_synset['name'] in covered:
                    continue

                for synset in coarse_synset.get('synsets', []):
                    if synset.get('name') == matched_synset.get('name'):
                        covered.add(coarse_synset['name'])
                        break
        result['coarse_synsets_covered'] = round(len(covered) / len(coarse_synsets), 2)
        return result


if __name__ == "__main__":
    file_in = 'batches/homonymy-high-freq/llama4-maverick-instruct-basic-new-old/homonymy-high-freq-llama4-maverick-instruct-basic-output-judge-simple_en.jsonl'
    file_out = 'batches/homonymy/homonymy-output-judge-child-parsed.jsonl'
    EvaluationParser().parse_evaluation(file_in, file_out)
