from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import nltk
from nltk.corpus import semcor, wordnet as wn

import pickle as pkl

with open('homonymy-wsd/filtered_homonyms.pkl', 'rb') as f:
    filtered_homonyms = pkl.load(f)

def count_occurence(target_synset: str):
    target_synset = wn.synset(target_synset)
    count = 0
    for sent in semcor.tagged_sents(tag='sem'):
        for chunk in sent:
            if isinstance(chunk, nltk.tree.Tree):
                if isinstance(chunk.label(), nltk.corpus.reader.wordnet.Lemma):
                    synset = chunk.label().synset()
                    if synset == target_synset:
                        count += 1
    return count


def process_word(word_definitions_pair):
    word, definitions = word_definitions_pair
    word_result = {}

    for key, definition in definitions.items():
        total_count = 0
        for synset_tuple in definition:
            synset = synset_tuple[0]
            if synset:
                total_count += count_occurence(synset)
        word_result[key] = {
            'synsets': definition,
            'semcor_occurances': total_count
        }

    return word, word_result

if __name__ == "__main__":
    homonyms = list(filtered_homonyms.items())
    filtered_homonyms_w_occurences = {}

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_word, item) for item in homonyms]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel processing"):
            word, result = future.result()
            filtered_homonyms_w_occurences[word] = result

    pkl.dump(filtered_homonyms_w_occurences, open('filtered_homonyms_w_occurences.pkl', 'wb'))
