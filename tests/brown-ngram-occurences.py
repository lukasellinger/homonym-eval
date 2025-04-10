from collections import Counter
from datasets import load_dataset, Dataset
from nltk.corpus import brown
from nltk.util import ngrams
from config import Config, Credentials

tokens = [w.lower() for w in brown.words()]

unigram_freq = Counter(ngrams(tokens, 1))
bigram_freq = Counter(ngrams(tokens, 2))

dataset = load_dataset(Config.DATASETS['homonymy'], token=Credentials.hf_api_key)['train']

def get_brown_occurrence(word: str):
    split = word.lower().split()
    if len(split) == 1:
        return unigram_freq.get((split[0],), 0)
    elif len(split) == 2:
        return bigram_freq.get((split[0], split[1]), 0)
    return 0

dataset = dataset.map(lambda entry: {
    **entry,
    'brown_occurence': get_brown_occurrence(entry['word'])
})

dataset.push_to_hub(
    repo_id=Config.DATASETS['homonymy'],
    private=True,
    token=Credentials.hf_api_key
)