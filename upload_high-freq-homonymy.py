import pickle
from datasets import Dataset, load_dataset

from config import Credentials


with open('filtered_homonyms_w_occurences.pkl', 'rb') as f:
    filtered_homonyms_w_occurences = pickle.load(f)

total = 0
high_occurence_homonyms = {}
for k, v in filtered_homonyms_w_occurences.items():
    count = 0
    for def_key, def_value in v.items():
        if def_value.get('semcor_occurances', 0) > 0:
            count += 1
    if count > 1:
        total += 1
        high_occurence_homonyms[k] = v

dataset = load_dataset(
    "lukasellinger/homonym-homonymy-wsd",
    token=Credentials.hf_api_key
)['train']

data = []
for word, v in high_occurence_homonyms.items():
    for entry in dataset:
        if entry['word'] == word:
            avg_google_ngrams_frequency = entry['avg_google_ngrams_frequency']
            coarse_synsets = [{"name": name,
                               "semcor_occurances": info['semcor_occurances'],
                               "synsets": [{"name": synset[0], "definition": synset[1]} for synset in info['synsets']]} for name, info in v.items()]
            data.append({'word': word, 'avg_google_ngrams_frequency': avg_google_ngrams_frequency, 'coarse_synsets': coarse_synsets})
            break

dataset = Dataset.from_list(data)
dataset.push_to_hub(
    repo_id="lukasellinger/homonym-high-freq-homonymy-wsd",
    private=True,
    token=Credentials.hf_api_key
)

print(len(dataset))