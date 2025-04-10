from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter

lemmatizer = WordNetLemmatizer()

def is_noun(tag):
    return tag.startswith('NN')  # includes NN, NNS, NNP, NNPS

def get_wordnet_pos(tag):
    return wordnet.NOUN if is_noun(tag) else wordnet.NOUN  # default to noun here

# Get tagged words
tagged_words = brown.tagged_words()

# Filter nouns and lemmatize them
nouns = [
    lemmatizer.lemmatize(word.lower(), pos=get_wordnet_pos(tag))
    for word, tag in tagged_words if is_noun(tag)
]

# Count noun frequencies
noun_freq = Counter(nouns)

# Example
print(f"Lemmatized count for 'bank': {noun_freq['bank']}")