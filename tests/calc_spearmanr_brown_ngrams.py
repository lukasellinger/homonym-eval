from datasets import load_dataset
from scipy.stats import spearmanr
from config import Config, Credentials

# Load dataset
dataset = load_dataset(Config.DATASETS['homonymy-high-freq'], token=Credentials.hf_api_key)['train']

brown = dataset['brown_occurence']
google = dataset['avg_google_ngrams_frequency']

# Keep only entries where both values are not None
valid_pairs = [(b, g) for b, g in zip(brown, google) if b is not None and g is not None]

# Unzip the filtered pairs
brown_filtered, google_filtered = zip(*valid_pairs)

# Compute Spearman correlation
spearman_corr, _ = spearmanr(brown_filtered, google_filtered)
print(f"Spearman correlation (rank similarity): {spearman_corr:.3f}")