import time
from statistics import mean
import requests
from requests import HTTPError

from config import Config

class NgramFetcher:
    def __init__(self):
        self.config = Config.NGRAM_CONFIG

    def fetch_ngram_data(self, words: list[str], inflections: bool = False, corpus='en') -> dict[str, dict]:
        """Fetch Ngram frequency data for multiple words in batches."""
        result = {}
        batch_size = self.config.batch_size
        years = list(range(self.config.year_start, self.config.year_end + 1))

        if corpus == 'ru':
            words = [word.replace('ё', 'е') for word in words]

        # Process words in batches
        for i in range(0, len(words), batch_size):
            batch = words[i:i + batch_size]
            processed_batch = [
                f"[{word.replace('-', ' - ')}]" if '-' in word else f"{word}_INF" if inflections else word
                for word in batch
            ]
            query = ",".join(processed_batch)
            try:
                params = {
                    "content": query,
                    "year_start": self.config.year_start,
                    "year_end": self.config.year_end,
                    "corpus": corpus,
                    "smoothing": self.config.smoothing,
                }
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.87"
                }
                response = requests.get(self.config.base_url, params=params, headers=headers, timeout=30)
                try:
                    response.raise_for_status()
                except HTTPError as e:
                    print(f"{e} - sleeping 10 sec.")
                    time.sleep(10)
                    response = requests.get(self.config.base_url, params=params, headers=headers, timeout=30)
                    response.raise_for_status()
                data = response.json()

                if data and isinstance(data, list):
                    ngram_map = {entry["ngram"].replace(' ', ''): entry["timeseries"] for entry in data}
                    for word in batch:
                        word = word.replace(' ', '')
                        lookup_candidates = []
                        if "-" in word:
                            lookup_candidates.append(word)
                        else:
                            inf_batch_word = f"{word}_INF"
                            if inflections:
                                for key in [inf_batch_word, word]:
                                    lookup_candidates.extend([key, key.capitalize()])
                            else:
                                lookup_candidates.extend([word])

                        for key in lookup_candidates:
                            if key in ngram_map:
                                frequencies = ngram_map[key]
                                if len(frequencies) != len(years):
                                    print(
                                        f"Warning: Frequency data length mismatch for {word} (got {len(frequencies)}, expected {len(years)})")
                                result[word] = {
                                    "avg_frequency": mean(frequencies) if frequencies else 0.0,
                                }
                                break
                        else:
                            print(f"No Ngram data for {word}")
                            result[word] = {}
                else:
                    print(f"No valid Ngram data returned for batch: {batch}")
                    for word in batch:
                        result[word] = {}

            except Exception as e:
                print(f"Error fetching Ngram data for batch {batch}: {e}")
                for word in batch:
                    result[word] = {}

        return result


if __name__ == "__main__":
    print(NgramFetcher().fetch_ngram_data(['ещё'], inflections=False, corpus='ru'))
    #print(NgramFetcher().fetch_ngram_data(['创造者'], inflections=False, corpus='zh'))

    # Inflections posses multiple problems, see examples below, e.g. odessa_INF -> only ODESSA, and what do to with saint thomas
    #print(NgramFetcher().fetch_ngram_data(['saint thomas'], inflections=True))
    #print(NgramFetcher().fetch_ngram_data(['odessa'], inflections=True))
    #print(NgramFetcher().fetch_ngram_data(['ball'], inflections=True))