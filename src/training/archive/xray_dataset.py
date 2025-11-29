import pandas as pd
import os
from collections import Counter
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def get_top_phrase(text_series):
    all_text = " ".join(text_series.astype(str).tolist()).lower()
    clean = re.sub(r'[^a-z\s]', '', all_text)
    words = clean.split()
    bigrams = zip(words, words[1:])
    trigrams = zip(words, words[1:], words[2:])
    
    counts = Counter(trigrams)
    most_common = counts.most_common(1)[0][0]
    return " ".join(most_common)

def xray():
    train_path = os.path.join(DATA_DIR, 'ledgar_train.csv')
    if not os.path.exists(train_path): return

    print("Loading full dataset...")
    df = pd.read_csv(train_path)

    grouped = df.groupby('label')
    
    print(f"\n{'ID':<5} | {'Count':<6} | {'Most Common Phrase (The Fingerprint)':<40}")
    print("-" * 80)

    sorted_groups = sorted(grouped, key=lambda x: len(x[1]), reverse=True)

    for label_id, group in sorted_groups[:50]:
        fingerprint = get_top_phrase(group['text'].head(100))
        print(f"{label_id:<5} | {len(group):<6} | {fingerprint}")

if __name__ == "__main__":
    xray()