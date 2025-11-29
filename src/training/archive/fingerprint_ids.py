import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def get_top_phrases(corpus):
    """
    Finds the 'Signature Phrases' (DNA) of a category.
    TARGET: 5 to 6 words (The User's Sweet Spot).
    """
    if len(corpus) < 10: return ["(Not enough data)"]
    
    # ngram_range=(5, 6): Looks for 5 or 6 word patterns.
    # stop_words='english': Ignores "the", "and", "of" to focus on the legal verbs/nouns.
    vec = CountVectorizer(ngram_range=(5, 6), stop_words='english', max_features=3)
    
    try:
        bag_of_words = vec.fit_transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        
        # Sort by frequency (Most common first)
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return [w[0] for w in words_freq]
    except ValueError:
        return ["(No consistent phrase found)"]

def fingerprint_all():
    train_path = os.path.join(DATA_DIR, 'ledgar_train.csv')
    if not os.path.exists(train_path): return

    print("Loading dataset...")
    df = pd.read_csv(train_path)
    
    # Filter for significant categories (count > 50)
    counts = df['label'].value_counts()
    valid_ids = counts[counts > 50].index
    df = df[df['label'].isin(valid_ids)]
    
    grouped = df.groupby('label')
    
    print(f"\n{'ID':<5} | {'Count':<6} | {'SIGNATURE PHRASES (5-6 Words)'}")
    print("-" * 130)
    
    # Loop through IDs in order
    for label_id in sorted(grouped.groups.keys()):
        group = grouped.get_group(label_id)
        
        # Take up to 1000 rows to ensure we catch the pattern
        corpus = group['text'].head(1000).astype(str).tolist()
        
        phrases = get_top_phrases(corpus)
        
        # Format the output nicely
        phrase_str = " || ".join([f"'{p}'" for p in phrases])
        
        # Truncate if too long for screen
        if len(phrase_str) > 100:
            phrase_str = phrase_str[:100] + "..."
            
        print(f"{label_id:<5} | {len(group):<6} | {phrase_str}")

if __name__ == "__main__":
    fingerprint_all()