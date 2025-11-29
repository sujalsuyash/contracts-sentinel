import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def discover_topics():
    train_path = os.path.join(DATA_DIR, 'ledgar_train.csv')
    if not os.path.exists(train_path): return

    print("Loading dataset for Topic Discovery...")
    df = pd.read_csv(train_path)
    
    # Group all text by ID
    # We create a "Document" for each ID containing ALL its text
    print("Grouping text by Category ID...")
    grouped = df.groupby('label')['text'].apply(lambda x: " ".join(x.astype(str))).reset_index()
    
    # TF-IDF Magic
    # We look for words that are highly frequent in ONE ID but rare in others.
    # This filters out generic words like "agreement" or "shall".
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(2, 3))
    tfidf_matrix = tfidf.fit_transform(grouped['text'])
    feature_names = tfidf.get_feature_names_out()
    
    print(f"\n{'ID':<5} | {'Count':<6} | {'DISCOVERED TOPIC NAME (Based on Math)'}")
    print("-" * 100)
    
    # For each ID, get the top 3 scoring words
    for i, row in grouped.iterrows():
        label_id = row['label']
        count = len(df[df['label'] == label_id])
        
        # Get the row from the matrix
        row_vector = tfidf_matrix[i]
        
        # Sort words by score
        scores = zip(row_vector.indices, row_vector.data)
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
        
        top_words = [feature_names[idx] for idx, score in sorted_scores]
        topic_name = " + ".join(top_words)
        
        print(f"{label_id:<5} | {count:<6} | {topic_name}")

if __name__ == "__main__":
    discover_topics()