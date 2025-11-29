import pandas as pd
import os

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def show_truth():
    train_path = os.path.join(DATA_DIR, 'ledgar_train.csv')
    if not os.path.exists(train_path):
        print("Data file not found.")
        return

    print("Loading data... (This might take 10 seconds)")
    df = pd.read_csv(train_path)
    
    # Get Top 20 most frequent IDs
    top_ids = df['label'].value_counts().head(20).index.tolist()
    
    print("\n--- THE TRUTH TABLE ---")
    print(f"{'ID':<5} | {'Snippet (First 80 chars)':<80}")
    print("-" * 90)
    
    for label_id in top_ids:
        # Get one example text
        text = df[df['label'] == label_id]['text'].iloc[0]
        # Remove newlines for clean printing
        clean_snip = text.replace('\n', ' ')[:80]
        print(f"{label_id:<5} | {clean_snip}...")

if __name__ == "__main__":
    show_truth()