import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
train_path = os.path.join(BASE_DIR, 'data', 'ledgar_train.csv')

def decode():
    if not os.path.exists(train_path):
        print(f"Error: File not found at {train_path}")
        return

    print("Loading dataset...")
    df = pd.read_csv(train_path)

    top_labels = df['label'].value_counts().head(15).index.tolist()
    
    print("\n--- DETECTIVE MODE ACTIVATED ---")
    print("We are looking at the text for each numeric label to guess its name.\n")
    
    for label_id in top_labels:
        example_text = df[df['label'] == label_id]['text'].iloc[0]

        print(f"LABEL ID: {label_id}")
        print(f"Snippet: {example_text[:200]}...") # Show first 200 chars
        print("-" * 60)

if __name__ == "__main__":
    decode()