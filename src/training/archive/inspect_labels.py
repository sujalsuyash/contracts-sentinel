import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
train_path = os.path.join(BASE_DIR, 'data', 'ledgar_train.csv')

def inspect():
    if not os.path.exists(train_path):
        print(f"Error: File not found at {train_path}")
        return

    print("Loading dataset...")
    df = pd.read_csv(train_path)

    print("\n--- TOP 20 LABELS IN YOUR DATASET ---")
    print(df['label'].value_counts().head(20))
    
    print("\n--- CHECKING YOUR TARGETS ---")
    targets_to_check = ['Termination', 'Indemnification', 'Governing Law', 'Waiver', 'Non-Compete']
    
    for t in targets_to_check:
        # Check if the singular or plural version exists
        matches = df[df['label'].str.contains(t, case=False, na=False)]['label'].unique()
        print(f"Variations found for '{t}': {matches}")

if __name__ == "__main__":
    inspect()