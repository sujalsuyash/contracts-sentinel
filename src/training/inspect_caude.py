import json
import os
import random

# Path to your downloaded CUAD_v1.json
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILE_PATH = os.path.join(BASE_DIR, 'data', 'CUAD_v1.json')

def xray_cuad():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    print("Loading CUAD JSON... (This takes a few seconds)")
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    found_categories = {}

    print("Scanning...")
    for paper in data['data']:
        for qas in paper['paragraphs'][0]['qas']:
            label = qas['id']  

            if qas['answers'] and label not in found_categories:
                example_text = qas['answers'][0]['text']
                found_categories[label] = example_text

    print(f"\n Found {len(found_categories)} unique categories.\n")
    print(f"{'CUAD LABEL':<35} | {'EXAMPLE TEXT (Truncated)'}")
    print("-" * 100)
    
    for label, text in found_categories.items():
        clean_text = text[:80].replace('\n', ' ') + "..."
        print(f"{label:<35} | {clean_text}")

if __name__ == "__main__":
    xray_cuad()