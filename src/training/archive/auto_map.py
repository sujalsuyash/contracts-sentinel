import json
import os
import torch
from sentence_transformers import SentenceTransformer, util

# --- 1. SETUP ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'CUAD_v1.json')

# Your 12 Core Risk Pillars
MY_CATEGORIES = {
    "Termination": "Ending the agreement, cancellation, expiration, stop services, termination for convenience",
    "Indemnification": "Pay for damages, liability cap, hold harmless, reimburse legal fees, liquidated damages, uncapped liability",
    "Non-Compete": "Exclusivity, right of first refusal, rofr, non-solicitation, restrict competition, no-solicit",
    "Governing Law": "Jurisdiction, venue, laws of the state, dispute resolution, arbitration",
    "Audit Rights": "Inspect books, access records, verify compliance, examination",
    "Insurance": "Coverage requirements, policy limits, additional insured, workers compensation",
    "Intellectual Property": "Ownership of work, license grant, patents, copyrights, inventions, joint ownership",
    "Payment Terms": "Invoicing, net 30, fees, revenue share, minimum commitment, pricing",
    "Warranty": "Guarantee of quality, fix defects, warranty duration, product warranty",
    "Assignment": "Transfer rights, successors and assigns, delegation, anti-assignment",
    "Confidentiality": "Non-disclosure, secret information, return of data, privacy",
    "Waiver": "Giving up rights, failure to enforce, covenant not to sue"
}

def get_clean_unique_labels():
    """
    OPTIMIZED: Scans CUAD but splits the ID to remove the filename immediately.
    This reduces 20,000+ items down to just ~41 unique categories.
    """
    if not os.path.exists(DATA_PATH):
        print(f"Error: Could not find {DATA_PATH}")
        return []
    
    print("Scanning CUAD dataset... (Fast Mode)")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    unique_labels = set()

    first_contract = data['data'][0]
    
    for qas in first_contract['paragraphs'][0]['qas']:
        raw_id = qas['id']
        if "__" in raw_id:
            clean_label = raw_id.split("__")[-1]
        else:
            clean_label = raw_id
            
        unique_labels.add(clean_label)
            
    return list(unique_labels)

def auto_map():
    print("Loading AI Brain (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    unknown_labels = get_clean_unique_labels()
    print(f"Found {len(unknown_labels)} unique categories (Optimized).\n")

    cat_names = list(MY_CATEGORIES.keys())
    cat_descriptions = list(MY_CATEGORIES.values())
    cat_vectors = model.encode(cat_descriptions, convert_to_tensor=True)

    generated_map = {}

    print(f"{'CLEAN CUAD LABEL':<40} -> {'YOUR CATEGORY':<20} (Confidence)")
    print("-" * 75)

    for label in unknown_labels:
        # Encode the Clean Label
        label_vec = model.encode(label, convert_to_tensor=True)
        
        # Measure distance
        scores = util.cos_sim(label_vec, cat_vectors)[0]
        
        # Find winner
        best_score_idx = torch.argmax(scores).item()
        best_score = scores[best_score_idx].item()
        best_category = cat_names[best_score_idx]
        
        # Filter: Only map if confidence is reasonable
        if best_score > 0.25:
            generated_map[label] = best_category
            print(f"{label:<40} -> {best_category:<20} ({best_score:.2f})")
        else:
            print(f"{label:<40} -> [SKIPPED - Low Match]  ({best_score:.2f})")

    # --- OUTPUT THE CODE ---
    print("\n\n✅ DONE! Copy the block below into train_model.py:\n")
    print("-" * 30)
    print("CUAD_MAPPING = {")
    for k, v in sorted(generated_map.items()):
        print(f"    '{k}': '{v}',")
    print("}")
    print("-" * 30)

if __name__ == "__main__":
    auto_map()