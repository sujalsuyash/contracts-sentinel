import pandas as pd
import numpy as np
import os
import joblib
import torch
import re
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Priority categories (Must match main.py)
PRIORITY_CATEGORIES = [
    "Non-Compete", "Indemnification", "Audit Rights", 
    "Arbitration", "Intellectual Property", "Warranty", 
    "Insurance", "Payment Terms", "Termination"
]

# --- FULL COVERAGE TEST SET ---
# Ensures every category appears at least once so you don't get "0.00" scores.
GOLDEN_RISKS_TEST = [
    # 1. Payment Terms
    {"text": "Consultant acknowledges that payment is contingent upon the Company's receipt of funds from the End Client.", "label": "Payment Terms"},
    {"text": "Client shall remit payment 120 days after invoice receipt.", "label": "Payment Terms"},
    
    # 2. Audit Rights
    {"text": "Client shall have the right to inspect and audit Service Provider's books and records.", "label": "Audit Rights"},
    {"text": "We reserve the right to install monitoring software to track your activity.", "label": "Audit Rights"},
    
    # 3. Non-Compete
    {"text": "Contractor shall devote their full productive capacity to the Company exclusively.", "label": "Non-Compete"},
    {"text": "Provider agrees not to work for any competitor in the same vertical.", "label": "Non-Compete"},
    
    # 4. Insurance
    {"text": "Contractor must maintain General Liability Insurance with coverage of at least $1,000,000.", "label": "Insurance"},
    {"text": "Waiver of subrogation endorsement required on all policies.", "label": "Insurance"},
    
    # 5. Indemnification
    {"text": "Freelancer agrees to indemnify, defend, and hold harmless the Client from any claims.", "label": "Indemnification"},
    {"text": "You are responsible for all legal costs if we are sued.", "label": "Indemnification"},
    
    # 6. Intellectual Property
    {"text": "Author waives all moral rights of paternity and integrity.", "label": "Intellectual Property"},
    {"text": "All work product shall be the sole and exclusive property of the Client.", "label": "Intellectual Property"},
    
    # 7. Assignment
    {"text": "The Services are deemed personal in nature and may not be delegated.", "label": "Assignment"},
    {"text": "No subcontracting without prior written consent.", "label": "Assignment"},
    
    # 8. Arbitration
    {"text": "Disputes shall be resolved by private binding arbitration in Singapore.", "label": "Arbitration"},
    
    # 9. Warranty
    {"text": "Contractor warrants the code will remain bug-free in perpetuity.", "label": "Warranty"},
    
    # 10. Confidentiality (Safe Data)
    {"text": "Vendor agrees to implement technical measures to ensure security of Client Data.", "label": "Confidentiality"},
    
    # 11. Other (Safe)
    {"text": "The web developer will upload the source code to GitHub weekly.", "label": "Other"}
]

# ID MAPPING (for LEDGAR)
ID_MAPPING = {
    4: 'Governing Law', 6: 'Governing Law', 21: 'Governing Law', 47: 'Governing Law', 56: 'Governing Law', 82: 'Governing Law', 94: 'Governing Law',
    27: 'Termination', 30: 'Termination', 54: 'Termination', 61: 'Termination', 62: 'Termination', 88: 'Termination', 89: 'Termination',
    25: 'Indemnification', 41: 'Indemnification', 49: 'Indemnification', 50: 'Indemnification',
    7: 'Assignment', 8: 'Assignment', 13: 'Assignment', 74: 'Assignment', 84: 'Assignment',
    63: 'Waiver', 96: 'Waiver', 97: 'Waiver',
    38: 'Entire Agreement', 52: 'Entire Agreement', 34: 'Amendment', 60: 'Amendment',
    20: 'Confidentiality', 64: 'Confidentiality', 71: 'Confidentiality', 85: 'Confidentiality',
    79: 'Severability', 65: 'Notices', 26: 'Counterparts', 59: 'Counterparts', 53: 'Intellectual Property',
}

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'§', '', text)            
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]', '', text) 
    text = re.sub(r'[0-9]+\.[0-9]+', '', text) 
    return text.strip()

def load_models():
    print("Loading ...")
    vec = joblib.load(os.path.join(MODELS_DIR, 'vectorizer.pkl'))
    clf = joblib.load(os.path.join(MODELS_DIR, 'classifier.pkl'))
    anchors = joblib.load(os.path.join(MODELS_DIR, 'semantic_anchors.pkl'))
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    return vec, clf, anchors, semantic_model

def get_semantic_prediction(text, semantic_model, anchors, threshold=0.35):
    input_vec = semantic_model.encode(text, convert_to_tensor=True)
    best_score = 0
    best_category = None
    
    for category, anchor_tensor in anchors.items():
        cosine_scores = util.cos_sim(input_vec, anchor_tensor)
        max_score = torch.max(cosine_scores).item()
        if max_score > best_score:
            best_score = max_score
            best_category = category
            
    if best_score > threshold:
        return best_category, best_score
    return None, 0.0

def run_hybrid_logic(text, vec, clf, semantic_model, anchors):
    clean_input = clean_text(text)
    
    # 1. SVM Prediction
    vectorized_text = vec.transform([clean_input])
    svm_pred = clf.predict(vectorized_text)[0]
    svm_prob = float(np.max(clf.predict_proba(vectorized_text)))
    
    # 2. Semantic Prediction
    sem_pred, sem_prob = get_semantic_prediction(text, semantic_model, anchors)
    
    # 3. The Logic Gate
    final_pred = svm_pred
    
    if sem_pred:
        if (sem_pred in PRIORITY_CATEGORIES) or (sem_pred == "Other") or (svm_prob < 0.90):
            final_pred = sem_pred
            
    return final_pred

def evaluate():
    vec, clf, anchors, semantic_model = load_models()
    
    # 1. Load LEDGAR Data
    print(" Loading LEDGAR Test Data...")
    df_full = pd.read_csv(os.path.join(DATA_DIR, 'ledgar_test.csv'))
    
    target_mask = df_full['label'].isin(ID_MAPPING.keys())
    df_risks = df_full[target_mask].copy()
    df_risks['label_name'] = df_risks['label'].map(ID_MAPPING)
    
    df_others = df_full[~target_mask].copy()
    df_others['label_name'] = 'Other'
    
    # Take samples
    df_sample_risks = df_risks.sample(n=min(len(df_risks), 300), random_state=42)
    df_sample_others = df_others.sample(n=min(len(df_others), 50), random_state=42)
    
    X_exam = df_sample_risks['text'].tolist() + df_sample_others['text'].tolist()
    y_true_exam = df_sample_risks['label_name'].tolist() + df_sample_others['label_name'].tolist()
    
    # 2. Add Manual Coverage Data
    X_golden = [item['text'] for item in GOLDEN_RISKS_TEST]
    y_true_golden = [item['label'] for item in GOLDEN_RISKS_TEST]
    
    # 3. Combine
    X_all = X_exam + X_golden
    y_true_all = y_true_exam + y_true_golden
    
    print(f"Running Evaluation on {len(X_all)} examples...")
    
    y_pred_all = []
    for text in X_all:
        pred = run_hybrid_logic(text, vec, clf, semantic_model, anchors)
        y_pred_all.append(pred)
        
    print("\n" + "="*60)
    print(" SYSTEM PERFORMANCE REPORT")
    print("="*60)
    print(classification_report(y_true_all, y_pred_all, zero_division=0))
    print(f"Overall Accuracy: {accuracy_score(y_true_all, y_pred_all):.2%}")

if __name__ == "__main__":
    evaluate()