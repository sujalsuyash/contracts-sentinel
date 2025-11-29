import pandas as pd
import numpy as np
import re
import os
import joblib
import json
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURATION & PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

SYNTHETIC_SAFE = [
    # Design & Creative
    "The graphic designer will deliver the final logo in PNG and SVG formats.",
    "The photographer agrees to edit 50 photos within 5 business days.",
    "The video editor will provide two rounds of revisions for the final cut.",
    "Illustrator will create 5 vector assets for the marketing campaign.",
    "The UX designer will provide wireframes via Figma.",
    # Tech & Dev
    "The web developer shall upload the code to the GitHub repository every Friday.",
    "The app will be built using React Native and Firebase.",
    "Developer will set up the staging environment on AWS.",
    "The contractor will fix bugs reported in the Jira backlog.",
    "Source code ownership remains with the client upon full payment.",
    # Writing & Marketing
    "Writer will submit the first draft of the blog post by noon.",
    "Services include social media management and content creation.",
    "The copywriter will deliver 3 email newsletters per month.",
    "Consultant is hired to perform search engine optimization (SEO) services.",
    "The translator will convert the document from English to Spanish.",
    # Admin & General (The "Virtual Assistant" Fix)
    "Consultant will provide weekly status updates via email.",
    "The contractor shall attend a Zoom meeting once a week to discuss progress.",
    "The project timeline is estimated to be three weeks from the start date.",
    "Client will provide all necessary assets and login credentials.",
    "The final deliverable will be a 10-page PDF report.",
    "This agreement outlines the scope of work for the project.",
    "Freelancer will track hours using Harvest and invoice bi-weekly.",
    "Virtual Assistant will manage the client's calendar and inbox.",
    "Data entry services will be performed using Excel sheets provided by Client.",
    # FIX: Explicitly teach that "Weekly Basis" for tasks is SAFE
    "Perform data entry into the provided Excel spreadsheets on a weekly basis.",
    "Organize the Outlook inbox and manage Google Calendar.",
    "Administrative support shall be provided on a daily basis."
]

ADDITIONAL_NEW_RISKS = [
    # --- PAYMENT TERMS (Ghost Pay & Aggressive Terms) ---
    {"text": "Consultant acknowledges that payment is contingent upon the Company's receipt of funds from the End Client.", "label_name": "Payment Terms"},
    {"text": "If the End Client fails to pay, Company shall have no obligation to remit fees.", "label_name": "Payment Terms"},
    {"text": "Pay-when-paid: Contractor gets paid only when Client gets paid.", "label_name": "Payment Terms"},
    {"text": "Payment is conditional upon the client being paid by the customer.", "label_name": "Payment Terms"},
    {"text": "Fees shall only be disbursed following settlement by the upstream customer.", "label_name": "Payment Terms"},
    {"text": "Company holds no liability for fees if upstream customer fails to pay.", "label_name": "Payment Terms"},
    {"text": "Client reserves the right to withhold payment for any reason in its sole discretion.", "label_name": "Payment Terms"},
    {"text": "Contractor shall not be reimbursed for any out-of-pocket expenses.", "label_name": "Payment Terms"},
    {"text": "Company shall be entitled to deduct and offset from any fees any amounts claims are owed.", "label_name": "Payment Terms"},

    # --- INDEMNIFICATION (Hidden Liability) ---
    {"text": "Indemnify, defend and hold harmless from any and all claims, losses, and liabilities.", "label_name": "Indemnification"},
    {"text": "Contractor assumes all liability for damages arising out of the performance of services.", "label_name": "Indemnification"},
    {"text": "Freelancer agrees to pay for all legal costs and judgments against the Client.", "label_name": "Indemnification"},
    {"text": "Contractor shall be liable for all incidental, consequential, and indirect damages.", "label_name": "Indemnification"},
    {"text": "There shall be no cap or limit on the Contractor's liability.", "label_name": "Indemnification"},
    # Informal Language Fix
    {"text": "If we are sued because of your work, you will be responsible for all our legal costs.", "label_name": "Indemnification"},
    {"text": "You agree to pay for any lawyers we hire if there is a problem with your work.", "label_name": "Indemnification"},
    {"text": "Any legal fees we incur will be billed back to you.", "label_name": "Indemnification"},

    # --- AUDIT RIGHTS (Invasive / Spyware) ---
    {"text": "Client shall have the right to inspect and audit books and records.", "label_name": "Audit Rights"},
    {"text": "Freelancer must allow access to premises for security audits.", "label_name": "Audit Rights"},
    # Spyware Fix
    {"text": "Client reserves the right to install monitoring software on Contractor's devices.", "label_name": "Audit Rights"},
    {"text": "We might need to install a time-tracking and screen-recording agent on your laptop.", "label_name": "Audit Rights"},
    {"text": "You must install our monitoring software to track your activity.", "label_name": "Audit Rights"},

    # --- NON-COMPETE (Stealth Mode) ---
    {"text": "Contractor shall devote their full productive capacity to the Company exclusively.", "label_name": "Non-Compete"},
    {"text": "During the term, Provider shall not engage in any other professional activities.", "label_name": "Non-Compete"},
    {"text": "Contractor agrees not to perform services for any other entity during this engagement.", "label_name": "Non-Compete"},
    {"text": "Exclusivity: Provider shall not accept other clients.", "label_name": "Non-Compete"},
    {"text": "Contractor agrees not to perform any services for any business operating in the same vertical as the Company.", "label_name": "Non-Compete"},
    {"text": "Restriction on working with competitors in the same industry vertical.", "label_name": "Non-Compete"},

    # --- INTELLECTUAL PROPERTY (Moral Rights & Perpetual) ---
    {"text": "Author waives all moral rights of paternity, integrity, and disclosure.", "label_name": "Intellectual Property"},
    {"text": "Waiver of moral rights in any jurisdiction worldwide.", "label_name": "Intellectual Property"},
    {"text": "Grant of a perpetual, irrevocable, worldwide, royalty-free license.", "label_name": "Intellectual Property"},
    {"text": "Work made for hire: Company owns all results and proceeds.", "label_name": "Intellectual Property"},

    # --- ASSIGNMENT (The Subcontracting Fix) ---
    {"text": "The Services are deemed personal in nature and may not be delegated.", "label_name": "Assignment"},
    {"text": "Contractor may not subcontract or transfer any duties to a third party.", "label_name": "Assignment"},
    {"text": "Rights and obligations under this Agreement are non-assignable without consent.", "label_name": "Assignment"},

    # --- ARBITRATION (Venue Traps) ---
    {"text": "Any dispute arising hereunder shall be resolved exclusively by binding arbitration.", "label_name": "Arbitration"},
    {"text": "Arbitration administered by the ICC in Singapore.", "label_name": "Arbitration"},
    {"text": "Submitted to a private dispute resolution service for a final, binding decision.", "label_name": "Arbitration"},
    {"text": "Waiving all rights to a civil court trial in favor of a neutral third party.", "label_name": "Arbitration"},

    # --- AMENDMENT (Unilateral Changes) ---
    {"text": "Client reserves the right to modify the Terms of Service at any time by posting an updated version.", "label_name": "Amendment"},
    {"text": "We reserve the right to modify the terms of this engagement at any time by sending you an email.", "label_name": "Amendment"},
    {"text": "Continued work implies acceptance of new terms sent via email.", "label_name": "Amendment"},

    # --- WARRANTY (Zombie Clauses) ---
    {"text": "Contractor warrants that the Work will be free from defects in perpetuity.", "label_name": "Warranty"},
    {"text": "Obligation to fix bugs shall survive termination indefinitely.", "label_name": "Warranty"},

    # --- SAFE ANCHORS (The "Don't Panic" Fixes) ---
    # 1. Data Privacy (GDPR) -> Confidentiality (Low Risk)
    {"text": "Vendor agrees to implement appropriate technical measures to ensure the security of Client Data.", "label_name": "Confidentiality"},
    {"text": "Process such data solely in accordance with GDPR and applicable privacy laws.", "label_name": "Confidentiality"},
    
    # 2. Background IP -> Confidentiality (Low Risk)
    {"text": "Contractor retains all right, title, and interest in and to any Background Technology.", "label_name": "Confidentiality"},
    {"text": "Anything you owned before this project remains yours.", "label_name": "Confidentiality"},

    # 3. Limitation of Liability -> Other (Safe)
    {"text": "Either party's liability under this Agreement is limited to the value of the contract.", "label_name": "Other"},
    {"text": "In no event shall either party be liable for indirect, special, or consequential damages.", "label_name": "Other"},
    {"text": "Total liability shall not exceed the fees paid to Contractor in the preceding 12 months.", "label_name": "Other"},
    {"text": "Hiring Party will not hold Freelance Worker in breach for failure to complete work due to illness.", "label_name": "Other"}
]

# --- 3. MAPPINGS ---
CUAD_MAPPING = {
    'Non-Compete': 'Non-Compete', 'Exclusivity': 'Non-Compete', 'Rofr/Rofo/Rofn': 'Non-Compete',
    'No-Solicit Of Customers': 'Non-Solicitation', 'No-Solicit Of Employees': 'Non-Solicitation',
    'Competitive Restriction Exception': 'Non-Compete',
    'Revenue/Profit Sharing': 'Payment Terms', 'Minimum Commitment': 'Payment Terms',
    'Price Restrictions': 'Payment Terms', 'Source Code Escrow': 'Intellectual Property',
    'Cap On Liability': 'Indemnification', 'Uncapped Liability': 'Indemnification',
    'Liquidated Damages': 'Indemnification',
    'Ip Ownership Assignment': 'Intellectual Property', 'Joint Ip Ownership': 'Intellectual Property',
    'License Grant': 'Intellectual Property', 'Unlimited/All-You-Can-Eat-License': 'Intellectual Property',
    'Irrevocable Or Perpetual License': 'Intellectual Property', 'Affiliate License-Licensee': 'Intellectual Property',
    'Affiliate License-Licensor': 'Intellectual Property',
    'Termination For Convenience': 'Termination', 'Post-Termination Services': 'Termination',
    'Renewal Term': 'Termination', 'Notice Period To Terminate Renewal': 'Termination',
    'Expiration Date': 'Termination',
    'Governing Law': 'Governing Law', 'Audit Rights': 'Audit Rights', 'Insurance': 'Insurance',
    'Warranty Duration': 'Warranty', 'Anti-Assignment': 'Assignment', 'Covenant Not To Sue': 'Waiver'
}

ID_MAPPING = {
    4: 'Governing Law', 6: 'Governing Law', 21: 'Governing Law', 47: 'Governing Law', 56: 'Governing Law',
    82: 'Governing Law', 94: 'Governing Law', 27: 'Termination', 30: 'Termination', 54: 'Termination',
    61: 'Termination', 62: 'Termination', 88: 'Termination', 89: 'Termination', 25: 'Indemnification',
    41: 'Indemnification', 49: 'Indemnification', 50: 'Indemnification', 7: 'Assignment', 8: 'Assignment',
    13: 'Assignment', 74: 'Assignment', 84: 'Assignment', 63: 'Waiver', 96: 'Waiver', 97: 'Waiver',
    38: 'Entire Agreement', 52: 'Entire Agreement', 34: 'Amendment', 60: 'Amendment', 20: 'Confidentiality',
    64: 'Confidentiality', 71: 'Confidentiality', 85: 'Confidentiality', 79: 'Severability', 65: 'Notices',
    26: 'Counterparts', 59: 'Counterparts', 53: 'Intellectual Property',
}

# --- 4. HELPER FUNCTIONS ---

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'§', '', text)            
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]', '', text) 
    text = re.sub(r'[0-9]+\.[0-9]+', '', text) 
    return text.strip()

def load_cuad_dataset():
    cuad_path = os.path.join(DATA_DIR, 'CUAD_v1.json')
    if not os.path.exists(cuad_path):
        print(f" CUAD dataset not found at {cuad_path}. Training will skip CUAD.")
        return pd.DataFrame(columns=['text', 'label_name'])

    print(" Parsing CUAD JSON ...")
    with open(cuad_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for paper in data['data']:
        for qas in paper['paragraphs'][0]['qas']:
            raw_id = qas['id']
            if "__" in raw_id:
                clean_label = raw_id.split("__")[-1]
            else:
                clean_label = raw_id
            
            if clean_label in CUAD_MAPPING:
                mapped_label = CUAD_MAPPING[clean_label]
                for answer in qas['answers']:
                    text_snippet = answer['text']
                    if len(text_snippet) > 40: 
                        rows.append({'text': text_snippet, 'label_name': mapped_label})
    
    print(f" Extracted {len(rows)} clauses from CUAD.")
    return pd.DataFrame(rows)

def prepare_dataset(df_ledgar, df_cuad, is_training=False):
    target_ids = list(ID_MAPPING.keys())
    ledgar_target = df_ledgar[df_ledgar['label'].isin(target_ids)].copy()
    ledgar_target['label_name'] = ledgar_target['label'].map(ID_MAPPING)
    
    other_subset = df_ledgar[~df_ledgar['label'].isin(target_ids)]
    n_samples = min(len(other_subset), 6000)
    other_df = other_subset.sample(n=n_samples, random_state=42).copy()
    other_df['label_name'] = 'Other'

    if is_training:
        print(" Loading Synthetic Data ...")
        # Boost SAFE text so "Weekly Basis" is learned as Safe
        safe_df = pd.DataFrame({'text': SYNTHETIC_SAFE * 100, 'label_name': 'Other'}) 
        additional_new_risks_df = pd.DataFrame(ADDITIONAL_NEW_RISKS * 100)
        
        other_df = pd.concat([other_df, safe_df])
        ledgar_target = pd.concat([ledgar_target, additional_new_risks_df])

    final_df = pd.concat([ledgar_target, other_df, df_cuad], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return final_df

def generate_semantic_anchors(train_df):
    print("\nGenerating Semantic Anchors ...")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    cuad_cats = set(CUAD_MAPPING.values())
    additional_new_risks_cats = set([item['label_name'] for item in ADDITIONAL_NEW_RISKS])
    ANCHOR_CATEGORIES = list(cuad_cats.union(additional_new_risks_cats))    
    anchors = {}
    
    for category in ANCHOR_CATEGORIES:
        cat_df = train_df[train_df['label_name'] == category]
        
        # --- PART 1: K-Means ---
        if len(cat_df) >= 5:
            all_texts = cat_df['text'].tolist()
            all_vectors = semantic_model.encode(all_texts, convert_to_tensor=False)
            
            num_clusters = min(len(all_texts), 25)
            clustering_model = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
            clustering_model.fit(all_vectors)
            category_anchors = torch.tensor(clustering_model.cluster_centers_)
        else:
            category_anchors = torch.tensor([])

        # --- PART 2: additional_new_risks Injection ---
        additional_new_risks_texts = [item['text'] for item in ADDITIONAL_NEW_RISKS if item['label_name'] == category]
        
        if additional_new_risks_texts:
            print(f"   -> Loading {len(additional_new_risks_texts)} additional_new_risks Anchors for: {category}")
            additional_new_risks_vectors = semantic_model.encode(additional_new_risks_texts, convert_to_tensor=True)
            
            if len(category_anchors) > 0:
                category_anchors = torch.cat((category_anchors, additional_new_risks_vectors))
            else:
                category_anchors = additional_new_risks_vectors
        
        if len(category_anchors) > 0:
            anchors[category] = category_anchors

    anchor_path = os.path.join(MODELS_DIR, 'semantic_anchors.pkl')
    joblib.dump(anchors, anchor_path)
    print(f" Semantic Anchors saved to {anchor_path}")

# --- 5. MAIN EXECUTION ---

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("--- 1. Loading Data ---")
    df_train_ledgar = pd.read_csv(os.path.join(DATA_DIR, 'ledgar_train.csv'))
    df_test_ledgar = pd.read_csv(os.path.join(DATA_DIR, 'ledgar_test.csv'))
    df_cuad = load_cuad_dataset()
    
    print("\n--- 2. Preparing & Cleaning ---")
    train_df = prepare_dataset(df_train_ledgar, df_cuad, is_training=True)
    test_df = prepare_dataset(df_test_ledgar, pd.DataFrame(), is_training=False) 
    
    train_df['clean_text'] = train_df['text'].apply(clean_text)
    test_df['clean_text'] = test_df['text'].apply(clean_text)

    # --- 3. GENERATE SEMANTIC BRAIN ---
    generate_semantic_anchors(train_df)

    # --- 4. TRAIN STATISTICAL MODEL (SVM) ---
    print("\n--- 4. Training SVM Classifier ---")
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 3), stop_words='english')
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    y_train = train_df['label_name']
    X_test = vectorizer.transform(test_df['clean_text'])
    y_test = test_df['label_name']

    print("   Fitting SVM...")
    svm_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, random_state=42, max_iter=2000, class_weight='balanced')
    clf = CalibratedClassifierCV(estimator=svm_clf, method='isotonic')
    clf.fit(X_train, y_train)
    
    print("\n--- 5. Evaluation ---")
    predictions = clf.predict(X_test)
    print(classification_report(y_test, predictions))
    
    print("\n--- 6. Saving Artifacts ---")
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'vectorizer.pkl'))
    joblib.dump(clf, os.path.join(MODELS_DIR, 'classifier.pkl'))
    print("All models saved successfully.")

if __name__ == "__main__":
    train()