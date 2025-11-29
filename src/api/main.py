from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import re
import numpy as np
from pypdf import PdfReader
import io

# Import the semantic router
from src.api.semantic_router import check_semantic_match 

app = FastAPI(title="ContractSentinel API")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PATHS & CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

model_artifacts = {}

PRIORITY_CATEGORIES = [
    "Non-Compete", "Indemnification", "Audit Rights", 
    "Arbitration", "Intellectual Property", "Warranty", 
    "Insurance", "Payment Terms", "Termination"
]

RISK_MAPPING = {
    'Termination': {'level': 'High', 'description': 'Rules regarding how the contract ends or can be cancelled.'},
    'Indemnification': {'level': 'Critical', 'description': 'Requirement to pay for the other party’s legal costs or damages.'},
    'Non-Compete': {'level': 'High', 'description': 'Restrictions on working for competitors or soliciting clients.'},
    'Arbitration': {'level': 'High', 'description': 'Forces disputes into private arbitration (waiving jury trial).'},
    'Waiver': {'level': 'Medium', 'description': 'Giving up specific legal rights or claims.'},
    'Intellectual Property': {'level': 'Medium', 'description': 'Transfer of ownership of work/inventions to the client.'},
    'Confidentiality': {'level': 'Low', 'description': 'Standard NDA terms regarding secrecy.'},
    'Amendment': {'level': 'Medium', 'description': 'How the contract can be changed (requires written agreement).'},
    'Assignment': {'level': 'Medium', 'description': 'Rules on transferring the contract to others (subcontracting).'},
    'Governing Law': {'level': 'Medium', 'description': 'Which state/country laws apply to disputes.'},
    'Payment Terms': {'level': 'High', 'description': 'Payment schedules, late fees, and revenue sharing.'},
    'Insurance': {'level': 'High', 'description': 'Requirements to hold specific insurance policies.'},
    'Audit Rights': {'level': 'High', 'description': 'Client right to inspect your financial books and premises.'},
    'Warranty': {'level': 'High', 'description': 'Obligation to guarantee work quality for a specific duration.'},
    'Non-Solicitation': {'level': 'Medium', 'description': 'Restrictions on poaching clients or employees.'},
    'Other': {'level': 'Safe', 'description': 'Standard operational language.'}
}

@app.on_event("startup")
async def load_models():
    print("Starting API...")
    try:
        vec_path = os.path.join(MODELS_DIR, 'vectorizer.pkl')
        clf_path = os.path.join(MODELS_DIR, 'classifier.pkl')

        if os.path.exists(vec_path) and os.path.exists(clf_path):
            model_artifacts['vectorizer'] = joblib.load(vec_path)
            model_artifacts['classifier'] = joblib.load(clf_path)
            print("SVM Models loaded")
        else:
            print(" Error: Model files not found.")
    except Exception as e:
        print(f" Critical Error loading models: {e}")

# --- DATA MODELS ---
class ContractRequest(BaseModel):
    text: str

class ContractResponse(BaseModel):
    category: str
    confidence: float
    risk_level: str
    description: str

# --- HELPER FUNCTIONS ---
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'§', '', text)            
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]', '', text) 
    text = re.sub(r'[0-9]+\.[0-9]+', '', text) 
    return text.strip()

def run_ai_pipeline(raw_text: str):
    """
    This is the SHARED BRAIN. Both /analyze (Text) and /upload (PDF) call this.
    """
    clean_input = clean_text(raw_text)
    vec = model_artifacts['vectorizer']
    clf = model_artifacts['classifier']

    # 1. SVM Prediction
    vectorized_text = vec.transform([clean_input])
    svm_prediction = clf.predict(vectorized_text)[0]
    svm_probs = clf.predict_proba(vectorized_text)
    svm_confidence = float(np.max(svm_probs))

    # 2. Semantic Check
    semantic_result = check_semantic_match(raw_text)

    # 3. Decision Logic
    final_category = svm_prediction
    final_confidence = svm_confidence
    final_source = "SVM"

    if semantic_result:
        sem_cat = semantic_result['category']
        sem_conf = semantic_result['confidence']

        if (sem_cat in PRIORITY_CATEGORIES) or (sem_cat == "Other") or (svm_confidence < 0.90):
            final_category = sem_cat
            final_confidence = sem_conf
            final_source = "Semantic"

    # 4. Formatter
    risk_info = RISK_MAPPING.get(final_category, {'level': 'Safe', 'description': 'Standard text.'})
    
    if final_source == "Semantic":
        description = (
            f"{risk_info['description']}\n\n"
            f"Insight: The AI detected language conceptually similar to standard '{final_category}' clauses."
        )
    else:
        description = risk_info['description']

    print(f" {final_source} won: {final_category} ({final_confidence:.0%})")

    return {
        "category": final_category,
        "confidence": round(min(0.99, final_confidence), 2),
        "risk_level": risk_info['level'],
        "description": description
    }

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "ContractSentinel is running"}

@app.post("/analyze", response_model=ContractResponse)
def analyze_text_endpoint(request: ContractRequest):
    """Endpoint for manual text paste"""
    if 'vectorizer' not in model_artifacts:
        raise HTTPException(status_code=500, detail="Models loading...")
    return run_ai_pipeline(request.text)

@app.post("/upload", response_model=ContractResponse)
async def analyze_pdf_endpoint(file: UploadFile = File(...)):
    """Endpoint for PDF Upload"""
    if 'vectorizer' not in model_artifacts:
        raise HTTPException(status_code=500, detail="Models loading...")
    
    # Check file type
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Read and Extract Text
        content = await file.read()
        pdf = PdfReader(io.BytesIO(content))
        full_text = ""
        
        # Safety Limit: Only read first 10 pages to prevent memory crashes
        max_pages = min(len(pdf.pages), 10)
        
        for i in range(max_pages):
            text = pdf.pages[i].extract_text()
            if text:
                full_text += text + "\n"
        
        if len(full_text.strip()) < 10:
            raise HTTPException(status_code=400, detail="PDF appears to be empty or an image scan.")

        # Truncate to first 3000 characters for the MVP analysis
        # (In a pro version, you would loop through clauses)
        analyzable_chunk = full_text[:3000]
        
        return run_ai_pipeline(analyzable_chunk)

    except Exception as e:
        print(f"PDF Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process PDF.")