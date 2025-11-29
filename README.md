_**CONTRACT SENTINEL: AI-POWERED LEGAL RISK DETECTOR**_ ->

Engineered a full-stack legal technology application designed to democratize contract review for freelancers and gig workers. Developed a Hybrid AI architecture that combines statistical classification with deep semantic understanding to analyze legal documents and identify high-risk clauses. Optimized the system for zero-cost deployment, achieving production-grade accuracy on a shared CPU environment within a 512MB RAM constraint.

_**BUSINESS PROBLEM**_ ->

Accessibility: Professional legal review is cost-prohibitive for freelancers, exposing them to predatory terms like "pay-when-paid" clauses and unlimited liability. Technical Limitation: Traditional keyword-based search fails to detect semantic traps (e.g., "client holds no liability" effectively meaning "zero payment"), while Large Language Models (LLMs) are too expensive to host for free. Objective: Develop a lightweight, high-precision intelligence engine capable of distinguishing between standard boilerplate and critical financial risks without relying on external paid APIs.

_**TECHNICAL ARCHITECTURE & METHODOLOGY**_ ->

Hybrid AI Engine Implemented a "Judge and Jury" architecture to balance speed and accuracy.

Statistical Model (The Judge): Trained an SGDClassifier (Linear SVM) with TF-IDF vectorization on the LEDGAR dataset (80,000+ clauses) to instantly classify standard boilerplate text like "Governing Law" or "Entire Agreement" with high confidence.

Semantic Brain (The Jury): Integrated a quantized Sentence Transformer (all-MiniLM-L6-v2) to detect nuanced risks. Utilized K-Means Clustering to generate mathematical centroids for 12 distinct risk categories based on the CUAD dataset.

Priority Logic Gate: Engineered a custom decision algorithm that allows the Semantic Brain to override the Statistical Model when high-severity risks (e.g., Indemnification) are detected, preventing false negatives on critical clauses.

_**Data Engineering & Augmentation**_ ->

Dataset Unification: Merged the LEDGAR dataset (SEC filings) with the CUAD dataset (Commercial contracts), standardizing labels across 40+ disparate categories into 12 core risk pillars.

Synthetic Injection: Addressed class imbalance by generating and injecting 1,500+ synthetic "Safe Text" examples. This trained the model to correctly identify freelance job descriptions as non-risky, reducing false positives from 48% to <10%.

Golden Anchors: Implemented a surgical data injection strategy for edge cases. Manually injected specific adversarial examples (e.g., "Ghost Pay" clauses) into the semantic vector space to force correct classification of previously mislabeled concepts.

_**Infrastructure Optimization**_ -> 

Memory Management: Optimized the PyTorch and Scikit-learn inference pipeline to run strictly on CPU with under 512MB of RAM usage.

Deployment: Containerized the application using Docker (Multi-stage build) for the backend and deployed to Render. Deployed the React frontend to Vercel.

_**KEY RESULTS**_ ->

The optimized Hybrid model achieved production-ready performance metrics on the evaluation set:

Safe Text Accuracy: 91.1% (Successfully filters out standard job descriptions without generating false alarms).

Audit Rights Recall: 96.4% (Demonstrates successful transfer learning of new concepts from the CUAD dataset).

Governing Law Precision: 98.2% (Achieved near-perfect identification of standard legal boilerplate).

Ghost Pay Detection: 100% success rate on adversarial test cases involving "Pay-when-paid" logic.

_**TECH STACK**_ ->

Language: Python 3.10, JavaScript (ES6+)

Backend: FastAPI, Uvicorn

Machine Learning: Scikit-Learn (SVM), Sentence-Transformers (BERT), PyTorch (CPU), K-Means Clustering

Data Processing: Pandas, NumPy, PyPDF (PDF Text Extraction)

Frontend: React, Vite, Axios, CSS Modules

_**HOW TO RUN**_ ->

Backend (API)

Install dependencies: pip install -r requirements.txt

Start the server: uvicorn src.api.main:app --reload

Frontend (Client)

Navigate to the frontend directory: cd frontend

Install packages: npm install

Start the development server: npm run dev

Usage Send a POST request to /analyze with raw text, or use the /upload endpoint to process a PDF contract. The API returns a risk category, confidence score, and a plain-English insight.
