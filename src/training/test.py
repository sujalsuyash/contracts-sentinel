import joblib
import os
import re

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def test_the_brain():
    print("Loading ...")
    clf = joblib.load(os.path.join(MODELS_DIR, 'classifier.pkl'))
    vec = joblib.load(os.path.join(MODELS_DIR, 'vectorizer.pkl'))

    # Test cases that FAILED in the report (Audit, Non-Compete, Insurance)
    test_cases = [
        ("Audit Rights", "Client shall have the right to inspect and audit Service Provider's books and records once per year."),
        ("Non-Compete", "Employee agrees not to work for any direct competitor for a period of two years after termination."),
        ("Insurance", "Contractor must maintain General Liability Insurance with coverage of at least $1,000,000."),
        ("Governing Law", "This agreement shall be governed by the laws of the State of New York."),
        ("Safe Text", "The graphic designer shall deliver the logo in PNG format by Friday.")
    ]

    print("\n--- LIVE PREDICTION TEST ---")
    print(f"{'EXPECTED':<20} | {'PREDICTED':<20} | {'CONFIDENCE':<10}")
    print("-" * 60)

    for label, text in test_cases:
        # Prepare
        clean_input = clean_text(text)
        vectorized_input = vec.transform([clean_input])
        
        # Predict
        prediction = clf.predict(vectorized_input)[0]
        probs = clf.predict_proba(vectorized_input)
        confidence = max(probs[0])

        print(f"{label:<20} | {prediction:<20} | {confidence:.1%}")

if __name__ == "__main__":
    test_the_brain()