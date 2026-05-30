
# AIRIMF: AI‑Driven Risk Identification and Mitigation Framework

Predicts whether a GitHub issue will become "Challenging" (resolution time >14 days) using only early‑stage features available at ticket creation time.

## Key Results

- Random Forest accuracy: **76.4%** (VS Code) / **84.7%** (React)
- Challenging‑class recall: **82%** (VS Code) / **89%** (React)
- ROC‑AUC: **0.8436** (VS Code)

## Features (no leakage, only creation‑time data)

- `Title_Length`, `Body_Length`, `Comments` (initial count)
- `Sentiment_Polarity`, `Sentiment_Subjectivity` (from title)

## Requirements

```bash
pip install pandas numpy scikit-learn imbalanced-learn textblob shap joblib
How to Run
Prepare your raw GitHub issues CSV – must contain Title, Body_Length, Comments, Created_At, Closed_At.

Run the full pipeline

bash
python airimf_v3_advanced_pipeline.py
This trains the model, prints classification report, confusion matrix, feature importance, and performs McNemar's test.

Export the champion model (done automatically by the script) – you get airimf_rf_model.joblib and airimf_shap_explainer.joblib.

Test local inference

bash
python infer_risk.py
Enter a title, description, and initial comment count to get risk probability + SHAP explanations.

Repository Structure
text
AIRIMF/
├── airimf_v3_advanced_pipeline.py   # main training & evaluation
├── infer_risk.py                    # local inference example
├── airimf_rf_model.joblib           # trained Random Forest
├── airimf_shap_explainer.joblib     # SHAP TreeExplainer
├── requirements.txt
└── README.md
Citation
If you use this code, cite the AIRIMF paper (available on arXiv/your publication).
