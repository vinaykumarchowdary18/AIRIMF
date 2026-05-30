# 🚀 AIRIMF: AI-Driven Risk Identification and Mitigation Framework

## 📖 Overview
AIRIMF is a machine-learning pipeline designed to shift software project management from reactive firefighting to proactive triage. It predicts whether a GitHub issue will become "Challenging" (taking more than 14 days to resolve) at the exact moment of creation. 

By strictly eliminating target leakage and relying entirely on early-stage metadata and NLP sentiment, it provides actionable risk scores without artificially inflating accuracy.

## ✨ Key Capabilities
* **Leakage-Free Architecture:** Evaluates risk using only 5 pre-resolution features: Title Length, Body Length, Initial Comment Count, Sentiment Polarity, and Sentiment Subjectivity.
* **Class Imbalance Handling:** Uses the SMOTE algorithm to synthetically balance datasets, preventing the model from just guessing the majority class.
* **XAI Explanations:** Integrates SHAP to provide real-time, human-readable feature impact explanations for every single flagged issue.
* **Live Webhook Integration:** Features a FastAPI backend that listens for live GitHub events and acts as an automated triage bot.
* **Cross-Project Generalizability:** Trained on `microsoft/vscode` and successfully validated on `facebook/react`.

---

## 🏗️ System Architecture
The framework operates on a four-layer DevSecOps architecture:

| Layer | Component | Description |
| :--- | :--- | :--- |
| **1. Data Ingestion** | GitHub REST API & Webhooks | Scrapes historical issues or listens for live JSON payloads. |
| **2. Feature Engineering** | VADER/TextBlob & SMOTE | Extracts NLP sentiment, calculates metadata, and balances classes. |
| **3. Predictive Engine** | Random Forest | Evaluates features to output a risk probability and binary classification. |
| **4. Action & Mitigation** | FastAPI & GitHub Bot | Posts the SHAP explanation and risk score back to the live issue. |

---

## 📊 Empirical Performance
The champion Random Forest model (150 estimators, max depth 15, min-samples-split 5) was rigorously evaluated using 5-fold cross-validation.

| Metric | Microsoft/VS Code | Facebook/React |
| :--- | :--- | :--- |
| **Holdout Accuracy** | 76.42% | 84.70% |
| **High-Risk Recall** | 82.00% | 89.00% |
| **ROC-AUC Score** | 0.8436 | 0.9087 |

**Statistical Highlights:**
* The model's superiority over standard SVM baselines was validated using McNemar's test, yielding a chi-squared of 30.25 and a p-value of p < 0.0001. 
* Ablation studies proved that adding NLP sentiment features boosted accuracy by 6.17 percentage points over structural metadata alone. 
* SHAP analysis confirms `Body_Length` is the dominant risk signal at creation time (36.4% importance).

---

## ⚙️ Installation & Setup
Install the required pipeline and backend dependencies:

```bash
pip install pandas scikit-learn imbalanced-learn textblob shap joblib fastapi uvicorn
🚀 Pipeline Execution
Phase 1: Training & Model Export
Train the champion Random Forest model, calculate NLP features, and export the .joblib files required for the backend server:

Bash
python airimf_v3_advanced_pipeline.py
(Optionally, run python airimf_v4_shap_ablation_lightgbm.py to generate the SHAP visual plots and ablation tables).

Phase 2: Live DevSecOps Deployment
Local Inference Test: Verify the decoupled model can predict risk on a simulated issue.

Bash
python step2_local_inference.py
Launch Webhook Server: Start the live FastAPI inference server.

Bash
python step4_github_bot.py
Expose to GitHub: Create a public tunnel to your local server so GitHub can send webhooks.

Bash
ngrok http 8000
(Copy the generated https://...ngrok-free.app URL and paste it into your GitHub Repository -> Settings -> Webhooks).

📝 Citation
If you utilize AIRIMF in your research or production environment, please cite the foundational framework:

Kumar, M. V. (2026). AIRIMF: An AI-Driven Risk Identification and Mitigation Framework for Software Project Management. Department of Computer Science and Engineering, Lovely Professional University.
