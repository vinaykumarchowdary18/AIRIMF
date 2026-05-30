🚀 AIRIMF: AI-Driven Risk Identification and Mitigation Framework
📖 Overview
AIRIMF is a machine-learning pipeline designed for proactive software project management. It flags high-risk GitHub tickets (issues taking > 14 days to resolve) at the exact moment of creation. By strictly utilizing pre-resolution metadata and NLP sentiment, AIRIMF completely eliminates target leakage while providing project managers with real-time, actionable risk assessments.

✨ Key Features
Strict Early-Stage Prediction: Evaluates risk using only Title Length, Body Length, Initial Comment Count, Sentiment Polarity, and Sentiment Subjectivity.

Leakage-Free Architecture: Ensures no features derived from the actual resolution time are included in the predictive model.

Imbalance Handling: Utilizes SMOTE to generate balanced training sets, significantly improving the recall of minority-class (challenging) bugs.

XAI (Explainable AI): Integrates SHAP to provide real-time feature importance explanations for every flagged issue.

Cross-Ecosystem Generalizability: Validated on major open-source repositories including microsoft/vscode and facebook/react.

🏗️ System Architecture
The framework operates on a four-layer DevSecOps architecture:

Layer 1 (Data Ingestion Layer): Collects real-time or historical bug reports via GitHub REST API / WebHooks.

Layer 2 (Feature Engineering Layer): Processes NLP sentiment and applies SMOTE for class balancing.

Layer 3 (Predictive Layer): A trained Random Forest classifier evaluates the feature vector to generate a risk probability.

Layer 4 (Action & Mitigation Layer): A FastAPI-driven webhook that posts risk assessments directly to GitHub issues and project management dashboards.

📊 Empirical Results
The champion Random Forest model has been rigorously evaluated under leakage-free conditions using 5-fold cross-validation and McNemar's test.

VS Code Benchmark
Accuracy: 76.21% (Cross-Validation) / 76.42% (Holdout)

High-Risk Recall: 82%

ROC-AUC: 0.8436

React (Cross-Project Validation)
Accuracy: 83.37% (Cross-Validation)

High-Risk Recall: 89%

ROC-AUC: 0.9087

Note: Ablation studies demonstrate that adding NLP sentiment features boosts accuracy by over 6 percentage points compared to structural metadata alone. SHAP analysis confirms Body_Length is the dominant risk signal at creation time (accounting for 36.4% of predictive importance).

⚙️ Installation & Setup
Ensure you have Python installed. Install the required pipeline and backend dependencies:

Bash
pip install pandas scikit-learn imbalanced-learn textblob shap joblib fastapi uvicorn
🚀 Usage
1. Model Export
To train and export the champion model and SHAP explainer from the advanced pipeline:

Bash
python airimf_v3_advanced_pipeline.py
2. Local Inference Test
To test the decoupled engine on a simulated new issue to view real-time classifications and SHAP explanations:

Bash
python step2_local_inference.py
3. Live FastAPI Webhook
To launch the live DevSecOps inference server that listens for GitHub webhook payloads:

Bash
python step3_fastapi_webhook.py
(To expose this local server to GitHub, authenticate with ngrok, run ngrok http 8000, and paste the generated public URL into your repository's webhook settings).

📝 Citation
If you use AIRIMF in your research, please cite the foundational framework:

Kumar, M. V. (2026). AIRIMF: An AI-Driven Risk Identification and Mitigation Framework for Software Project Management. Department of Computer Science and Engineering, Lovely Professional University.
