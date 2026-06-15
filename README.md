<div align="center">

# 🚀 AIRIMF
### AI-Driven Risk Identification and Mitigation Framework for Software Project Management

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20691021.svg)](https://doi.org/10.5281/zenodo.20691021)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![Paper](https://img.shields.io/badge/Preprint-Zenodo-blue)](https://doi.org/10.5281/zenodo.20691021)

**Submitted to ICMACC 2026 (IEEE) · Preprint available on Zenodo**

*Mandadi Vinay Kumar — Department of Computer Science and Engineering, Lovely Professional University*

</div>

---

## 📖 Overview

Most software projects fail not from a lack of talent or budget, but because risk signals appear **weeks before anyone acts on them**. AIRIMF shifts project management from reactive firefighting to **proactive triage**.

It predicts whether a GitHub issue will become **"Challenging"** (requiring more than 14 days to resolve) at the **exact moment of creation** — before any development work begins, before any engineer is assigned, and before delays begin to compound.

By strictly eliminating target leakage and relying entirely on 5 early-stage features, it provides actionable risk scores without artificially inflating accuracy. A project manager could realistically deploy this system today: every new issue receives a risk score within seconds of creation.

> **"The gap between 'issue created' and 'risk understood' collapses to near zero."**

---

## ✨ Key Capabilities

| Capability | Description |
|:---|:---|
| 🔒 **Leakage-Free Architecture** | Uses only 5 pre-resolution features: Title Length, Body Length, Initial Comment Count, Sentiment Polarity, Sentiment Subjectivity — all verified to exist at issue creation time |
| ⚖️ **Class Imbalance Handling** | SMOTE algorithm synthetically balances training data, boosting high-risk recall from <50% to 82% |
| 🧠 **XAI Explanations** | SHAP integration provides human-readable feature impact explanations for every flagged issue |
| ⚡ **Live Webhook Integration** | FastAPI backend listens for GitHub events and acts as an automated triage bot in real time |
| 🌍 **Cross-Project Generalizability** | Trained on `microsoft/vscode`, validated on `facebook/react` (+8 pp accuracy improvement), calibrated on `angular/angular` via Platt scaling |
| 📊 **Statistical Rigor** | McNemar's test confirms superiority over SVM (χ²=30.25, p<0.0001); 5-fold cross-validation throughout |

---

## 📊 Empirical Performance

Champion model: **Random Forest** (150 estimators, max depth 15, min-samples-split 5)

### In-Domain Performance

| Metric | Microsoft / VS Code | Facebook / React |
|:---|:---:|:---:|
| **Holdout Accuracy** | 76.42% | 84.70% |
| **High-Risk (Class 2) Recall** | 82% | 89% |
| **ROC-AUC Score** | 0.8436 | 0.9087 |
| **5-Fold CV Accuracy** | 76.21% ± 0.88% | — |

### Cross-Project Calibration (Angular/Angular)

| Metric | Raw Model | After Platt Scaling |
|:---|:---:|:---:|
| **Accuracy** | 39.56% | 57.14% |
| **Expected Calibration Error** | 0.2521 | 0.0501 |
| **Brier Score** | 0.3302 | 0.2283 |

### Statistical Highlights

- 📈 **McNemar's test** vs SVM baseline: χ²(1) = 30.25, **p < 0.0001** — improvement is not a statistical artefact
- 🔍 **SHAP analysis**: `Body_Length` is the dominant risk signal at 36.4% feature importance — long descriptions signal complexity
- ⚗️ **Ablation study**: Adding NLP sentiment features boosted accuracy by **+6.17 pp** over structural metadata alone
- 🎯 **Confusion matrix**: Roughly symmetric errors (121 FP vs 70 FN) — model does not sacrifice precision for recall

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AIRIMF Pipeline                              │
├──────────────┬──────────────────┬──────────────┬────────────────┤
│   Layer 1    │     Layer 2      │   Layer 3    │    Layer 4     │
│ Data         │ Feature          │ Predictive   │ Action &       │
│ Ingestion    │ Engineering      │ Engine       │ Mitigation     │
├──────────────┼──────────────────┼──────────────┼────────────────┤
│ GitHub REST  │ VADER/TextBlob   │ Random       │ FastAPI        │
│ API &        │ NLP Sentiment    │ Forest       │ Webhook        │
│ Webhooks     │ + SMOTE          │ Classifier   │ Server +       │
│              │ + StandardScaler │ + SHAP XAI   │ GitHub Bot     │
└──────────────┴──────────────────┴──────────────┴────────────────┘
```

### Feature Vector at Prediction Time

All 5 features are verified to be available at GitHub issue creation time:

```python
features = {
    "title_length":          len(issue.title),           # Character count of title
    "body_length":           len(issue.body),            # Character count of description  
    "comments":              issue.comments_first_hour,  # Engagement in first hour
    "sentiment_polarity":    TextBlob(issue.title).sentiment.polarity,     # [-1, +1]
    "sentiment_subjectivity":TextBlob(issue.title).sentiment.subjectivity, # [0, 1]
}
```

---

## 📁 Repository Structure

```
AIRIMF/
├── 📄 airimf_v3_advanced_pipeline.py    # Main training pipeline — leakage-free RF model
├── 📄 airimf_v4_shap_ablation_lightgbm.py  # SHAP plots + ablation study + LightGBM comparison
├── 📄 step2_local_inference.py          # Local inference test on simulated issues
├── 📄 step4_github_bot.py              # FastAPI webhook server — live DevSecOps deployment
├── 📄 requirements.txt                 # All dependencies
├── 📊 results/                         # Confusion matrices, ROC curves, SHAP plots
│   ├── confusion_matrix_vscode.png
│   ├── shap_beeswarm_vscode.png
│   ├── calibration_angular.png
│   └── roc_curves.png
└── 📖 README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/vinaykumarchowdary18/AIRIMF.git
cd AIRIMF
pip install -r requirements.txt
```

**Requirements:**
```bash
pip install pandas scikit-learn imbalanced-learn textblob shap joblib fastapi uvicorn requests PyGithub
```

---

## 🚀 Usage

### Phase 1 — Training & Model Export

Train the champion Random Forest model, compute NLP features, export `.joblib` files:

```bash
python airimf_v3_advanced_pipeline.py
```

Optionally, generate SHAP visual plots and ablation tables:

```bash
python airimf_v4_shap_ablation_lightgbm.py
```

**Output:** `rf_model.joblib`, `scaler.joblib`, confusion matrices, SHAP plots

---

### Phase 2 — Local Inference Test

Verify the decoupled model can score a simulated issue:

```bash
python step2_local_inference.py
```

**Expected output:**
```
Issue: "App crashes on startup after latest update"
Risk Score: 0.73
Prediction: 🔴 HIGH RISK (Challenging)
SHAP Explanation: Body_Length (+0.31), Sentiment_Polarity (-0.12), ...
```

---

### Phase 3 — Live DevSecOps Deployment

**Step 1:** Launch the FastAPI inference server:
```bash
python step4_github_bot.py
```

**Step 2:** Expose to GitHub via public tunnel:
```bash
ngrok http 8000
```

**Step 3:** Add webhook in your repository:
```
GitHub Repo → Settings → Webhooks → Add webhook
Payload URL: https://[your-ngrok-id].ngrok-free.app/webhook
Content type: application/json
Events: Issues
```

**Step 4:** Create a new issue in your repository. Within seconds, the bot posts:
```
🔴 AIRIMF Risk Assessment: HIGH RISK
Risk Score: 0.78 | Predicted resolution: >14 days

Top Risk Signals:
• Body_Length: +0.31 (long description → complex issue)
• Sentiment_Polarity: -0.18 (negative tone → urgent/critical)
• Title_Length: +0.09

Recommendation: Assign senior engineer. Flag for sprint planning.
```

---

## 🔬 Reproducing Paper Results

```bash
# 1. Collect data from GitHub API (requires token)
export GITHUB_TOKEN=your_token_here
python airimf_v3_advanced_pipeline.py --collect-data --repos microsoft/vscode facebook/react

# 2. Train with exact paper hyperparameters
python airimf_v3_advanced_pipeline.py --n-estimators 150 --max-depth 15 --min-samples-split 5

# 3. Generate all paper figures (SHAP, ROC, calibration)
python airimf_v4_shap_ablation_lightgbm.py --generate-all-figures

# 4. Run cross-project validation on Angular
python airimf_v3_advanced_pipeline.py --cross-project angular/angular --platt-scaling
```

Expected results match Table I–V in the paper (±0.5% due to random seed variation).

---

## 📈 Key Findings

**1. Description length is the strongest early risk signal**
When a developer writes a long, detailed description at submission time, they're implicitly signalling the problem is complex. Vague short reports tend to be simpler bugs.

**2. Sentiment features add meaningful signal (+6.17 pp)**
High subjectivity and negative polarity consistently push predictions toward Class 2 (Challenging). Sentiment alone isn't enough — but layered on structural metadata, it matters.

**3. Cross-project transfer is possible, but culture-dependent**
React (similar culture to VS Code): 84.70% without retraining.
Angular (different triage culture): 39.56% raw → 57.14% after Platt scaling on 100 local samples.

**4. The leakage lesson**
Our initial model hit 100% accuracy. A single feature — resolution time — had leaked into training. After removing it, accuracy dropped substantially. What remained was a model that actually learned something genuine about software risk.

---

## 🗺️ Roadmap

- [ ] **Layer 4 Live Deployment** — SLA_RISK labelling + CAB dashboard notification system
- [ ] **Richer features** — full-body sentiment, reporter experience, cross-reference counts
- [ ] **Non-JavaScript ecosystems** — Python, Go, Rust, Java repositories
- [ ] **BERT embeddings** — replace bag-of-words sentiment with contextual embeddings
- [ ] **Per-issue SHAP force plots** — inline explanations in issue creation workflow
- [ ] **Temporal drift monitoring** — recalibration trigger when incoming distribution shifts

---

## 📚 Citation

If you use AIRIMF in your research or production environment, please cite:

```bibtex
@misc{mandadi2026airimf,
  author       = {Mandadi, Vinay Kumar},
  title        = {AIRIMF: An AI-Driven Risk Identification and Mitigation Framework 
                  for Software Project Management},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20691021},
  url          = {https://doi.org/10.5281/zenodo.20691021}
}
```

**Preprint:** [https://doi.org/10.5281/zenodo.20691021](https://doi.org/10.5281/zenodo.20691021)

---

## 👤 Author

**Mandadi Vinay Kumar**
B.Tech Computer Science and Engineering, Lovely Professional University
- 📧 mvkchowdary20@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/vinay-kumar-chowdary)
- 🐙 [GitHub](https://github.com/vinaykumarchowdary18)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**If AIRIMF helped your research or project, please ⭐ star this repository**

*Submitted to ICMACC 2026 (IEEE) · Preprint: [doi.org/10.5281/zenodo.20691021](https://doi.org/10.5281/zenodo.20691021)*

</div>
