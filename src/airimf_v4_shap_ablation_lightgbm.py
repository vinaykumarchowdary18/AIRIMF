import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 AIRIMF V4 – SHAP, ABLATION & LIGHTGBM ENHANCEMENTS 🚀")
print("="*60)

# ------------------------------------------------------------
# Load and prepare VS Code data (same as V3)
# ------------------------------------------------------------
df_raw = pd.read_csv('vscode_bugs_datasets.csv')
df_raw['Created_At'] = pd.to_datetime(df_raw['Created_At'])
df_raw['Closed_At'] = pd.to_datetime(df_raw['Closed_At'])
df_raw['Fixing_Duration_Days'] = (df_raw['Closed_At'] - df_raw['Created_At']).dt.total_seconds() / (24*3600)
df_raw['Resolution_Class'] = np.where(df_raw['Fixing_Duration_Days'] <= 14, 1, 2)

df_raw['Title_Length'] = df_raw['Title'].astype(str).apply(len)
df_raw['Body_Length'] = pd.to_numeric(df_raw['Body_Length'], errors='coerce').fillna(0)
df_raw['Comments'] = pd.to_numeric(df_raw['Comments'], errors='coerce').fillna(0)

def get_polarity(text):
    return TextBlob(str(text)).sentiment.polarity
def get_subjectivity(text):
    return TextBlob(str(text)).sentiment.subjectivity

df_raw['Sentiment_Polarity'] = df_raw['Title'].astype(str).apply(get_polarity)
df_raw['Sentiment_Subjectivity'] = df_raw['Title'].astype(str).apply(get_subjectivity)

# ------------------------------------------------------------
# 1. Full Hybrid Model + SHAP (Your current champion)
# ------------------------------------------------------------
print("\n[1] Training Full Hybrid Model (All 5 features) with SHAP...")
X_full = df_raw[['Title_Length', 'Body_Length', 'Comments', 'Sentiment_Polarity', 'Sentiment_Subjectivity']]
y = df_raw['Resolution_Class']

smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_full, y)
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_full = RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=5, random_state=42)
rf_full.fit(X_train_scaled, y_train)
y_pred_full = rf_full.predict(X_test_scaled)
probs_full = rf_full.predict_proba(X_test_scaled)[:,1]

print(f"Full Model Accuracy: {accuracy_score(y_test, y_pred_full)*100:.2f}%")
print(f"ROC-AUC: {roc_auc_score(y_test, probs_full):.4f}")
print(classification_report(y_test, y_pred_full))

# SHAP Explanation – use probability Explainer to avoid all shape/additivity bugs
print("Generating SHAP summary plot (may take ~30 seconds)...")
explainer = shap.Explainer(rf_full.predict_proba, X_train_scaled[:100])
shap_values_all = explainer(X_test_scaled[:200])

# shap_values_all is an Explanation object with shape (200, 5, 2)
# We need the SHAP values for class 2 (index 1) and the feature matrix
shap_values_c2 = shap_values_all[..., 1]   # shape (200, 5)
X_sub_df = pd.DataFrame(X_test_scaled[:200], columns=X_full.columns)

plt.figure()
shap.summary_plot(shap_values_c2, X_sub_df, show=False)
plt.tight_layout()
plt.savefig('IEEE_SHAP_Explanation.png', dpi=300, bbox_inches='tight')
print("✅ SHAP summary plot saved as IEEE_SHAP_Explanation.png\n")
# ------------------------------------------------------------
# 2. Ablation Study
# ------------------------------------------------------------
print("[2] Running Ablation Study...")
ablations = {
    "Metadata Only": ['Title_Length', 'Body_Length', 'Comments'],
    "NLP Only": ['Sentiment_Polarity', 'Sentiment_Subjectivity'],
    "Full Hybrid": ['Title_Length', 'Body_Length', 'Comments', 'Sentiment_Polarity', 'Sentiment_Subjectivity']
}

results = []
for name, feats in ablations.items():
    X_abl = df_raw[feats]
    Xb, yb = smote.fit_resample(X_abl, y)
    Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.2, random_state=42)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    rf_abl = RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=5, random_state=42)
    rf_abl.fit(Xtr_s, ytr)
    ypred = rf_abl.predict(Xte_s)
    acc = accuracy_score(yte, ypred)
    # cross‑validation for stability
    cv_scores = cross_val_score(rf_abl, sc.fit_transform(Xb), yb, cv=StratifiedKFold(5, shuffle=True, random_state=42))
    results.append({
        "Experiment": name,
        "Accuracy (%)": round(acc*100, 2),
        "CV Mean (%)": round(cv_scores.mean()*100, 2),
        "CV Std (%)": round(cv_scores.std()*100, 2)
    })

ablation_df = pd.DataFrame(results)
print("\n--- Ablation Study Results ---")
print(ablation_df.to_string(index=False))
ablation_df.to_csv('ablation_study_results.csv', index=False)
print("✅ Ablation study saved to ablation_study_results.csv\n")

# ------------------------------------------------------------
# 3. LightGBM Baseline
# ------------------------------------------------------------
print("[3] Training LightGBM Classifier...")
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
lgb_model.fit(X_train_scaled, y_train)
lgb_pred = lgb_model.predict(X_test_scaled)
lgb_acc = accuracy_score(y_test, lgb_pred)
lgb_probs = lgb_model.predict_proba(X_test_scaled)[:,1]
lgb_auc = roc_auc_score(y_test, lgb_probs)
print(f"LightGBM Accuracy: {lgb_acc*100:.2f}%")
print(f"LightGBM ROC-AUC: {lgb_auc:.4f}")
print(classification_report(y_test, lgb_pred))

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------
print("\n" + "="*60)
print("🎉 All enhancements completed successfully!")
print("Generated files: IEEE_SHAP_Explanation.png, ablation_study_results.csv")
print("="*60)