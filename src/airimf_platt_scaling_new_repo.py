# airimf_platt_scaling_new_repo.py
import pandas as pd
import numpy as np
import joblib
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import matplotlib.pyplot as plt

# ===== CONFIGURATION =====
NEW_REPO_CSV = "angular_issues.csv"      # raw GitHub issues CSV
CALIB_FRAC = 0.10
RANDOM_STATE = 42

# 1. Load trained model
print("Loading AIRIMF model (trained on VS Code)...")
model = joblib.load('../airimf_rf_model.joblib')  # adjust path if needed

# 2. Load raw issues CSV
print(f"Loading {NEW_REPO_CSV}...")
df = pd.read_csv(NEW_REPO_CSV)

# 3. Feature engineering (same as AIRIMF pipeline)
print("Engineering features...")
df['Title_Length'] = df['Title'].astype(str).apply(len)
# Body_Length – if not present, compute from 'Body' column
if 'Body_Length' in df.columns:
    df['Body_Length'] = pd.to_numeric(df['Body_Length'], errors='coerce').fillna(0)
elif 'Body' in df.columns:
    df['Body_Length'] = df['Body'].astype(str).apply(len)
else:
    df['Body_Length'] = 0
df['Comments'] = pd.to_numeric(df['Comments'], errors='coerce').fillna(0)

# Sentiment from Title (fast)
def get_polarity(text):
    return TextBlob(str(text)).sentiment.polarity
def get_subjectivity(text):
    return TextBlob(str(text)).sentiment.subjectivity
df['Sentiment_Polarity'] = df['Title'].astype(str).apply(get_polarity)
df['Sentiment_Subjectivity'] = df['Title'].astype(str).apply(get_subjectivity)

# Target: Resolution_Class (1 = Nominal ≤14 days, 2 = Challenging >14 days)
if 'Fixing_Duration_Days' not in df.columns and 'Created_At' in df.columns and 'Closed_At' in df.columns:
    df['Created_At'] = pd.to_datetime(df['Created_At'])
    df['Closed_At'] = pd.to_datetime(df['Closed_At'])
    df['Fixing_Duration_Days'] = (df['Closed_At'] - df['Created_At']).dt.total_seconds() / (24*3600)
if 'Fixing_Duration_Days' in df.columns:
    df['Resolution_Class'] = np.where(df['Fixing_Duration_Days'] <= 14, 1, 2)
else:
    print("❌ Cannot determine target. Need 'Fixing_Duration_Days' or date columns.")
    exit()

# Convert target to binary (1 = Challenging)
y_raw = (df['Resolution_Class'] == 2).astype(int)
feature_cols = ['Title_Length', 'Body_Length', 'Comments', 'Sentiment_Polarity', 'Sentiment_Subjectivity']
X = df[feature_cols]

# 4. Split into calibration + test
X_cal, X_test, y_cal, y_test = train_test_split(
    X, y_raw, test_size=1-CALIB_FRAC, random_state=RANDOM_STATE, stratify=y_raw
)
print(f"Calibration set: {len(X_cal)} | Test set: {len(X_test)}")

# 5. Raw predictions on test set
raw_probs_test = model.predict_proba(X_test)[:, 1]
raw_pred_test = (raw_probs_test >= 0.5).astype(int)
raw_acc = accuracy_score(y_test, raw_pred_test)
raw_brier = brier_score_loss(y_test, raw_probs_test)

# 6. Platt scaling
raw_probs_cal = model.predict_proba(X_cal)[:, 1].reshape(-1, 1)
platt = LogisticRegression(C=1e10, solver='lbfgs')
platt.fit(raw_probs_cal, y_cal)

# 7. Calibrated predictions
cal_probs_test = platt.predict_proba(raw_probs_test.reshape(-1, 1))[:, 1]
cal_pred_test = (cal_probs_test >= 0.5).astype(int)
cal_acc = accuracy_score(y_test, cal_pred_test)
cal_brier = brier_score_loss(y_test, cal_probs_test)

# 8. Expected Calibration Error
def compute_ece(y_true, y_proba, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_proba >= bin_boundaries[i]) & (y_proba < bin_boundaries[i+1])
        if np.sum(in_bin) > 0:
            acc_bin = np.mean(y_true[in_bin])
            conf_bin = np.mean(y_proba[in_bin])
            ece += np.abs(acc_bin - conf_bin) * np.sum(in_bin) / len(y_true)
    return ece

raw_ece = compute_ece(y_test, raw_probs_test)
cal_ece = compute_ece(y_test, cal_probs_test)

print("\n" + "="*50)
print("Results on new repository")
print("="*50)
print(f"Raw model    – Acc: {raw_acc:.4f}, Brier: {raw_brier:.4f}, ECE: {raw_ece:.4f}")
print(f"Calibrated   – Acc: {cal_acc:.4f}, Brier: {cal_brier:.4f}, ECE: {cal_ece:.4f}")

# 9. Reliability diagram
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
frac_pos, mean_pred = calibration_curve(y_test, raw_probs_test, n_bins=10)
plt.plot(mean_pred, frac_pos, 'o-', label='Raw')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Raw Model')
plt.legend()

plt.subplot(1,2,2)
frac_pos_cal, mean_pred_cal = calibration_curve(y_test, cal_probs_test, n_bins=10)
plt.plot(mean_pred_cal, frac_pos_cal, 'o-', label='Platt scaled', color='green')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('After Platt Scaling')
plt.legend()
plt.tight_layout()
plt.savefig('platt_scaling_angular.png', dpi=150)
print("\n✅ Reliability diagram saved as platt_scaling_angular.png")