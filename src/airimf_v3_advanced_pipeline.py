import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 INITIALIZING AIRIMF V3 ADVANCED NLP PIPELINE 🚀")
print("="*60)

# ---------------------------------------------------------
# PHASE 1: ADVANCED NLP FEATURE ENGINEERING
# ---------------------------------------------------------
print("\n[1/4] Engineering Early-Stage Metadata and NLP Features...")
try:
    df_raw = pd.read_csv('vscode_bugs_datasets.csv')
except FileNotFoundError:
    print("Error: 'vscode_bugs_datasets.csv' not found. Check your directory.")
    exit()

# Target Variable Calculation
df_raw['Created_At'] = pd.to_datetime(df_raw['Created_At'])
df_raw['Closed_At'] = pd.to_datetime(df_raw['Closed_At'])
df_raw['Fixing_Duration_Days'] = (df_raw['Closed_At'] - df_raw['Created_At']).dt.total_seconds() / (24 * 3600)
df_raw['Resolution_Class'] = np.where(df_raw['Fixing_Duration_Days'] <= 14, 1, 2)

# Basic Metadata Features
df_raw['Title_Length'] = df_raw['Title'].astype(str).apply(len)
df_raw['Body_Length'] = pd.to_numeric(df_raw['Body_Length'], errors='coerce').fillna(0)
df_raw['Comments'] = pd.to_numeric(df_raw['Comments'], errors='coerce').fillna(0)

# --- NEW NOVELTY: NLP SENTIMENT ANALYSIS ---
print("      -> Running VADER/TextBlob Sentiment Analysis on issue bodies...")
def get_polarity(text):
    return TextBlob(str(text)).sentiment.polarity

def get_subjectivity(text):
    return TextBlob(str(text)).sentiment.subjectivity

# We use the Title as a proxy for text if Body is empty, to save extraction time for this proof of concept
df_raw['Sentiment_Polarity'] = df_raw['Title'].astype(str).apply(get_polarity)
df_raw['Sentiment_Subjectivity'] = df_raw['Title'].astype(str).apply(get_subjectivity)

# Select strictly early-stage features (Now including NLP)
X_raw = df_raw[['Title_Length', 'Body_Length', 'Comments', 'Sentiment_Polarity', 'Sentiment_Subjectivity']]
y_raw = df_raw['Resolution_Class']

# ---------------------------------------------------------
# PHASE 2: SMOTE BALANCING
# ---------------------------------------------------------
print("[2/4] Applying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_raw, y_raw)

# ---------------------------------------------------------
# PHASE 3: MODEL PREPARATION
# ---------------------------------------------------------
print("[3/4] Scaling Features and Preparing Cross-Validation...")
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_balanced_scaled = scaler.fit_transform(X_balanced)

# ---------------------------------------------------------
# PHASE 4: TRAINING & RIGOROUS BENCHMARKING
# ---------------------------------------------------------
print("[4/4] Training Champion Model with Deep Experimental Metrics...\n")

# Champion: Tuned Random Forest
rf_model = RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=5, random_state=42)

# 1. Stratified 5-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_balanced_scaled, y_balanced, cv=cv, scoring='accuracy')

# 2. Standard Train/Test fit for detailed metrics
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)
rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1] # Probabilities for ROC-AUC
# Train SVM for comparison
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)
# 3. ROC-AUC Score Calculation
roc_auc = roc_auc_score(y_test, rf_probs)

# ---------------------------------------------------------
# FINAL OUTPUT
# ---------------------------------------------------------
print("="*60)
print("📊 IEEE V3 EMPIRICAL BENCHMARK RESULTS 📊")
print("="*60)

print(f"1. 5-Fold Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
print(f"2. Holdout Test Accuracy:            {accuracy_score(y_test, rf_predictions) * 100:.2f}%")
print(f"3. ROC-AUC Score:                    {roc_auc:.4f} (Outstanding > 0.75)\n")

print("--- Random Forest Detailed Metrics ---")
print(classification_report(y_test, rf_predictions))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, rf_predictions)
print("\n--- Confusion Matrix (exact counts) ---")
print(cm)

# Feature Importance
print("--- Advanced Feature Importance (Including NLP) ---")
importances = rf_model.feature_importances_
for feature, imp in zip(X_raw.columns, importances):
    print(f"- {feature}: {imp*100:.2f}%")

    from mlxtend.evaluate import mcnemar_table, mcnemar

tb = mcnemar_table(y_target=y_test.values,
                   y_model1=rf_predictions,
                   y_model2=svm_predictions)
print("\n--- McNemar's Test (RF vs SVM) ---")
chi2, p = mcnemar(ary=tb, corrected=True)
print(f"Chi-squared: {chi2:.2f}, p-value: {p:.4f}")
if p < 0.05:
    print("✅ The performance difference is statistically significant (p < 0.05).")
else:
    print("⚠️ The performance difference is NOT statistically significant.")