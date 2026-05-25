import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("🚀 INITIALIZING AIRIMF V2 MASTER PIPELINE 🚀")
print("="*50)

# ---------------------------------------------------------
# PHASE 1: FEATURE ENGINEERING (Early-Stage Signals Only)
# ---------------------------------------------------------
print("\n[1/4] Engineering Early-Stage Features...")
try:
    df_raw = pd.read_csv('vscode_bugs_datasets.csv')
except FileNotFoundError:
    print("Error: 'vscode_bugs_datasets.csv' not found. Check your directory.")
    exit()

df_raw['Created_At'] = pd.to_datetime(df_raw['Created_At'])
df_raw['Closed_At'] = pd.to_datetime(df_raw['Closed_At'])
df_raw['Fixing_Duration_Days'] = (df_raw['Closed_At'] - df_raw['Created_At']).dt.total_seconds() / (24 * 3600)

# Define our Target Variable (Strictly hidden from the AI during training)
df_raw['Resolution_Class'] = np.where(df_raw['Fixing_Duration_Days'] <= 14, 1, 2)

# Engineer new early-stage features to satisfy IEEE reviewers
df_raw['Title_Length'] = df_raw['Title'].astype(str).apply(len)
df_raw['Body_Length'] = pd.to_numeric(df_raw['Body_Length'], errors='coerce').fillna(0)
df_raw['Comments'] = pd.to_numeric(df_raw['Comments'], errors='coerce').fillna(0)

# Select strictly early-stage features to prevent Target Leakage
X_raw = df_raw[['Title_Length', 'Body_Length', 'Comments']]
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
print("[3/4] Scaling Features and Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# PHASE 4: TRAINING & BENCHMARKING
# ---------------------------------------------------------
print("[4/4] Training Baselines and Champion Models...\n")

# Baseline 1: ZeroR (Predicts the majority class)
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train_scaled, y_train)
dummy_acc = accuracy_score(y_test, dummy_model.predict(X_test_scaled))

# Baseline 2: Logistic Regression (Standard statistical model)
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_acc = accuracy_score(y_test, lr_model.predict(X_test_scaled))

# Challenger: Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_acc = accuracy_score(y_test, svm_model.predict(X_test_scaled))

# Champion: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_predictions)

# ---------------------------------------------------------
# FINAL OUTPUT
# ---------------------------------------------------------
print("="*50)
print("📊 IEEE EMPIRICAL BENCHMARK RESULTS 📊")
print("="*50)
print(f"1. ZeroR Baseline (Dumb Guess):       {dummy_acc * 100:.2f}%")
print(f"2. Logistic Regression Baseline:      {lr_acc * 100:.2f}%")
print(f"3. Support Vector Machine (SVM):      {svm_acc * 100:.2f}%")
print(f"4. Random Forest (Proposed Model):    {rf_acc * 100:.2f}%\n")

print("--- Random Forest Detailed Metrics ---")
print(classification_report(y_test, rf_predictions))

# Feature Importance
print("--- Early-Stage Feature Importance ---")
importances = rf_model.feature_importances_
for feature, imp in zip(X_raw.columns, importances):
    print(f"- {feature}: {imp*100:.2f}%")