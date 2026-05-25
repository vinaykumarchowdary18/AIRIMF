import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("Initializing AIRIMF Predictive Modeling Pipeline...\n")

# 1. Load the balanced dataset
df = pd.read_csv('AIRIMF_Final_Balanced_Dataset.csv')

# 2. Prepare Features and Target
# We must encode the textual 'Magnitude' back to numbers for the models
magnitude_map = {'Negligible': 0, 'Moderate': 1, 'Extreme': 2}
df['Magnitude'] = df['Magnitude'].map(magnitude_map)

X = df[['Priority_Score', 'Magnitude']]   # Only early-stage signals
y = df['Resolution_Class']

# 3. Split the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling (Crucial for SVM performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- MODEL 1: Random Forest (The upgraded Decision Tree) ---
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_predictions)

# --- MODEL 2: Support Vector Machine (SVM) ---
print("Training Support Vector Machine (SVM)...")
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, svm_predictions)

# --- RESULTS OUTPUT ---
print("\n" + "="*40)
print("🏆 AIRIMF MODEL BENCHMARK RESULTS 🏆")
print("="*40)

print(f"\n1. Random Forest Accuracy: {rf_acc * 100:.2f}%")
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

print(f"\n2. SVM Accuracy: {svm_acc * 100:.2f}%")
print("SVM Classification Report:")
print(classification_report(y_test, svm_predictions))