import pandas as pd
from imblearn.over_sampling import SMOTE

print("Initializing SMOTE Data Augmentation...")

# 1. Load the engineered data
df = pd.read_csv('AIRIMF_Engineered_Dataset.csv')

# 2. Preprocessing for SMOTE
# SMOTE requires strictly numerical data. We must map our qualitative 'Magnitude' into integers.
magnitude_map = {'Negligible': 0, 'Moderate': 1, 'Extreme': 2}
reverse_map = {0: 'Negligible', 1: 'Moderate', 2: 'Extreme'}
df['Magnitude'] = df['Magnitude'].map(magnitude_map)

# 3. Separate Features (X) and Target (y)
# We drop Issue_ID because it is just a label, not a predictive metric.
X = df.drop(['Issue_ID', 'Resolution_Class'], axis=1)
y = df['Resolution_Class']

print(f"Original dataset distribution:\n{y.value_counts()}\n")

# 4. Apply SMOTE using your local hardware
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 5. Reconstruct the balanced DataFrame
df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_balanced['Resolution_Class'] = y_resampled

# Map Magnitude back to textual categories for your paper's taxonomy
df_balanced['Magnitude'] = df_balanced['Magnitude'].map(reverse_map).fillna('Extreme')

# 6. Save the final, publication-ready dataset
save_path = 'AIRIMF_Final_Balanced_Dataset.csv'
df_balanced.to_csv(save_path, index=False)

print(f"SMOTE Complete! Synthetically expanded dataset to {len(df_balanced)} records.")
print(f"New balanced distribution:\n{df_balanced['Resolution_Class'].value_counts()}")