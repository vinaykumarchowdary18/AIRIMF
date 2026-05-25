import pandas as pd
import numpy as np

print("Initializing AIRIMF Data Engineering Pipeline...")

# 1. Load the raw data (Using the exact filename you provided)
try:
    df = pd.read_csv('vscode_bugs_datasets.csv')
except FileNotFoundError:
    print("Error: Could not find the CSV. Please ensure the filename matches exactly.")
    exit()

# 2. Date Processing: Calculate 'Fixing Duration'
# We convert the raw ISO timestamps into workable datetime objects
df['Created_At'] = pd.to_datetime(df['Created_At'])
df['Closed_At'] = pd.to_datetime(df['Closed_At'])

# Calculate the exact time it took to resolve the bug in days
df['Fixing_Duration_Days'] = (df['Closed_At'] - df['Created_At']).dt.total_seconds() / (24 * 3600)
df['Fixing_Duration_Days'] = df['Fixing_Duration_Days'].round(2)

# 3. Feature Engineering: Complexity and Risk Metrics
# We use 'Comments' and 'Body_Length' as proxies for project friction

# Calculate Priority Score (0-100 scale, as defined in your paper taxonomy)
# Heuristic: More comments and a longer issue description = higher complexity/priority
df['Priority_Score'] = (df['Comments'] * 5) + (df['Body_Length'] / 100)
df['Priority_Score'] = df['Priority_Score'].clip(upper=100).round(0)

# Calculate Risk Magnitude (Qualitative Assessment matching the paper)
# If a bug takes over 30 days and has many comments, it is 'Extreme'
conditions = [
    (df['Fixing_Duration_Days'] > 30) & (df['Comments'] > 10),
    (df['Fixing_Duration_Days'] > 7) & (df['Comments'] > 3),
]
choices = ['Extreme', 'Moderate']
df['Magnitude'] = np.select(conditions, choices, default='Negligible')

# 4. Define the Target Variable: Resolution Class
# Matching your paper: Class 1 (Success - Fast Resolution) vs Class 2 (Challenging - Slow Resolution)
df['Resolution_Class'] = np.where(df['Fixing_Duration_Days'] <= 14, 1, 2)

# 5. Clean up and select the final engineered columns
final_features = [
    'Issue_ID', 'Fixing_Duration_Days', 'Comments', 
    'Body_Length', 'Priority_Score', 'Magnitude', 'Resolution_Class'
]
df_clean = df[final_features]

# 6. Save the analysis-ready dataset
save_path = 'AIRIMF_Engineered_Dataset.csv'
df_clean.to_csv(save_path, index=False)

print(f"Pipeline Complete! Engineered {len(df_clean)} records.")
print(f"Data saved to: {save_path}\n")
print("--- Data Sample ---")
print(df_clean[['Fixing_Duration_Days', 'Priority_Score', 'Magnitude', 'Resolution_Class']].head())