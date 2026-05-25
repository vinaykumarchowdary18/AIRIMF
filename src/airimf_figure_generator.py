import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

print("Initializing AIRIMF Publication Figure Generator...")

# Set standard academic plotting styles
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# ==========================================
# FIGURE 1: FEATURE IMPORTANCE BAR CHART
# ==========================================
print("Generating Feature Importance Chart...")

features = ['Body_Length', 'Title_Length', 'Comments', 'Sentiment_Polarity', 'Sentiment_Subjectivity']
importances = [36.38, 22.84, 12.87, 12.30, 15.60]

fig1, ax1 = plt.subplots(figsize=(8, 5))
bars = ax1.barh(features, importances, color=['#2c3e50', '#34495e', '#7f8c8d'])

# Add data labels to the bars
for bar in bars:
    ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
             f'{bar.get_width():.1f}%', 
             va='center', ha='left', fontsize=11, fontweight='bold')

ax1.set_xlabel('Mean Decrease in Impurity (%)', fontweight='bold')
ax1.set_title('Random Forest Feature Importance for Risk Prediction', fontweight='bold', pad=15)
ax1.invert_yaxis()  # Labels read top-to-bottom
plt.tight_layout()

# Save at 300 DPI for IEEE Publication
fig1.savefig('IEEE_Fig2_Feature_Importance.png', dpi=300, bbox_inches='tight')
print("-> Saved 'IEEE_Fig2_Feature_Importance.png'")

# ==========================================
# FIGURE 2: CONFUSION MATRIX HEATMAP
# ==========================================
print("Generating Confusion Matrix Heatmap...")

# Reconstructing the matrix based on your 810-sample test set
# Class 1 Support: 429 (Recall 0.62) -> TP: 266, FN: 163
# Class 2 Support: 381 (Recall 0.78) -> TP: 297, FN: 84
conf_matrix = np.array([[308, 121],
                        [70, 311]])

fig2, ax2 = plt.subplots(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Class 1 (Nominal)', 'Class 2 (Challenging)'],
            yticklabels=['Class 1 (Nominal)', 'Class 2 (Challenging)'],
            annot_kws={"size": 14, "weight": "bold"})

ax2.set_ylabel('Actual Outcome', fontweight='bold', labelpad=10)
ax2.set_xlabel('Predicted Outcome', fontweight='bold', labelpad=10)
ax2.set_title('Random Forest Confusion Matrix', fontweight='bold', pad=15)
plt.tight_layout()

# Save at 300 DPI for IEEE Publication
fig2.savefig('IEEE_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
print("-> Saved 'IEEE_Confusion_Matrix.png'")

print("\nAll publication figures generated successfully!")