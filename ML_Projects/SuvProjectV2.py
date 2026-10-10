import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# ────────────────────────────────────────────
# 1. Load & Explore Data
# ────────────────────────────────────────────
df = pd.read_csv(r'suv_data.csv')   # ← غيّر المسار لو لزم

print("=" * 55)
print("   SUV PURCHASE PREDICTION - LOGISTIC REGRESSION")
print("=" * 55)
print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n🔍 First 5 rows:")
print(df.head())
print("\n📈 Statistical Summary:")
print(df.describe())
print(f"\n🎯 Target Distribution:")
print(df['Purchased'].value_counts().rename({0: 'Did NOT buy', 1: 'Bought SUV'}))

# ────────────────────────────────────────────
# 2. Preprocessing
# ────────────────────────────────────────────
le = LabelEncoder()
df['Gender_enc'] = le.fit_transform(df['Gender'])   # Female=0, Male=1

X = df[['Gender_enc', 'Age', 'EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n✂️  Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ────────────────────────────────────────────
# 3. Train Logistic Regression
# ────────────────────────────────────────────
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_sc, y_train)

y_pred      = model.predict(X_test_sc)
y_prob      = model.predict_proba(X_test_sc)[:, 1]
accuracy    = accuracy_score(y_test, y_pred)

print(f"\n✅ Logistic Regression Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=['Not Purchased', 'Purchased']))

# Coefficients
feat_names = ['Gender', 'Age', 'Estimated Salary']
coefs      = model.coef_[0]
print("📐 Model Coefficients (after scaling):")
for name, coef in zip(feat_names, coefs):
    direction = "⬆️ increases" if coef > 0 else "⬇️ decreases"
    print(f"   {name:20s}: {coef:+.4f}  → {direction} purchase probability")
print(f"   Intercept: {model.intercept_[0]:+.4f}")

# ────────────────────────────────────────────
# 4. Visualizations
# ────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
BLUE   = '#1A5276'
LBLUE  = '#AED6F1'
RED    = '#E74C3C'
GREEN  = '#1E8449'

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#F8F9FA')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── Plot 1: Confusion Matrix
ax1 = fig.add_subplot(gs[0, 0])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Not Bought', 'Bought'],
            yticklabels=['Not Bought', 'Bought'],
            linewidths=1, linecolor='white', annot_kws={'size': 14})
ax1.set_title('Confusion Matrix', fontweight='bold', fontsize=12)
ax1.set_ylabel('Actual')
ax1.set_xlabel('Predicted')

# ── Plot 2: ROC Curve
ax2 = fig.add_subplot(gs[0, 1])
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc     = auc(fpr, tpr)
ax2.plot(fpr, tpr, color=BLUE, lw=2.5,
         label=f'Logistic Regression\nAUC = {roc_auc:.3f}')
ax2.fill_between(fpr, tpr, alpha=0.1, color=BLUE)
ax2.plot([0, 1], [0, 1], 'gray', linestyle='--', lw=1.2)
ax2.set_xlabel('False Positive Rate', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontsize=11)
ax2.set_title('ROC Curve', fontweight='bold', fontsize=12)
ax2.legend(fontsize=10, loc='lower right')

# ── Plot 3: Coefficients Bar Chart
ax3 = fig.add_subplot(gs[0, 2])
bar_colors = [RED if c < 0 else GREEN for c in coefs]
bars = ax3.barh(feat_names, coefs, color=bar_colors, edgecolor='white', height=0.5)
ax3.axvline(0, color='black', linewidth=0.8)
ax3.set_title('Feature Coefficients\n(Impact on Purchase)', fontweight='bold', fontsize=12)
ax3.set_xlabel('Coefficient Value')
for bar, val in zip(bars, coefs):
    ax3.text(val + (0.02 if val >= 0 else -0.02),
             bar.get_y() + bar.get_height()/2,
             f'{val:+.3f}', va='center',
             ha='left' if val >= 0 else 'right',
             fontsize=10, fontweight='bold')

# ── Plot 4: Predicted Probability Distribution
ax4 = fig.add_subplot(gs[1, 0])
prob_not = y_prob[y_test == 0]
prob_yes = y_prob[y_test == 1]
ax4.hist(prob_not, bins=20, alpha=0.6, color=LBLUE,  label='Not Purchased', edgecolor='white')
ax4.hist(prob_yes, bins=20, alpha=0.6, color=RED,    label='Purchased',     edgecolor='white')
ax4.axvline(0.5, color='black', linestyle='--', lw=1.5, label='Threshold = 0.5')
ax4.set_xlabel('Predicted Probability', fontsize=11)
ax4.set_ylabel('Count', fontsize=11)
ax4.set_title('Predicted Probability\nDistribution', fontweight='bold', fontsize=12)
ax4.legend(fontsize=9)

# ── Plot 5: Age vs Probability
ax5 = fig.add_subplot(gs[1, 1])
sc = ax5.scatter(X_test['Age'], y_prob,
                 c=y_test, cmap='coolwarm', alpha=0.7, s=60, edgecolors='white')
ax5.axhline(0.5, color='black', linestyle='--', lw=1.2, label='Decision boundary')
ax5.set_xlabel('Age', fontsize=11)
ax5.set_ylabel('Predicted Probability', fontsize=11)
ax5.set_title('Age vs Purchase\nProbability', fontweight='bold', fontsize=12)
ax5.legend(fontsize=9)
plt.colorbar(sc, ax=ax5, label='Actual (0=No, 1=Yes)')

# ── Plot 6: Salary vs Probability
ax6 = fig.add_subplot(gs[1, 2])
sc2 = ax6.scatter(X_test['EstimatedSalary'], y_prob,
                  c=y_test, cmap='coolwarm', alpha=0.7, s=60, edgecolors='white')
ax6.axhline(0.5, color='black', linestyle='--', lw=1.2, label='Decision boundary')
ax6.set_xlabel('Estimated Salary ($)', fontsize=11)
ax6.set_ylabel('Predicted Probability', fontsize=11)
ax6.set_title('Salary vs Purchase\nProbability', fontweight='bold', fontsize=12)
ax6.legend(fontsize=9)
plt.colorbar(sc2, ax=ax6, label='Actual (0=No, 1=Yes)')

fig.suptitle('🚙 SUV Purchase Prediction — Logistic Regression Dashboard',
             fontsize=15, fontweight='bold', y=0.98, color=BLUE)

import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'suv_logistic_dashboard.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\n✅ Dashboard saved → {out_path}")

# ────────────────────────────────────────────
# 5. Predict on New Sample
# ────────────────────────────────────────────
print("\n" + "=" * 55)
print("         PREDICT ON NEW CUSTOMERS")
print("=" * 55)

new_customers = pd.DataFrame({
    'Gender_enc':       [1,    0,    1],      # 1=Male, 0=Female
    'Age':              [45,   22,   35],
    'EstimatedSalary':  [90000, 25000, 60000]
})

new_scaled = scaler.transform(new_customers)
preds      = model.predict(new_scaled)
probs      = model.predict_proba(new_scaled)[:, 1]

labels = ['Male', 'Female', 'Male']
ages   = [45, 22, 35]
sals   = [90000, 25000, 60000]

for i in range(len(preds)):
    decision = "✅ WILL BUY" if preds[i] == 1 else "❌ Won't Buy"
    print(f"\n  Customer {i+1}: {labels[i]}, Age={ages[i]}, Salary=${sals[i]:,}")
    print(f"  → Prediction: {decision}  (Probability: {probs[i]:.2%})")

print("\n" + "=" * 55)