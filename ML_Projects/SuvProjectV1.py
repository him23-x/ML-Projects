# ============================================================
# SUV Data Analysis - Purchase Prediction using Random Forest
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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
df = pd.read_csv('suv_data.csv')

print("=" * 55)
print("         SUV DATA ANALYSIS - FULL REPORT")
print("=" * 55)
print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n🔍 First 5 rows:")
print(df.head())
print("\n📋 Data Info:")
print(df.dtypes)
print("\n📈 Statistical Summary:")
print(df.describe())
print(f"\n🎯 Target Distribution:")
print(df['Purchased'].value_counts().rename({0: 'Did NOT buy', 1: 'Bought SUV'}))

# ────────────────────────────────────────────
# 2. Preprocessing
# ────────────────────────────────────────────
le = LabelEncoder()
df['Gender_enc'] = le.fit_transform(df['Gender'])  # Male=1, Female=0

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
# 3. Train Models
# ────────────────────────────────────────────
# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_sc, y_train)
y_pred_rf = rf.predict(X_test_sc)
acc_rf = accuracy_score(y_test, y_pred_rf)

# Logistic Regression (for comparison)
lr = LogisticRegression(random_state=42)
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)
acc_lr = accuracy_score(y_test, y_pred_lr)

print(f"\n🌲 Random Forest Accuracy : {acc_rf:.4f} ({acc_rf*100:.2f}%)")
print(f"📉 Logistic Regression Acc: {acc_lr:.4f} ({acc_lr*100:.2f}%)")

print("\n📊 Random Forest - Classification Report:")
print(classification_report(y_test, y_pred_rf,
                             target_names=['Not Purchased', 'Purchased']))

# Feature importance
importances = rf.feature_importances_
feat_names  = ['Gender', 'Age', 'Estimated Salary']

# ────────────────────────────────────────────
# 4. Visualizations
# ────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
colors = {'main': '#1A5276', 'accent': '#2E86C1', 'highlight': '#E74C3C',
          'green': '#1E8449', 'light': '#AED6F1'}

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#F8F9FA')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── Plot 1: Target Distribution (Pie)
ax1 = fig.add_subplot(gs[0, 0])
vals  = df['Purchased'].value_counts()
wedge_props = dict(width=0.55, edgecolor='white', linewidth=2)
ax1.pie(vals, labels=['Not Purchased', 'Purchased'],
        autopct='%1.1f%%', colors=[colors['light'], colors['main']],
        startangle=90, wedgeprops=wedge_props,
        textprops={'fontsize': 10})
ax1.set_title('Purchase Distribution', fontweight='bold', fontsize=12, pad=10)

# ── Plot 2: Age vs Purchase (Box)
ax2 = fig.add_subplot(gs[0, 1])
df.boxplot(column='Age', by='Purchased', ax=ax2,
           patch_artist=True,
           boxprops=dict(facecolor=colors['light'], color=colors['main']),
           medianprops=dict(color=colors['highlight'], linewidth=2))
ax2.set_title('Age by Purchase Decision', fontweight='bold', fontsize=12)
ax2.set_xlabel('Purchased (0=No, 1=Yes)')
ax2.set_ylabel('Age')
plt.sca(ax2); plt.title('Age by Purchase Decision', fontweight='bold', fontsize=12)

# ── Plot 3: Salary vs Purchase (Box)
ax3 = fig.add_subplot(gs[0, 2])
df.boxplot(column='EstimatedSalary', by='Purchased', ax=ax3,
           patch_artist=True,
           boxprops=dict(facecolor='#ABEBC6', color=colors['green']),
           medianprops=dict(color=colors['highlight'], linewidth=2))
ax3.set_title('Salary by Purchase Decision', fontweight='bold', fontsize=12)
ax3.set_xlabel('Purchased (0=No, 1=Yes)')
ax3.set_ylabel('Estimated Salary')
plt.sca(ax3); plt.title('Salary by Purchase Decision', fontweight='bold', fontsize=12)

# ── Plot 4: Age vs Salary Scatter
ax4 = fig.add_subplot(gs[1, :2])
not_bought = df[df['Purchased'] == 0]
bought     = df[df['Purchased'] == 1]
ax4.scatter(not_bought['Age'], not_bought['EstimatedSalary'],
            c=colors['light'], edgecolors=colors['accent'],
            alpha=0.7, s=50, label='Not Purchased')
ax4.scatter(bought['Age'], bought['EstimatedSalary'],
            c=colors['highlight'], edgecolors='#922B21',
            alpha=0.8, s=60, label='Purchased', marker='D')
ax4.set_xlabel('Age', fontsize=11)
ax4.set_ylabel('Estimated Salary ($)', fontsize=11)
ax4.set_title('Age vs Salary — Who Bought the SUV?', fontweight='bold', fontsize=12)
ax4.legend(fontsize=10)

# ── Plot 5: Feature Importance
ax5 = fig.add_subplot(gs[1, 2])
sorted_idx = np.argsort(importances)
bar_colors = [colors['accent'], colors['highlight'], colors['green']]
bars = ax5.barh([feat_names[i] for i in sorted_idx],
                [importances[i] for i in sorted_idx],
                color=[bar_colors[i] for i in sorted_idx],
                edgecolor='white', height=0.5)
ax5.set_title('Feature Importance\n(Random Forest)', fontweight='bold', fontsize=12)
ax5.set_xlabel('Importance Score')
for bar, val in zip(bars, [importances[i] for i in sorted_idx]):
    ax5.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

# ── Plot 6: Confusion Matrix
ax6 = fig.add_subplot(gs[2, 0])
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
            xticklabels=['Not Bought', 'Bought'],
            yticklabels=['Not Bought', 'Bought'],
            linewidths=1, linecolor='white', annot_kws={'size': 14})
ax6.set_title('Confusion Matrix\n(Random Forest)', fontweight='bold', fontsize=12)
ax6.set_ylabel('Actual')
ax6.set_xlabel('Predicted')

# ── Plot 7: ROC Curve
ax7 = fig.add_subplot(gs[2, 1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf.predict_proba(X_test_sc)[:, 1])
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr.predict_proba(X_test_sc)[:, 1])
ax7.plot(fpr_rf, tpr_rf, color=colors['main'], lw=2,
         label=f'Random Forest (AUC = {auc(fpr_rf,tpr_rf):.3f})')
ax7.plot(fpr_lr, tpr_lr, color=colors['highlight'], lw=2, linestyle='--',
         label=f'Logistic Reg  (AUC = {auc(fpr_lr,tpr_lr):.3f})')
ax7.plot([0, 1], [0, 1], 'gray', linestyle=':', lw=1)
ax7.set_xlabel('False Positive Rate')
ax7.set_ylabel('True Positive Rate')
ax7.set_title('ROC Curve — Model Comparison', fontweight='bold', fontsize=12)
ax7.legend(fontsize=9)

# ── Plot 8: Gender vs Purchase (Bar)
ax8 = fig.add_subplot(gs[2, 2])
gender_purchase = df.groupby(['Gender', 'Purchased']).size().unstack(fill_value=0)
gender_purchase.plot(kind='bar', ax=ax8, color=[colors['light'], colors['main']],
                     edgecolor='white', width=0.5)
ax8.set_title('Gender vs Purchase Decision', fontweight='bold', fontsize=12)
ax8.set_xlabel('Gender')
ax8.set_ylabel('Count')
ax8.set_xticklabels(['Female', 'Male'], rotation=0)
ax8.legend(['Not Purchased', 'Purchased'], fontsize=9)

# ── Super Title
fig.suptitle('🚙 SUV Purchase Prediction — Full Analysis Dashboard',
             fontsize=16, fontweight='bold', y=0.98, color=colors['main'])

import os
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'suv_analysis_dashboard.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\n✅ Dashboard saved → {output_path}")
print("\n✅ Dashboard saved → suv_analysis_dashboard.png")

# ────────────────────────────────────────────
# 5. Key Insights Summary
# ────────────────────────────────────────────
print("\n" + "=" * 55)
print("              KEY INSIGHTS")
print("=" * 55)
print(f"  • Most important feature : {feat_names[np.argmax(importances)]}")
print(f"  • Second most important  : {feat_names[np.argsort(importances)[-2]]}")
print(f"  • Model Accuracy         : {acc_rf*100:.2f}%")
avg_age_bought   = df[df['Purchased']==1]['Age'].mean()
avg_sal_bought   = df[df['Purchased']==1]['EstimatedSalary'].mean()
avg_age_notbought= df[df['Purchased']==0]['Age'].mean()
avg_sal_notbought= df[df['Purchased']==0]['EstimatedSalary'].mean()
print(f"\n  Buyers    → Avg Age: {avg_age_bought:.1f}  | Avg Salary: ${avg_sal_bought:,.0f}")
print(f"  Non-Buyers→ Avg Age: {avg_age_notbought:.1f}  | Avg Salary: ${avg_sal_notbought:,.0f}")
print("\n  ✅ Older people with higher salaries are more likely to buy.")
print("=" * 55)