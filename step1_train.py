"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BURNOUT RISK ASSESSMENT — STEP 1: MODEL TRAINING                          ║
║   Run AFTER step0_load_wesad.py has saved wesad_features.csv                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── CELL 1  Install (run once in Colab) ─────────────────────────────────────
# !pip install neurokit2 xgboost shap scikit-learn pandas numpy scipy matplotlib seaborn tqdm -q

# ─── CELL 2  Imports ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import pickle, os, warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneGroupOut
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, accuracy_score,
                             precision_score, recall_score)
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# KEY: EnsembleModel lives in model_utils so pickle works in every script
from model_utils import EnsembleModel, FEATURE_DEFAULTS

print("✅ All imports successful")


# ─── CELL 3  Config ──────────────────────────────────────────────────────────
class Config:
    FEATURES_CSV = "/content/wesad_features.csv"
    MODEL_PATH   = "trained_model.pkl"
    SCALER_PATH  = "scaler.pkl"
    COLS_PATH    = "feature_columns.pkl"
    SHAP_CSV     = "shap_importance.csv"

cfg = Config()
print("✅ Config ready")


# ─── CELL 4  Load WESAD features ─────────────────────────────────────────────
def load_wesad_features(csv_path):
    if not os.path.exists(csv_path):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wesad_features.csv")
        if os.path.exists(alt):
            csv_path = alt
        else:
            raise FileNotFoundError(
                f"Feature CSV not found: {csv_path}\n"
                "Run step0_load_wesad.py first to generate wesad_features.csv"
            )
    print(f"📂 Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Shape    : {df.shape}")
    print(f"   Subjects : {sorted(df['subject'].unique().tolist())}")
    vc = df['label'].value_counts().rename({0: 'Normal', 1: 'Stress'})
    print(f"   Classes  :\n{vc.to_string()}")
    return df


print("Loading WESAD feature dataset...")
df = load_wesad_features(cfg.FEATURES_CSV)


# ─── CELL 5  Feature matrix ──────────────────────────────────────────────────
META_COLS    = ['subject', 'window', 'label']
FEATURE_COLS = [c for c in df.columns if c not in META_COLS]
groups       = df['subject'].values

X = df[FEATURE_COLS].copy()
y = df['label'].values

X.replace([np.inf, -np.inf], np.nan, inplace=True)
feat_medians = X.median()
X.fillna(feat_medians, inplace=True)

print(f"\n✅ Feature matrix : {X.shape}")
print(f"   {len(FEATURE_COLS)} features  |  {len(np.unique(groups))} subjects")

scaler   = RobustScaler()
X_scaled = scaler.fit_transform(X)
print("✅ RobustScaler fitted")


# ─── CELL 6  Model definitions ───────────────────────────────────────────────
xgb_model = xgb.XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    eval_metric='logloss', random_state=42, n_jobs=-1,
    use_label_encoder=False,
)

rf_model = RandomForestClassifier(
    n_estimators=400, max_depth=10, min_samples_split=8,
    min_samples_leaf=3, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1,
)
print("✅ Models defined")


# ─── CELL 7  Leave-One-Subject-Out CV ────────────────────────────────────────
print("\n📊 Leave-One-Subject-Out CV (gold-standard for WESAD)...")
logo = LeaveOneGroupOut()
loso_auc, loso_f1, loso_acc, loso_subjects = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(logo.split(X_scaled, y, groups)):
    subj = groups[test_idx[0]]
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y[train_idx],         y[test_idx]

    xgb_model.fit(X_tr, y_tr)
    proba = xgb_model.predict_proba(X_te)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    if len(np.unique(y_te)) < 2:
        continue

    a = roc_auc_score(y_te, proba)
    f = f1_score(y_te, pred, zero_division=0)
    c = accuracy_score(y_te, pred)
    loso_auc.append(a); loso_f1.append(f); loso_acc.append(c)
    loso_subjects.append(subj)
    print(f"   Fold {fold+1:2d} ({subj}):  AUC={a:.3f}  F1={f:.3f}  ACC={c:.3f}")

print(f"\n  ► LOSO Mean AUC : {np.mean(loso_auc):.3f} ± {np.std(loso_auc):.3f}")
print(f"  ► LOSO Mean F1  : {np.mean(loso_f1):.3f} ± {np.std(loso_f1):.3f}")
print(f"  ► LOSO Mean ACC : {np.mean(loso_acc):.3f} ± {np.std(loso_acc):.3f}")


# ─── CELL 8  Stratified K-Fold (secondary) ───────────────────────────────────
print("\n📊 Stratified 5-Fold CV (for comparison)...")
skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_auc= cross_val_score(xgb_model, X_scaled, y, cv=skf, scoring='roc_auc', n_jobs=-1)
xgb_f1 = cross_val_score(xgb_model, X_scaled, y, cv=skf, scoring='f1',      n_jobs=-1)
rf_auc = cross_val_score(rf_model,  X_scaled, y, cv=skf, scoring='roc_auc', n_jobs=-1)
print(f"  XGBoost AUC: {xgb_auc.mean():.3f} ± {xgb_auc.std():.3f} | F1: {xgb_f1.mean():.3f}")
print(f"  RF      AUC: {rf_auc.mean():.3f} ± {rf_auc.std():.3f}")


# ─── CELL 9  Fit final models ────────────────────────────────────────────────
print("\n🔧 Fitting final ensemble on full dataset...")
xgb_model.fit(X_scaled, y)
rf_model.fit(X_scaled, y)


# ─── CELL 10  Build ensemble ─────────────────────────────────────────────────
# EnsembleModel imported from model_utils — pickle path = model_utils.EnsembleModel
ensemble = EnsembleModel(
    models          = [xgb_model, rf_model],
    weights         = [0.6, 0.4],
    feature_medians = feat_medians.to_dict(),
    feature_cols    = FEATURE_COLS,
)

y_proba = ensemble.predict_proba(X_scaled)[:, 1]
y_pred  = ensemble.predict(X_scaled)

print(f"\n✅ Ensemble Training-Set Metrics:")
print(f"   Accuracy : {accuracy_score(y, y_pred):.3f}")
print(f"   ROC-AUC  : {roc_auc_score(y, y_proba):.3f}")
print(f"   F1-Score : {f1_score(y, y_pred):.3f}")
print(f"   Precision: {precision_score(y, y_pred):.3f}")
print(f"   Recall   : {recall_score(y, y_pred):.3f}")


# ─── CELL 11  SHAP ───────────────────────────────────────────────────────────
print("\n🔍 Computing SHAP values...")
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_scaled)

shap_importance = pd.DataFrame({
    'feature':    FEATURE_COLS,
    'importance': np.abs(shap_values).mean(0)
}).sort_values('importance', ascending=False).reset_index(drop=True)

print("\nTop 15 features:")
print(shap_importance.head(15).to_string(index=False))

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_scaled, feature_names=FEATURE_COLS,
                  max_display=15, show=False, plot_type='bar')
plt.title('SHAP Feature Importance — Burnout Risk (WESAD)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ shap_importance.png saved")


# ─── CELL 12  Confusion matrix ───────────────────────────────────────────────
print("\n📊 Classification Report:")
print(classification_report(y, y_pred, target_names=['Normal', 'High Risk']))

cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'High Risk'],
            yticklabels=['Normal', 'High Risk'],
            linewidths=0.5, ax=ax)
ax.set_title('Confusion Matrix — Ensemble (WESAD Training)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()


# ─── CELL 13  LOSO bar chart ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(8, len(loso_auc)), 4))
x = np.arange(len(loso_auc))
ax.bar(x - 0.2, loso_auc, 0.35, label='AUC',      color='#3b82f6', alpha=0.85)
ax.bar(x + 0.2, loso_f1,  0.35, label='F1-Score', color='#22c55e', alpha=0.85)
ax.axhline(np.mean(loso_auc), color='#3b82f6', linestyle='--', linewidth=1.2, alpha=0.7)
ax.axhline(np.mean(loso_f1),  color='#22c55e', linestyle='--', linewidth=1.2, alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(loso_subjects, fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_title('Leave-One-Subject-Out CV — AUC & F1 per Fold', fontsize=12, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('loso_cv_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ loso_cv_results.png saved")


# ─── CELL 14  Save artifacts ─────────────────────────────────────────────────
with open(cfg.MODEL_PATH,  'wb') as f: pickle.dump(ensemble,     f, protocol=pickle.HIGHEST_PROTOCOL)
with open(cfg.SCALER_PATH, 'wb') as f: pickle.dump(scaler,       f, protocol=pickle.HIGHEST_PROTOCOL)
with open(cfg.COLS_PATH,   'wb') as f: pickle.dump(FEATURE_COLS, f, protocol=pickle.HIGHEST_PROTOCOL)
shap_importance.to_csv(cfg.SHAP_CSV, index=False)

print(f"\n✅ trained_model.pkl    saved")
print(f"✅ scaler.pkl           saved")
print(f"✅ feature_columns.pkl  saved")
print(f"✅ shap_importance.csv  saved")
print(f"\n🎉 Training complete!  LOSO AUC = {np.mean(loso_auc):.3f}")
print("   Run step3_validate.py next.")
