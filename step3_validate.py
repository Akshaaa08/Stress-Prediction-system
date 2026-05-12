"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BURNOUT RISK ASSESSMENT — STEP 3: CROSS-DATASET VALIDATION (WESAD→SWELL)  ║
║                                                                              ║
║   Validates the WESAD-trained model on the SWELL HRV dataset.               ║
║                                                                              ║
║   SWELL HRV Dataset folder layout (what you downloaded):                    ║
║     hrv dataset/                                                             ║
║       raw/                                                                   ║
║         hrv reading labels.csv   ← raw HRV readings + labels               ║
║       final/                                                                 ║
║         train.csv                ← use this for validation                  ║
║         test.csv                 ← optional hold-out test set               ║
║                                                                              ║
║   HOW TO USE IN COLAB — RECOMMENDED (fastest):                              ║
║     1. Mount Google Drive (a pop-up will appear asking for permission).     ║
║        This script does it automatically.                                    ║
║     2. Upload train.csv to your Google Drive (browser upload, once).        ║
║     3. Set SWELL_DRIVE_PATH below to its path on Drive.                     ║
║                                                                              ║
║   ALTERNATIVE — Manual path:                                                 ║
║     If the file is already in /content/, set SWELL_TRAIN_CSV directly.     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Colab run order
───────────────
  %run step0_load_wesad.py   ← generates wesad_features.csv  (~20-40 min)
  %run step1_train.py         ← trains model, saves .pkl files  (~5 min)
  %run step3_validate.py      ← THIS FILE — validates on SWELL HRV
"""

# ─── Imports ─────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import pickle, os, warnings
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    accuracy_score, confusion_matrix, roc_curve,
    precision_score, recall_score
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
warnings.filterwarnings('ignore')

# KEY FIX: import EnsembleModel from model_utils BEFORE unpickling
from model_utils import EnsembleModel
print("✅ Imports OK")


# ─── Configuration ───────────────────────────────────────────────────────────
MODEL_PATH  = "trained_model.pkl"
SCALER_PATH = "scaler.pkl"
COLS_PATH   = "feature_columns.pkl"
SHAP_CSV    = "shap_importance.csv"

# ══════════════════════════════════════════════════════════════════════════════
#  ▼▼▼  EDIT THESE TWO LINES  ▼▼▼
#
#  OPTION A — Google Drive (FASTEST — avoids slow browser upload)
#  Step 1: Upload train.csv to your Google Drive once from your browser.
#  Step 2: Set the path below to where it lives on Drive.
#  Example: if you put it in "My Drive/swell/train.csv", use:
#              "train.csv"         → SWELL_DRIVE_PATH = "train.csv"
#              in a subfolder      → SWELL_DRIVE_PATH = "swell/train.csv"
SWELL_DRIVE_PATH = "train.csv"   # ← relative path inside your Google Drive root

#  OPTION B — File already copied to Colab's /content/ directory
#  Set this to the full /content/... path, leave SWELL_DRIVE_PATH as None.
SWELL_TRAIN_CSV  = None          # e.g. "/content/train.csv"

#  OPTION C — Also run on test.csv
SWELL_TEST_CSV   = None          # e.g. "/content/drive/MyDrive/swell/test.csv"
USE_TEST_CSV     = False
#
#  ▲▲▲  END OF EDIT SECTION  ▲▲▲
# ══════════════════════════════════════════════════════════════════════════════


# ─── Google Drive mount helper ────────────────────────────────────────────────
def mount_drive_and_resolve(drive_relative_path):
    """
    Mount Google Drive (shows permission popup once) and return the full path
    to drive_relative_path inside /content/drive/MyDrive/.

    Returns the resolved path string, or None if not in Colab / mount fails.
    """
    if not drive_relative_path:
        return None
    try:
        from google.colab import drive as colab_drive
        mount_point = "/content/drive"
        if not os.path.exists(os.path.join(mount_point, "MyDrive")):
            print("📂 Mounting Google Drive…")
            colab_drive.mount(mount_point)
        full_path = os.path.join(mount_point, "MyDrive", drive_relative_path)
        if os.path.exists(full_path):
            print(f"✅ Found on Drive: {full_path}")
            return full_path
        else:
            print(f"⚠  File not found on Drive: {full_path}")
            print(f"   Make sure you uploaded train.csv to Drive and set SWELL_DRIVE_PATH correctly.")
            return None
    except ImportError:
        return None   # Not running in Colab
    except Exception as e:
        print(f"⚠  Drive mount error: {e}")
        return None


# ─── Load artifacts ──────────────────────────────────────────────────────────
def load_artifacts():
    missing = [p for p in (MODEL_PATH, SCALER_PATH, COLS_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Artifact file(s) not found:\n" +
            "\n".join(f"  • {p}" for p in missing) +
            "\n\nFix: run step1_train.py first, then re-run step3_validate.py."
        )
    with open(MODEL_PATH,  'rb') as f: model        = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f: scaler       = pickle.load(f)
    with open(COLS_PATH,   'rb') as f: feature_cols = pickle.load(f)

    print(f"✅ Model  : {type(model).__name__}  ({len(feature_cols)} features)")
    print(f"✅ Scaler : {type(scaler).__name__}")
    return model, scaler, feature_cols


# ─── Colab upload widget (last-resort fallback) ───────────────────────────────
def colab_upload_csv(prompt_label="train.csv"):
    """
    Show a Colab file-upload widget and return the path of the uploaded file.
    Returns None if not running in Colab.
    NOTE: This is slow for large files — prefer Google Drive (SWELL_DRIVE_PATH).
    """
    try:
        from google.colab import files as colab_files
        print(f"\n📤 Please upload your SWELL HRV  '{prompt_label}'  file now.")
        print("   (Navigate to  hrv dataset/final/  on your computer and select the file.)")
        print("   ⚠  Tip: for faster access next time, upload to Google Drive instead.\n")
        uploaded = colab_files.upload()
        if not uploaded:
            print("   ⚠  No file uploaded.")
            return None
        fname = list(uploaded.keys())[0]
        dest  = f"/content/{fname}"
        print(f"   ✅ Uploaded → {dest}  ({len(uploaded[fname]):,} bytes)")
        return dest
    except ImportError:
        return None   # Not in Colab


# ─── SWELL HRV column mapping ─────────────────────────────────────────────────
SWELL_HRV_COL_MAP = {
    # ── Time-domain HRV ──────────────────────────────────────────────────────
    "mean_nni":          "hrv_mean_rr",
    "meanNN":            "hrv_mean_rr",
    "mean_rr":           "hrv_mean_rr",
    "mean_hr":           "hrv_mean_hr",
    "sdnn":              "hrv_sdnn",
    "SDNN":              "hrv_sdnn",
    "rmssd":             "hrv_rmssd",
    "RMSSD":             "hrv_rmssd",
    "pnn50":             "hrv_pnn50",
    "pNN50":             "hrv_pnn50",
    "pnn20":             "hrv_pnn20",
    "pNN20":             "hrv_pnn20",
    "cvnn":              "hrv_cv",
    "cv":                "hrv_cv",
    "nni_50":            "hrv_pnn50",
    "range_nni":         "hrv_tinn",
    "tinn":              "hrv_tinn",
    "TINN":              "hrv_tinn",
    "sdsd":              "hrv_sdsd",
    "SDSD":              "hrv_sdsd",
    "skewness":          "hrv_skewness",
    "kurtosis":          "hrv_kurtosis",
    # ── Frequency-domain HRV ─────────────────────────────────────────────────
    "lf":                "hrv_lf",
    "LF":                "hrv_lf",
    "hf":                "hrv_hf",
    "HF":                "hrv_hf",
    "vlf":               "hrv_vlf",
    "VLF":               "hrv_vlf",
    "lf_hf_ratio":       "hrv_lf_hf",
    "lf_hf":             "hrv_lf_hf",
    "lfhf":              "hrv_lf_hf",
    "lf_nu":             "hrv_lf_norm",
    "hf_nu":             "hrv_hf_norm",
    "total_power":       "hrv_tp",
    "tp":                "hrv_tp",
    # ── Non-linear HRV ───────────────────────────────────────────────────────
    "sd1":               "hrv_sd1",
    "SD1":               "hrv_sd1",
    "sd2":               "hrv_sd2",
    "SD2":               "hrv_sd2",
    "sd1_sd2":           "hrv_sd1_sd2",
    "ratio_sd2_sd1":     "hrv_sd1_sd2",
    "sample_entropy":    "hrv_sampen",
    "SampEn":            "hrv_sampen",
    # ── EDA (if present in SWELL) ────────────────────────────────────────────
    "eda_mean":          "eda_scl_mean",
    "eda_std":           "eda_scl_std",
    "scl_mean":          "eda_scl_mean",
    "scl_std":           "eda_scl_std",
    "scr_mean":          "eda_scr_mean",
    # ── Label synonyms ───────────────────────────────────────────────────────
    "label":             "label",
    "Label":             "label",
    "stress":            "label",
    "Stress":            "label",
    "condition":         "label",
    "Condition":         "label",
    "class":             "label",
    "Class":             "label",
}


def remap_swell_columns(df):
    """
    Rename SWELL HRV columns to WESAD feature names using SWELL_HRV_COL_MAP.
    """
    rename = {}
    for swell_col, wesad_col in SWELL_HRV_COL_MAP.items():
        if swell_col in df.columns and swell_col != wesad_col:
            rename[swell_col] = wesad_col

    if rename:
        safe_rename = {}
        for old, new in rename.items():
            if new not in df.columns:
                safe_rename[old] = new
        df = df.rename(columns=safe_rename)
        print(f"   ↳ Remapped {len(safe_rename)} SWELL columns → WESAD names")

    return df


# ─── FIX: robust binary label inference ───────────────────────────────────────
# The root cause of the TypeError was that the label column contained strings
# (e.g. "no stress", "time pressure", "interruption") instead of integers.
# pandas tried to compare a string Series with int 0 using >, which fails.
# The fix: detect strings first, map them explicitly, then handle numerics.

# String → int mappings seen in various SWELL releases
_SWELL_STR_LABEL_MAP = {
    # no-stress variants
    "no stress":       0, "no-stress":      0, "nostress":       0,
    "neutral":         0, "baseline":       0, "rest":           0,
    "normal":          0, "0":              0, "none":           0,
    # stress variants
    "time pressure":   1, "time-pressure":  1, "timepressure":   1,
    "stress":          1, "high stress":    1, "high-stress":    1,
    "interruption":    1, "interruptions":  1, "medium stress":  1,
    "medium-stress":   1, "1":              1, "2":              1,
}


def infer_binary_label(df):
    """
    Extract a binary stress label (0=normal, 1=stress) from whatever label
    column exists in the SWELL CSV.

    Handles:
      - Already-binary int/float  (0 / 1)
      - 3-class int               (0 / 1 / 2  →  0 / 1)
      - Boolean                   (True / False)
      - String labels             ("no stress" / "time pressure" / etc.)
      - Mixed string + numeric
    """
    label_col = next(
        (c for c in ['label', 'Label', 'stress', 'Stress',
                     'condition', 'Condition', 'class', 'Class']
         if c in df.columns),
        None
    )
    if label_col is None:
        raise ValueError(
            "No label column found in SWELL CSV.\n"
            "Expected one of: 'label', 'condition', 'stress', 'Stress', 'class'.\n"
            f"Columns present: {list(df.columns[:20])}"
        )

    raw        = df[label_col].copy()
    unique_raw = raw.dropna().unique()
    print(f"   Label column : '{label_col}'  |  unique values (up to 10): "
          f"{sorted(str(v) for v in unique_raw)[:10]}")

    # ── Case 1: boolean dtype ─────────────────────────────────────────────────
    if raw.dtype == bool:
        df['label'] = raw.astype(int)
        print("   Mapped: bool → int")
        return df

    # ── Case 2: string / object dtype ─────────────────────────────────────────
    if raw.dtype == object or any(isinstance(v, str) for v in unique_raw):
        mapped = raw.astype(str).str.strip().str.lower().map(_SWELL_STR_LABEL_MAP)
        n_unmapped = mapped.isna().sum()
        if n_unmapped > 0:
            unmapped_vals = (
                raw.astype(str).str.strip().str.lower()[mapped.isna()].unique()[:5]
            )
            print(f"   ⚠  {n_unmapped} rows have unrecognised label strings: {unmapped_vals}")
            print("      Treating unrecognised values as 0 (normal).")
            mapped = mapped.fillna(0)
        df['label'] = mapped.astype(int)
        print(f"   Mapped string labels → binary  "
              f"(stress={int(df['label'].sum())}, normal={int((df['label']==0).sum())})")
        return df

    # ── Case 3: numeric (int / float) ─────────────────────────────────────────
    # Force to numeric in case there are stray strings mixed in
    raw_num = pd.to_numeric(raw, errors='coerce')
    n_coerce_fail = raw_num.isna().sum() - raw.isna().sum()
    if n_coerce_fail > 0:
        print(f"   ⚠  {n_coerce_fail} values could not be parsed as numeric → set to 0")
    raw_num = raw_num.fillna(0)

    unique_num = sorted(raw_num.dropna().unique())

    if set(unique_num).issubset({0.0, 1.0}):
        df['label'] = raw_num.astype(int)
        print("   Already binary int (0/1)")
    elif set(unique_num).issubset({0.0, 1.0, 2.0}):
        df['label'] = (raw_num >= 1).astype(int)
        print("   Remapped 3-class (0=0, 1/2=1) → binary")
    else:
        df['label'] = (raw_num > 0).astype(int)
        print(f"   Mapped numeric to binary: 0=normal, >0=stress")

    return df


# ─── SWELL HRV CSV loader (primary) ──────────────────────────────────────────
def load_swell_hrv_csv(csv_path, feature_cols, split_name="train"):
    """
    Load and align the SWELL HRV  final/train.csv  (or test.csv).

    Steps:
      1. Read CSV
      2. Remap column names to WESAD feature names
      3. Infer binary stress label
      4. Zero-fill any WESAD features absent in SWELL
      5. Return (X, y, label_string)
    """
    print(f"\n{'─'*55}")
    print(f"📂 Loading SWELL HRV  [{split_name}]  →  {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Shape  : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"   Columns (first 10): {list(df.columns[:10])}")

    # Step 1 — remap SWELL column names
    df = remap_swell_columns(df)

    # Step 2 — label  (FIXED: handles string labels)
    df = infer_binary_label(df)

    # Step 3 — align to training feature columns
    missing_feats = [c for c in feature_cols if c not in df.columns]

    if missing_feats:
        print(f"   ⚠  {len(missing_feats)} WESAD features absent in SWELL CSV → zero-filled")
        if len(missing_feats) <= 12:
            print(f"      {missing_feats}")
        else:
            print(f"      (first 12) {missing_feats[:12]} ...")
        for c in missing_feats:
            df[c] = 0.0

    print(f"   Matched WESAD features in SWELL CSV: "
          f"{len(feature_cols) - len(missing_feats)} / {len(feature_cols)}")

    X = df[feature_cols].copy()
    y = df['label'].values

    # Clean
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    col_medians = X.median()
    X.fillna(col_medians, inplace=True)

    vc = pd.Series(y).value_counts().sort_index().rename({0: 'Normal', 1: 'Stress'})
    print(f"   Class balance:\n{vc.to_string()}")
    print(f"{'─'*55}")

    return X, y, f"SWELL HRV ({split_name}.csv)"


# ─── Physiological simulation (fallback) ─────────────────────────────────────
def generate_swell_simulation(feature_cols, n_samples=400, random_state=99):
    """
    Physiologically-grounded SWELL-KW simulation.
    Used only when no real CSV is available.
    """
    print("\nℹ  Running physiological simulation (no real SWELL CSV provided).")
    np.random.seed(random_state)
    n_stress   = n_samples // 2
    n_baseline = n_samples - n_stress
    SHIFT = 0.12

    def _sample(n, is_stress):
        rows = []
        for _ in range(n):
            s       = 1 if is_stress else -1
            mean_rr = np.clip(np.random.normal(760 * (1 - s*SHIFT*0.3), 65), 400, 1400)
            sdnn    = np.clip(np.random.normal(38  * (1 - s*SHIFT*0.5), 14), 5, 200)
            rmssd   = np.clip(np.random.normal(25  * (1 - s*SHIFT*0.5), 11), 5, 150)
            pnn50   = max(0, np.random.normal(9  * (1 - s*SHIFT*0.4), 5))
            pnn20   = max(0, np.random.normal(15 * (1 - s*SHIFT*0.3), 7))
            lf_hf   = np.clip(np.random.normal(3.5*(1 + s*SHIFT), 1.4), 0.3, 12)
            eda_m   = np.clip(np.random.normal(5.5*(1 + s*SHIFT*0.8), 2.2), 0, 20)
            eda_pk  = max(0, float(np.random.poisson(7 if is_stress else 3)))
            lf      = max(10, np.random.normal(580 if is_stress else 360, 140))
            hf      = max(10, np.random.normal(190 if is_stress else 270, 65))
            total   = lf + hf + 1e-6
            sdsd    = np.clip(np.random.normal(20 if is_stress else 35, 8), 1, 100)
            sd1     = rmssd / np.sqrt(2)
            sd2     = np.sqrt(max(2*sdnn**2 - sd1**2, 1e-6))
            t_mean  = np.random.normal(34.5 if is_stress else 35.5, 0.5)
            t_slope = np.random.normal(-0.002 if is_stress else 0.001, 0.001)

            rows.append({
                'hrv_mean_rr': mean_rr,   'hrv_mean_hr': 60000 / mean_rr,
                'hrv_sdnn': sdnn,          'hrv_rmssd': rmssd,
                'hrv_pnn50': pnn50,        'hrv_pnn20': pnn20,
                'hrv_cv': sdnn / mean_rr,
                'hrv_skewness': np.random.normal(0.1*s, 0.4),
                'hrv_kurtosis': np.random.normal(0.4, 0.4),
                'hrv_tinn': np.random.normal(300 if is_stress else 500, 80),
                'hrv_sdsd': sdsd,
                'hrv_lf': lf, 'hrv_hf': hf,
                'hrv_vlf': np.random.normal(160 + 90*int(is_stress), 45),
                'hrv_tp': lf + hf + np.random.normal(160, 45),
                'hrv_lf_hf': lf_hf,
                'hrv_lf_norm': lf / total, 'hrv_hf_norm': hf / total,
                'hrv_rr_slope': np.random.normal(-0.4 if is_stress else 0.25, 0.2),
                'hrv_stability': np.random.normal(1.4 if is_stress else 0.9, 0.25),
                'hrv_sd1': sd1, 'hrv_sd2': sd2, 'hrv_sd1_sd2': sd1 / max(sd2, 1e-6),
                'hrv_sampen': np.random.normal(1.5 if is_stress else 2.0, 0.3),
                'eda_scl_mean': eda_m,
                'eda_scl_std': eda_m * np.random.uniform(0.10, 0.28),
                'eda_scl_range': eda_m * np.random.uniform(0.20, 0.45),
                'eda_scl_min': max(0, eda_m * np.random.uniform(0.5, 0.8)),
                'eda_scl_max': eda_m * np.random.uniform(1.2, 1.6),
                'eda_scr_mean': np.random.uniform(0.08, 0.38 if is_stress else 0.12),
                'eda_scr_std': np.random.uniform(0.02, 0.18),
                'eda_scr_max': np.random.uniform(0.10, 1.10 if is_stress else 0.40),
                'eda_scr_energy': np.random.uniform(0.5, 3.0 if is_stress else 0.5),
                'eda_peak_count': eda_pk, 'eda_peak_rate': eda_pk,
                'eda_peak_amp_mean': np.random.uniform(0.05, 0.38 if is_stress else 0.15),
                'eda_peak_amp_std': np.random.uniform(0.02, 0.12),
                'eda_peak_amp_max': np.random.uniform(0.10, 0.90 if is_stress else 0.35),
                'eda_raw_mean': eda_m + np.random.normal(0, 0.18),
                'eda_raw_std': eda_m * np.random.uniform(0.04, 0.18),
                'eda_raw_skew': np.random.normal(0.45 if is_stress else 0.12, 0.25),
                'eda_raw_kurt': np.random.normal(0.8 if is_stress else 0.3, 0.3),
                'eda_trend_slope': np.random.normal(0.009 if is_stress else -0.003, 0.004),
                'temp_mean': t_mean, 'temp_std': np.random.uniform(0.05, 0.3),
                'temp_min': t_mean - np.random.uniform(0.1, 0.5),
                'temp_max': t_mean + np.random.uniform(0.1, 0.5),
                'temp_range': np.random.uniform(0.2, 1.0), 'temp_slope': t_slope,
                'fusion_autonomic_balance': lf_hf * (eda_m + 0.1),
                'fusion_parasympathetic_load': rmssd / (eda_m + 1.0),
                'fusion_stress_index': (lf_hf + 1) / (rmssd / 10 + 1),
                'fusion_recovery_capacity': sdnn / (lf_hf + 0.1),
                'fusion_scr_hrv_coupling': np.random.uniform(0.5, 3.0 if is_stress else 0.5) / (rmssd + 1e-6),
                'fusion_lfhf_eda_product': lf_hf * eda_m,
                'fusion_temp_hrv': t_mean * rmssd,
                'fusion_temp_trend': t_slope * lf_hf,
                'label': 1 if is_stress else 0,
            })
        return rows

    rows = _sample(n_stress, True) + _sample(n_baseline, False)
    df   = pd.DataFrame(rows).sample(frac=1, random_state=random_state).reset_index(drop=True)

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = df[feature_cols].copy()
    y = df['label'].values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    return X, y, "Simulated SWELL-KW (no CSV provided)"


# ─── Dataset resolver: pick the best available source ────────────────────────
def resolve_dataset(feature_cols):
    """
    Priority order:
      1. SWELL_TRAIN_CSV set directly (file already in /content/)
      2. SWELL_DRIVE_PATH  → mount Google Drive and read from there (FASTEST)
      3. Colab upload widget (slow fallback)
      4. Simulation fallback
    """

    # ── 1. Direct /content/ path ─────────────────────────────────────────────
    if SWELL_TRAIN_CSV and os.path.exists(SWELL_TRAIN_CSV):
        print(f"✅ Using SWELL_TRAIN_CSV: {SWELL_TRAIN_CSV}")
        return load_swell_hrv_csv(SWELL_TRAIN_CSV, feature_cols, split_name="train")

    if SWELL_TRAIN_CSV and not os.path.exists(SWELL_TRAIN_CSV):
        print(f"⚠  SWELL_TRAIN_CSV path not found: {SWELL_TRAIN_CSV}")

    # ── 2. Google Drive (recommended for speed) ──────────────────────────────
    drive_path = mount_drive_and_resolve(SWELL_DRIVE_PATH)
    if drive_path:
        return load_swell_hrv_csv(drive_path, feature_cols, split_name="train")

    # ── 3. Colab upload widget ───────────────────────────────────────────────
    print("\n⚠  Could not find train.csv via Drive. Falling back to upload widget.")
    print("   (This is slow. For faster access, upload train.csv to Google Drive")
    print("    and set SWELL_DRIVE_PATH at the top of this file.)\n")
    uploaded_path = colab_upload_csv("train.csv  (from  hrv dataset/final/)")
    if uploaded_path and os.path.exists(uploaded_path):
        return load_swell_hrv_csv(uploaded_path, feature_cols, split_name="uploaded")

    # ── 4. Simulation ────────────────────────────────────────────────────────
    print("\nℹ  No SWELL CSV found or uploaded — running simulation.")
    print("   To use real data:")
    print("     a) Upload train.csv to Google Drive and set SWELL_DRIVE_PATH, OR")
    print("     b) Set SWELL_TRAIN_CSV to the file's /content/ path.\n")
    return generate_swell_simulation(feature_cols)


# ─── Validation plots ─────────────────────────────────────────────────────────
def make_validation_plots(y_test, y_pred, y_proba, data_label,
                          acc, auc, f1, prec, rec):
    fig = plt.figure(figsize=(17, 10))
    fig.patch.set_facecolor('#0a0e1a')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    plt.rcParams.update({
        'axes.facecolor':  '#0d1526',
        'axes.edgecolor':  '#1e3055',
        'axes.labelcolor': '#8aa8d0',
        'xtick.color':     '#5a7ba8',
        'ytick.color':     '#5a7ba8',
        'grid.color':      '#1a2a45',
        'text.color':      '#c8d4e8',
    })

    # 1 — Confusion matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm  = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'High Risk'],
                yticklabels=['Normal', 'High Risk'],
                linewidths=0.5, ax=ax1,
                annot_kws={'color': '#e8f0ff', 'size': 14})
    ax1.set_title(f'Confusion Matrix\n{data_label}', color='#e8f0ff', fontsize=10)
    ax1.set_xlabel('Predicted', fontsize=9)
    ax1.set_ylabel('Actual',    fontsize=9)

    # 2 — ROC curve
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax2.plot(fpr, tpr, color='#3b82f6', linewidth=2.5, label=f'AUC = {auc:.3f}')
    ax2.plot([0, 1], [0, 1], color='#2a4a70', linestyle='--', linewidth=1)
    ax2.fill_between(fpr, tpr, alpha=0.12, color='#3b82f6')
    ax2.set_xlabel('False Positive Rate', fontsize=9)
    ax2.set_ylabel('True Positive Rate',  fontsize=9)
    ax2.set_title(f'ROC Curve — AUC={auc:.3f}', color='#e8f0ff', fontsize=10)
    ax2.legend(fontsize=9, facecolor='#0d1526', edgecolor='#1e3055', labelcolor='#8aa8d0')
    ax2.grid(True, alpha=0.3)

    # 3 — Score distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(y_proba[y_test == 0] * 100, bins=25, alpha=0.75,
             color='#22c55e', label='Normal',    density=True)
    ax3.hist(y_proba[y_test == 1] * 100, bins=25, alpha=0.75,
             color='#ef4444', label='High Risk', density=True)
    ax3.axvline(30, color='#eab308', linestyle='--', linewidth=1.4)
    ax3.axvline(60, color='#ef4444', linestyle='--', linewidth=1.4)
    ax3.set_xlabel('Risk Score (0-100)', fontsize=9)
    ax3.set_ylabel('Density',            fontsize=9)
    ax3.set_title('Score Distribution by Class', color='#e8f0ff', fontsize=10)
    ax3.legend(fontsize=8, facecolor='#0d1526', edgecolor='#1e3055', labelcolor='#8aa8d0')
    ax3.grid(True, alpha=0.3)

    # 4 — SHAP importance
    ax4 = fig.add_subplot(gs[1, 0:2])
    try:
        shap_df = pd.read_csv(SHAP_CSV)
        top     = shap_df.head(15)
        colours = ['#ef4444' if i < 5 else '#3b82f6' if i < 10 else '#5a7ba8'
                   for i in range(len(top))]
        bars = ax4.barh(top['feature'][::-1], top['importance'][::-1],
                        color=colours[::-1], edgecolor='none', height=0.65)
        ax4.set_title('Top 15 Features — SHAP Importance (from training)',
                      color='#e8f0ff', fontsize=10)
        ax4.set_xlabel('Mean |SHAP| Value', fontsize=9)
        ax4.grid(True, alpha=0.2, axis='x')
        for bar, val in zip(bars, top['importance'][::-1]):
            ax4.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                     f'{val:.3f}', va='center', fontsize=7, color='#8aa8d0')
    except FileNotFoundError:
        ax4.text(0.5, 0.5,
                 'shap_importance.csv not found\nRun step1_train.py first',
                 ha='center', va='center', transform=ax4.transAxes,
                 fontsize=10, color='#5a7ba8')

    # 5 — Metrics bar chart
    ax5  = fig.add_subplot(gs[1, 2])
    mvals = [acc, auc, f1, prec, rec]
    mnames = ['Accuracy', 'ROC-AUC', 'F1', 'Precision', 'Recall']
    bc   = ['#22c55e' if v >= 0.8 else '#eab308' if v >= 0.7 else '#ef4444' for v in mvals]
    bars2 = ax5.bar(mnames, mvals, color=bc, edgecolor='none', width=0.55)
    for b, v in zip(bars2, mvals):
        ax5.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=9,
                 color='#e8f0ff', fontweight='bold')
    ax5.set_ylim(0, 1.18)
    ax5.set_title(f'Performance Summary\n{data_label}', color='#e8f0ff', fontsize=10)
    ax5.tick_params(axis='x', rotation=30)
    ax5.axhline(0.80, color='#22c55e', linestyle='--', linewidth=0.9, alpha=0.6)
    ax5.grid(True, alpha=0.2, axis='y')

    fig.suptitle(
        f'Cross-Dataset Validation: WESAD (train) → {data_label}\nBurnout Risk Assessment',
        color='#e8f0ff', fontsize=13, fontweight='bold', y=1.01
    )
    plt.savefig('cross_validation_results.png', dpi=150,
                bbox_inches='tight', facecolor='#0a0e1a')
    plt.show()
    print("✅ cross_validation_results.png saved")


# ─── Main validation runner ───────────────────────────────────────────────────
def run_cross_validation():
    print("=" * 60)
    print("  CROSS-DATASET VALIDATION: WESAD (train) → SWELL HRV (test)")
    print("=" * 60)

    model, scaler, feature_cols = load_artifacts()

    # ── Resolve train split ──────────────────────────────────────────────────
    X_test, y_test, data_label = resolve_dataset(feature_cols)

    print(f"\n   Data source : {data_label}")
    print(f"   Samples     : {len(y_test):,}  |  "
          f"stress={int(y_test.sum())}  normal={int((y_test == 0).sum())}")

    X_test_sc = scaler.transform(X_test)
    y_pred    = model.predict(X_test_sc)
    y_proba   = model.predict_proba(X_test_sc)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)

    print(f"\n{'─'*55}")
    print(f"Classification Report  [{data_label}]:")
    print(classification_report(y_test, y_pred,
                                target_names=['Normal', 'High Risk'],
                                zero_division=0))
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"{'─'*55}")

    make_validation_plots(y_test, y_pred, y_proba, data_label,
                          acc, auc, f1, prec, rec)

    # ── Optional: also evaluate on test.csv ─────────────────────────────────
    if USE_TEST_CSV:
        test_path = None
        if SWELL_TEST_CSV and os.path.exists(SWELL_TEST_CSV):
            test_path = SWELL_TEST_CSV
        else:
            drive_test = mount_drive_and_resolve(
                SWELL_TEST_CSV.replace("/content/drive/MyDrive/", "")
                if SWELL_TEST_CSV else None
            )
            if drive_test:
                test_path = drive_test
            else:
                print("\n📤 Upload test.csv for hold-out evaluation (optional — press Cancel to skip).")
                test_path = colab_upload_csv("test.csv  (from  hrv dataset/final/)")

        if test_path and os.path.exists(test_path):
            X_ho, y_ho, ho_label = load_swell_hrv_csv(test_path, feature_cols, split_name="test")
            X_ho_sc    = scaler.transform(X_ho)
            y_ho_pred  = model.predict(X_ho_sc)
            y_ho_proba = model.predict_proba(X_ho_sc)[:, 1]

            ho_acc = accuracy_score(y_ho, y_ho_pred)
            ho_auc = roc_auc_score(y_ho, y_ho_proba)
            ho_f1  = f1_score(y_ho, y_ho_pred, zero_division=0)

            print(f"\n{'─'*55}")
            print(f"Hold-out Test Set  [{ho_label}]:")
            print(f"  Accuracy : {ho_acc:.4f}")
            print(f"  ROC-AUC  : {ho_auc:.4f}")
            print(f"  F1-Score : {ho_f1:.4f}")
            print(f"{'─'*55}")

    return {
        'accuracy':  acc, 'roc_auc': auc, 'f1':      f1,
        'precision': prec,'recall':  rec, 'data_source': data_label
    }


# ─── Entry point ─────────────────────────────────────────────────────────────
COLAB_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════
║  HOW TO RUN (Colab)
╠══════════════════════════════════════════════════════════════════════
║
║  FASTEST METHOD — Google Drive (recommended):
║
║  1. Open drive.google.com in your browser.
║     Upload  hrv dataset/final/train.csv  there.
║
║  2. In this file, set:
║       SWELL_DRIVE_PATH = "train.csv"
║     (or the subfolder path, e.g. "swell/train.csv")
║
║  3. Run cells in order:
║       %run step0_load_wesad.py   ← ~20-40 min
║       %run step1_train.py         ← ~5 min
║       %run step3_validate.py      ← THIS FILE
║
║  When step3_validate.py runs, Google Drive will mount automatically
║  (you'll see a permission popup — click Allow).
║
║  ALTERNATIVE — Direct path:
║     If train.csv is already at a known path, set:
║       SWELL_TRAIN_CSV = "/content/train.csv"
╚══════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(COLAB_INSTRUCTIONS)

    try:
        results = run_cross_validation()

        print("\n✅ Final Results:")
        for k, v in results.items():
            if isinstance(v, float):
                status = "✅" if v >= 0.80 else "⚠️ " if v >= 0.70 else "❌"
                print(f"   {status} {k:12s}: {v:.4f}")
            else:
                print(f"   ℹ️  {k:12s}: {v}")

    except FileNotFoundError as e:
        print(f"\n❌ {e}")
    except RuntimeError as e:
        print(f"\n❌ Runtime error:\n   {e}")
    except Exception as e:
        import traceback
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}")
        traceback.print_exc()
