"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BURNOUT RISK ASSESSMENT — STEP 0: WESAD DATA LOADING & FEATURE ENGINEERING║
║   Run this in Google Colab FIRST, before step1_train.py                     ║
║                                                                              ║
║   What this does:                                                            ║
║   1. Mounts Google Drive                                                     ║
║   2. Loads all 15 WESAD subjects (S2–S17) from Drive                        ║
║   3. Preprocesses raw ECG (700 Hz), EDA (4 Hz), TEMP (4 Hz) signals        ║
║   4. Extracts HRV, EDA, TEMP, and fusion features per 60-s window           ║
║   5. Saves  wesad_features.csv  →  read by step1_train.py                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

COLAB SETUP
───────────
Paste each block below into a separate cell and run top to bottom.

  CELL 1 — install
    !pip install neurokit2 xgboost shap scikit-learn tqdm scipy -q

  CELL 2 — mount drive
    from google.colab import drive
    drive.mount('/content/drive')

  CELL 3 — set paths and run
    import os
    os.environ["WESAD_PATH"] = "/content/drive/MyDrive/WESAD"
    os.environ["OUTPUT_CSV"] = "/content/wesad_features.csv"
    %run step0_load_wesad.py

  Expected WESAD folder layout on Drive:
    WESAD/
      S2/S2.pkl
      S3/S3.pkl
      ...
      S17/S17.pkl
"""

import numpy as np
import pandas as pd
import pickle, os, glob, warnings
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
import neurokit2 as nk
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
print("✅ Imports OK")


# ─── Default paths (overridden by env vars set in Colab) ─────────────────────
WESAD_PATH = os.environ.get("WESAD_PATH", "/content/drive/MyDrive/WESAD")
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "/content/wesad_features.csv")


# ─── Configuration ───────────────────────────────────────────────────────────
class Config:
    ECG_SR       = 700   # Hz — RespiBAN chest device
    EDA_SR       = 4     # Hz — Empatica E4 wrist device
    TEMP_SR      = 4     # Hz — Empatica E4
    WINDOW_SEC   = 60    # analysis window length (seconds)
    STEP_SEC     = 30    # 50 % overlap → one new window every 30 s

    # WESAD labels: 0=transient/init, 1=baseline, 2=stress, 3=amusement, 4=meditation
    STRESS_LABEL   = [2]      # burnout proxy  → label = 1
    BASELINE_LABEL = [1, 3]   # normal/resting → label = 0
    # Label 4 (meditation) and 0 (transient) are discarded

    ECG_LOWCUT  = 0.5
    ECG_HIGHCUT = 40.0
    NOTCH_FREQ  = 50.0   # EU powerline; change to 60.0 for North American recordings
    EDA_LOWPASS = 5.0

cfg = Config()
print("✅ Config ready")


# ─── Signal preprocessing ────────────────────────────────────────────────────
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq  = 0.5 * fs
    low  = max(lowcut / nyq, 1e-4)
    high = min(highcut / nyq, 0.999)
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    return scipy_signal.filtfilt(b, a, data)

def notch_filter(data, freq, fs, Q=30):
    b, a = scipy_signal.iirnotch(freq / (0.5 * fs), Q)
    return scipy_signal.filtfilt(b, a, data)

def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = scipy_signal.butter(order, min(cutoff / nyq, 0.999), btype='low')
    return scipy_signal.filtfilt(b, a, data)

def preprocess_ecg(ecg_raw, fs=700):
    ecg = bandpass_filter(ecg_raw, cfg.ECG_LOWCUT, cfg.ECG_HIGHCUT, fs)
    ecg = notch_filter(ecg, cfg.NOTCH_FREQ, fs)
    return (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)

def preprocess_eda(eda_raw, fs=4):
    eda = lowpass_filter(eda_raw, cfg.EDA_LOWPASS, fs)
    return np.clip(eda, 0, None)

def preprocess_temp(temp_raw, fs=4):
    temp = scipy_signal.detrend(temp_raw.astype(float))
    return lowpass_filter(temp, 0.5, fs)


# ─── Feature extraction — HRV ────────────────────────────────────────────────
def extract_hrv_features(ecg_segment, fs=700):
    feats = {}
    try:
        ecg_signals, info = nk.ecg_process(ecg_segment, sampling_rate=fs)
        rpeaks = info['ECG_R_Peaks']
        if len(rpeaks) < 4:
            return None
        rr_ms = np.diff(rpeaks) / fs * 1000
        rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
        if len(rr_ms) < 3:
            return None

        feats['hrv_mean_rr']  = float(np.mean(rr_ms))
        feats['hrv_mean_hr']  = 60000.0 / feats['hrv_mean_rr']
        feats['hrv_sdnn']     = float(np.std(rr_ms, ddof=1))
        feats['hrv_rmssd']    = float(np.sqrt(np.mean(np.diff(rr_ms)**2)))
        feats['hrv_pnn50']    = float(np.sum(np.abs(np.diff(rr_ms)) > 50) / max(len(rr_ms)-1, 1) * 100)
        feats['hrv_pnn20']    = float(np.sum(np.abs(np.diff(rr_ms)) > 20) / max(len(rr_ms)-1, 1) * 100)
        feats['hrv_cv']       = feats['hrv_sdnn'] / feats['hrv_mean_rr']
        feats['hrv_skewness'] = float(skew(rr_ms))
        feats['hrv_kurtosis'] = float(kurtosis(rr_ms))
        feats['hrv_tinn']     = float(np.max(rr_ms) - np.min(rr_ms))
        diff_rr = np.diff(rr_ms)
        feats['hrv_sdsd']     = float(np.std(diff_rr, ddof=1))

        try:
            hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=fs, show=False, normalize=True)
            feats['hrv_lf']      = float(hrv_freq['HRV_LF'].iloc[0])
            feats['hrv_hf']      = float(hrv_freq['HRV_HF'].iloc[0])
            feats['hrv_vlf']     = float(hrv_freq.get('HRV_VLF', pd.Series([0])).iloc[0])
            feats['hrv_tp']      = feats['hrv_lf'] + feats['hrv_hf'] + feats['hrv_vlf']
            hf_safe              = max(feats['hrv_hf'], 1e-6)
            feats['hrv_lf_hf']  = feats['hrv_lf'] / hf_safe
            total                = feats['hrv_lf'] + feats['hrv_hf'] + 1e-6
            feats['hrv_lf_norm'] = feats['hrv_lf'] / total
            feats['hrv_hf_norm'] = feats['hrv_hf'] / total
        except Exception:
            feats.update({'hrv_lf': 0, 'hrv_hf': 0, 'hrv_vlf': 0, 'hrv_tp': 0,
                          'hrv_lf_hf': 2.0, 'hrv_lf_norm': 0.5, 'hrv_hf_norm': 0.5})

        try:
            hrv_nl = nk.hrv_nonlinear(rpeaks, sampling_rate=fs, show=False)
            feats['hrv_sd1']     = float(hrv_nl['HRV_SD1'].iloc[0])
            feats['hrv_sd2']     = float(hrv_nl['HRV_SD2'].iloc[0])
            feats['hrv_sd1_sd2'] = float(hrv_nl['HRV_SD1SD2'].iloc[0])
            feats['hrv_sampen']  = float(hrv_nl.get('HRV_SampEn', pd.Series([0])).iloc[0])
        except Exception:
            feats.update({'hrv_sd1': 0, 'hrv_sd2': 0, 'hrv_sd1_sd2': 0, 'hrv_sampen': 0})

        if len(rr_ms) >= 6:
            feats['hrv_rr_slope'] = float(np.polyfit(np.arange(len(rr_ms)), rr_ms, 1)[0])
        else:
            feats['hrv_rr_slope'] = 0.0
        feats['hrv_stability'] = np.std(diff_rr) / (np.mean(np.abs(diff_rr)) + 1e-8)

    except Exception:
        return None
    return feats


# ─── Feature extraction — EDA ─────────────────────────────────────────────────
def extract_eda_features(eda_segment, fs=4):
    feats = {}
    try:
        eda_signals, info = nk.eda_process(eda_segment, sampling_rate=fs)
        scl = eda_signals['EDA_Tonic'].values
        scr = eda_signals['EDA_Phasic'].values

        feats['eda_scl_mean']  = float(np.mean(scl))
        feats['eda_scl_std']   = float(np.std(scl))
        feats['eda_scl_range'] = float(np.max(scl) - np.min(scl))
        feats['eda_scl_min']   = float(np.min(scl))
        feats['eda_scl_max']   = float(np.max(scl))
        feats['eda_scr_mean']  = float(np.mean(scr))
        feats['eda_scr_std']   = float(np.std(scr))
        feats['eda_scr_max']   = float(np.max(scr))
        feats['eda_scr_energy']= float(np.sum(scr**2))

        peaks_idx = info.get('SCR_Peaks', np.array([]))
        dur_min   = max(len(eda_segment) / fs / 60, 1e-6)
        feats['eda_peak_count'] = int(len(peaks_idx))
        feats['eda_peak_rate']  = len(peaks_idx) / dur_min

        if len(peaks_idx) > 0:
            pa = scr[peaks_idx]
            feats['eda_peak_amp_mean'] = float(np.mean(pa))
            feats['eda_peak_amp_std']  = float(np.std(pa))
            feats['eda_peak_amp_max']  = float(np.max(pa))
        else:
            feats.update({'eda_peak_amp_mean': 0.0, 'eda_peak_amp_std': 0.0, 'eda_peak_amp_max': 0.0})

        feats['eda_raw_mean']  = float(np.mean(eda_segment))
        feats['eda_raw_std']   = float(np.std(eda_segment))
        feats['eda_raw_skew']  = float(skew(eda_segment))
        feats['eda_raw_kurt']  = float(kurtosis(eda_segment))
        x = np.arange(len(scr))
        feats['eda_trend_slope'] = float(np.polyfit(x, scr, 1)[0]) if len(x) > 1 else 0.0

    except Exception:
        feats = {
            'eda_scl_mean': float(np.mean(eda_segment)), 'eda_scl_std': float(np.std(eda_segment)),
            'eda_scl_range': float(np.ptp(eda_segment)),
            'eda_scl_min': float(np.min(eda_segment)), 'eda_scl_max': float(np.max(eda_segment)),
            'eda_scr_mean': 0.0, 'eda_scr_std': 0.0, 'eda_scr_max': 0.0, 'eda_scr_energy': 0.0,
            'eda_peak_count': 0, 'eda_peak_rate': 0.0,
            'eda_peak_amp_mean': 0.0, 'eda_peak_amp_std': 0.0, 'eda_peak_amp_max': 0.0,
            'eda_raw_mean': float(np.mean(eda_segment)), 'eda_raw_std': float(np.std(eda_segment)),
            'eda_raw_skew': 0.0, 'eda_raw_kurt': 0.0, 'eda_trend_slope': 0.0
        }
    return feats


# ─── Feature extraction — TEMP ───────────────────────────────────────────────
def extract_temp_features(temp_segment, fs=4):
    feats = {}
    try:
        feats['temp_mean']  = float(np.mean(temp_segment))
        feats['temp_std']   = float(np.std(temp_segment))
        feats['temp_min']   = float(np.min(temp_segment))
        feats['temp_max']   = float(np.max(temp_segment))
        feats['temp_range'] = feats['temp_max'] - feats['temp_min']
        x = np.arange(len(temp_segment))
        feats['temp_slope'] = float(np.polyfit(x, temp_segment, 1)[0]) if len(x) > 1 else 0.0
    except Exception:
        feats = {'temp_mean': 0.0, 'temp_std': 0.0, 'temp_min': 0.0,
                 'temp_max': 0.0, 'temp_range': 0.0, 'temp_slope': 0.0}
    return feats


# ─── Cross-modal fusion ───────────────────────────────────────────────────────
def extract_fusion_features(hrv, eda, temp=None):
    lf_hf  = hrv.get('hrv_lf_hf', 2.0)
    eda_m  = eda.get('eda_scl_mean', 0.0)
    rmssd  = hrv.get('hrv_rmssd', 30.0)
    sdnn   = hrv.get('hrv_sdnn', 30.0)
    scr_en = eda.get('eda_scr_energy', 0.0)
    fused  = {
        'fusion_autonomic_balance':    lf_hf * (eda_m + 0.1),
        'fusion_parasympathetic_load': rmssd / (eda_m + 1.0),
        'fusion_stress_index':         (lf_hf + 1) / (rmssd / 10 + 1),
        'fusion_recovery_capacity':    sdnn / (lf_hf + 0.1),
        'fusion_scr_hrv_coupling':     scr_en / (rmssd + 1e-6),
        'fusion_lfhf_eda_product':     lf_hf * eda_m,
    }
    if temp is not None:
        t_slope = temp.get('temp_slope', 0.0)
        t_mean  = temp.get('temp_mean', 36.0)
        fused['fusion_temp_hrv']   = t_mean * rmssd
        fused['fusion_temp_trend'] = t_slope * lf_hf
    return fused


# ─── WESAD subject loader ─────────────────────────────────────────────────────
def load_wesad_subject(subject_path):
    """
    Load a single WESAD .pkl file.
    Returns dict with keys: ecg, eda, temp, labels.
    """
    with open(subject_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    chest  = data['signal']['chest']
    wrist  = data['signal']['wrist']
    labels = data['label'].flatten()

    return {
        'ecg':    chest['ECG'].flatten().astype(np.float64),
        'eda':    wrist['EDA'].flatten().astype(np.float64),   # 4 Hz
        'temp':   wrist['TEMP'].flatten().astype(np.float64),  # 4 Hz
        'labels': labels,
    }


# ─── Per-subject windowed processing ─────────────────────────────────────────
def process_subject(subject_path):
    sid     = os.path.basename(os.path.dirname(subject_path))
    records = []
    try:
        subj = load_wesad_subject(subject_path)
    except Exception as e:
        print(f"  ⚠ Could not load {sid}: {e}")
        return records

    ecg = subj['ecg']; eda = subj['eda']
    temp = subj['temp']; labels = subj['labels']

    win_ecg  = cfg.WINDOW_SEC * cfg.ECG_SR
    step_ecg = cfg.STEP_SEC   * cfg.ECG_SR
    win_eda  = cfg.WINDOW_SEC * cfg.EDA_SR
    step_eda = cfg.STEP_SEC   * cfg.EDA_SR
    win_tmp  = cfg.WINDOW_SEC * cfg.TEMP_SR
    step_tmp = cfg.STEP_SEC   * cfg.TEMP_SR

    n_windows = (len(ecg) - win_ecg) // step_ecg

    for i in range(int(n_windows)):
        s_ecg = i * step_ecg; e_ecg = s_ecg + win_ecg
        s_eda = i * step_eda; e_eda = s_eda + win_eda
        s_tmp = i * step_tmp; e_tmp = s_tmp + win_tmp

        if e_ecg > len(ecg) or e_eda > len(eda):
            break

        win_labels = labels[s_ecg:e_ecg]
        unique, counts = np.unique(win_labels, return_counts=True)
        majority = int(unique[np.argmax(counts)])

        if   majority in cfg.STRESS_LABEL:   y = 1
        elif majority in cfg.BASELINE_LABEL: y = 0
        else: continue

        ecg_c = preprocess_ecg(ecg[s_ecg:e_ecg], cfg.ECG_SR)
        eda_c = preprocess_eda(eda[s_eda:e_eda], cfg.EDA_SR)
        has_t = (e_tmp <= len(temp))
        tmp_c = preprocess_temp(temp[s_tmp:e_tmp], cfg.TEMP_SR) if has_t else None

        hrv_f = extract_hrv_features(ecg_c, cfg.ECG_SR)
        if hrv_f is None:
            continue

        eda_f  = extract_eda_features(eda_c, cfg.EDA_SR)
        tmp_f  = extract_temp_features(tmp_c, cfg.TEMP_SR) if tmp_c is not None else {}
        fuse_f = extract_fusion_features(hrv_f, eda_f, tmp_f if tmp_f else None)

        row = {**hrv_f, **eda_f, **tmp_f, **fuse_f,
               'subject': sid, 'window': i, 'label': y}
        records.append(row)

    return records


# ─── Build full dataset ───────────────────────────────────────────────────────
def build_wesad_feature_dataset(wesad_root, max_subjects=None, output_csv=None):
    print(f"\n{'='*60}")
    print("  WESAD FEATURE EXTRACTION")
    print(f"  Root   : {wesad_root}")
    print(f"  Window : {cfg.WINDOW_SEC}s  |  Step: {cfg.STEP_SEC}s (50% overlap)")
    print(f"  Signals: ECG ({cfg.ECG_SR} Hz), EDA ({cfg.EDA_SR} Hz), TEMP ({cfg.TEMP_SR} Hz)")
    print(f"{'='*60}")

    if not os.path.isdir(wesad_root):
        raise FileNotFoundError(
            f"WESAD root not found: {wesad_root}\n"
            "Check Google Drive is mounted and the path is correct.\n"
            "Expected layout:  WESAD/S2/S2.pkl,  WESAD/S3/S3.pkl  ..."
        )

    subject_dirs = sorted(glob.glob(os.path.join(wesad_root, 'S[0-9]*')))
    if not subject_dirs:
        raise FileNotFoundError(f"No S## folders found in {wesad_root}")

    if max_subjects:
        subject_dirs = subject_dirs[:max_subjects]

    print(f"\n  Found {len(subject_dirs)} subject(s): "
          f"{[os.path.basename(d) for d in subject_dirs]}\n")

    all_records = []
    for sdir in tqdm(subject_dirs, desc="Processing subjects"):
        sid      = os.path.basename(sdir)
        pkl_path = os.path.join(sdir, f'{sid}.pkl')
        if not os.path.exists(pkl_path):
            print(f"  ⚠ {sid}: .pkl not found — skipping")
            continue
        records = process_subject(pkl_path)
        print(f"  {sid}: {len(records)} windows")
        all_records.extend(records)

    if not all_records:
        raise RuntimeError("No features extracted. Check your WESAD_PATH.")

    df = pd.DataFrame(all_records)
    df.dropna(axis=1, how='all', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_cols = [c for c in df.columns if c not in ('subject', 'window', 'label')]
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())

    print(f"\n{'─'*50}")
    print(f"  Total windows : {len(df)}")
    print(f"  Features      : {len(feat_cols)}")
    print(f"  Class balance :\n{df['label'].value_counts().rename({0:'Normal', 1:'Stress'}).to_string()}")
    print(f"  Subjects      : {df['subject'].nunique()}")
    print(f"{'─'*50}")

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Saved → {output_csv}")

    return df


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        from google.colab import drive as _drive
        _drive.mount('/content/drive', force_remount=False)
        print("✅ Google Drive mounted")
    except ImportError:
        print("ℹ  Not in Colab — skipping Drive mount.")

    wesad_path = os.environ.get("WESAD_PATH", WESAD_PATH)
    output_csv = os.environ.get("OUTPUT_CSV", OUTPUT_CSV)

    df = build_wesad_feature_dataset(wesad_path, max_subjects=None, output_csv=output_csv)

    print(f"\n🎉 Step 0 complete!  Shape: {df.shape}")
    print("   Proceed to step1_train.py")
