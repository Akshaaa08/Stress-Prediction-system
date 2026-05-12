"""
model_utils.py
──────────────
Shared module imported by step1_train.py, step3_validate.py, and dashboard.py.

WHY THIS FILE EXISTS
────────────────────
Python's pickle stores the *class path* of every object it serialises.
When step1_train.py saves `EnsembleModel` to trained_model.pkl, pickle
records the path as  step1_train.EnsembleModel.
When step3_validate.py (a different __main__ module) tries to load that file,
Python cannot find the class and raises:

    AttributeError: Can't get attribute 'EnsembleModel' on <module '__main__' …>

Defining EnsembleModel here and importing it in every script fixes this
permanently — pickle stores the path as  model_utils.EnsembleModel  and
any script that does  `from model_utils import EnsembleModel`  can load it.

USAGE
─────
    from model_utils import EnsembleModel, FEATURE_DEFAULTS
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Ensemble model
# ─────────────────────────────────────────────────────────────────────────────
class EnsembleModel:
    """
    Weighted average ensemble of XGBoost and Random Forest classifiers.

    Parameters
    ----------
    models        : list of fitted sklearn-compatible classifiers
    weights       : list of floats (must sum to 1.0); defaults to equal weights
    feature_medians: pd.Series or dict  — per-feature medians used for
                    inference-time NaN imputation (fitted on training set)
    feature_cols  : list[str]           — ordered feature column names
    """

    def __init__(self, models, weights=None, feature_medians=None, feature_cols=None):
        self.models          = models
        self.weights         = weights if weights is not None else [1 / len(models)] * len(models)
        self.feature_medians = feature_medians   # pd.Series or dict
        self.feature_cols    = feature_cols      # list[str]

    # ── Core prediction ──────────────────────────────────────────────────────
    def predict_proba(self, X):
        """Weighted average of per-model predict_proba. Returns (N, 2) array."""
        proba = np.zeros((X.shape[0], 2))
        for m, w in zip(self.models, self.weights):
            proba += w * m.predict_proba(X)
        return proba

    def predict(self, X, threshold=0.5):
        """Hard classification at given probability threshold."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def predict_risk_score(self, X):
        """
        Returns a 0-100 burnout risk score (stress probability × 100).
        Used by the real-time dashboard.
        """
        return np.clip(self.predict_proba(X)[:, 1] * 100, 0, 100)

    # ── Convenience: predict from a feature dict (single window) ─────────────
    def predict_from_dict(self, feat_dict, scaler):
        """
        Accept a plain {feature_name: value} dict, align to training columns,
        scale, and return (risk_score_0_100, stress_probability).

        Requires self.feature_cols and self.feature_medians to be set
        (they are always set by step1_train.py).
        """
        if self.feature_cols is None:
            raise ValueError("EnsembleModel.feature_cols is not set.")

        row = np.array([[
            feat_dict.get(c, self.feature_medians.get(c, 0.0) if self.feature_medians else 0.0)
            for c in self.feature_cols
        ]])
        row_sc = scaler.transform(row)
        score  = float(self.predict_risk_score(row_sc)[0])
        proba  = float(self.predict_proba(row_sc)[0, 1])
        return score, proba

    def __repr__(self):
        model_names = [type(m).__name__ for m in self.models]
        return (f"EnsembleModel(models={model_names}, "
                f"weights={self.weights}, "
                f"n_features={len(self.feature_cols) if self.feature_cols else '?'})")


# ─────────────────────────────────────────────────────────────────────────────
# Default feature values (used when a feature is missing at inference time)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_DEFAULTS = {
    # HRV time-domain
    'hrv_mean_rr':    850.0,
    'hrv_mean_hr':     70.0,
    'hrv_sdnn':        50.0,
    'hrv_rmssd':       40.0,
    'hrv_pnn50':       18.0,
    'hrv_pnn20':       30.0,
    'hrv_cv':           0.06,
    'hrv_skewness':     0.0,
    'hrv_kurtosis':     0.0,
    'hrv_tinn':       400.0,
    'hrv_sdsd':        30.0,
    # HRV frequency-domain
    'hrv_lf':         400.0,
    'hrv_hf':         250.0,
    'hrv_vlf':        150.0,
    'hrv_tp':         800.0,
    'hrv_lf_hf':        2.0,
    'hrv_lf_norm':      0.5,
    'hrv_hf_norm':      0.5,
    # HRV non-linear
    'hrv_sd1':         28.0,
    'hrv_sd2':         60.0,
    'hrv_sd1_sd2':      0.47,
    'hrv_sampen':       1.8,
    # HRV trend
    'hrv_rr_slope':     0.0,
    'hrv_stability':    1.0,
    # EDA
    'eda_scl_mean':     3.0,
    'eda_scl_std':      0.5,
    'eda_scl_range':    1.0,
    'eda_scl_min':      2.0,
    'eda_scl_max':      4.0,
    'eda_scr_mean':     0.1,
    'eda_scr_std':      0.05,
    'eda_scr_max':      0.3,
    'eda_scr_energy':   0.5,
    'eda_peak_count':   3.0,
    'eda_peak_rate':    3.0,
    'eda_peak_amp_mean':0.1,
    'eda_peak_amp_std': 0.05,
    'eda_peak_amp_max': 0.3,
    'eda_raw_mean':     3.0,
    'eda_raw_std':      0.5,
    'eda_raw_skew':     0.2,
    'eda_raw_kurt':     0.3,
    'eda_trend_slope':  0.0,
    # Temperature
    'temp_mean':       35.0,
    'temp_std':         0.2,
    'temp_min':        34.5,
    'temp_max':        35.5,
    'temp_range':       1.0,
    'temp_slope':       0.0,
    # Fusion
    'fusion_autonomic_balance':     6.0,
    'fusion_parasympathetic_load':  20.0,
    'fusion_stress_index':           2.5,
    'fusion_recovery_capacity':     20.0,
    'fusion_scr_hrv_coupling':       0.01,
    'fusion_lfhf_eda_product':       6.0,
    'fusion_temp_hrv':            1190.0,
    'fusion_temp_trend':             0.0,
}
