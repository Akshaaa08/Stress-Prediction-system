"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BURNOUT RISK ASSESSMENT SYSTEM — STEP 2: DASHBOARD                        ║
║   Save this as dashboard.py and run from Colab using localtunnel/ngrok       ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW TO RUN IN COLAB:
    # Cell A — install
    !pip install streamlit neurokit2 xgboost shap plotly -q
    !npm install -g localtunnel -q

    # Cell B — run dashboard
    import subprocess, threading, time
    def run():
        subprocess.run(["streamlit", "run", "dashboard.py",
                        "--server.port=8501", "--server.headless=true"])
    threading.Thread(target=run, daemon=True).start()
    time.sleep(5)
    !lt --port 8501
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle, time, threading, queue, os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# KEY: import EnsembleModel so pickle.load() can find the class definition.
# Without this, loading trained_model.pkl raises AttributeError.
try:
    from model_utils import EnsembleModel, FEATURE_DEFAULTS
except ImportError:
    # Graceful fallback — dashboard still works in heuristic mode
    EnsembleModel    = None
    FEATURE_DEFAULTS = {}

# ─────────────────────────────────────────────────────────────────────────────
# Page config — MUST be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BurnoutSense | Autonomic Risk Monitor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark medical aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0a0e1a;
    color: #c8d4e8;
}

.main { background-color: #0a0e1a; }

/* Hide default header */
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }

/* ── Title bar ── */
.title-bar {
    background: linear-gradient(135deg, #0d1526 0%, #111c35 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 4px 24px rgba(0,100,255,0.08);
}
.title-main {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #e8f0ff;
    letter-spacing: -0.5px;
}
.title-sub {
    font-size: 0.75rem;
    color: #5a7ba8;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 2px;
}
.live-badge {
    background: #0f2d1a;
    border: 1px solid #1a5c32;
    color: #2ecc71;
    font-size: 0.65rem;
    letter-spacing: 2px;
    padding: 4px 10px;
    border-radius: 4px;
    text-transform: uppercase;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(46,204,113,0.3); }
    50%       { box-shadow: 0 0 0 5px rgba(46,204,113,0); }
}

/* ── Metric cards ── */
.metric-card {
    background: #0d1526;
    border: 1px solid #1e3055;
    border-radius: 10px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: #2a4a80; }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-color, #3b82f6);
}
.metric-label {
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #5a7ba8;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #e8f0ff;
    line-height: 1;
}
.metric-unit {
    font-size: 0.7rem;
    color: #5a7ba8;
    margin-top: 4px;
}
.metric-delta {
    font-size: 0.72rem;
    margin-top: 6px;
}
.delta-up   { color: #ef4444; }
.delta-down { color: #22c55e; }
.delta-flat { color: #5a7ba8; }

/* ── Risk gauge container ── */
.gauge-container {
    background: #0d1526;
    border: 1px solid #1e3055;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

/* ── Risk status banner ── */
.status-low {
    background: linear-gradient(135deg, #0a1f0f, #0d2a15);
    border: 1px solid #1a5c32;
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 14px 20px;
    color: #4ade80;
}
.status-moderate {
    background: linear-gradient(135deg, #1a1200, #221a00);
    border: 1px solid #4a3800;
    border-left: 4px solid #eab308;
    border-radius: 8px;
    padding: 14px 20px;
    color: #fde047;
}
.status-high {
    background: linear-gradient(135deg, #1a0505, #2a0808);
    border: 1px solid #5c1a1a;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 14px 20px;
    color: #fca5a5;
    animation: pulse-red 3s infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.2); }
    50%       { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
}
.status-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 4px;
}
.status-body {
    font-size: 0.78rem;
    opacity: 0.85;
    line-height: 1.5;
}

/* ── Suggestion cards ── */
.suggestion-card {
    background: #0d1828;
    border: 1px solid #1e3055;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.78rem;
    line-height: 1.5;
    color: #8aa8d0;
    display: flex;
    gap: 10px;
    align-items: flex-start;
}
.suggestion-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e3055;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #080d1a;
    border-right: 1px solid #1e3055;
}
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* ── Streamlit overrides ── */
.stButton > button {
    background: #1a3a6b;
    border: 1px solid #2a5aab;
    color: #c8d4e8;
    border-radius: 6px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    padding: 0.4rem 1rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #1e4a8a;
    border-color: #4a8ae8;
    color: #e8f0ff;
}
div[data-testid="stMetricValue"] { color: #e8f0ff; }
.stPlotlyChart { background: transparent; }

/* ── Window timeline ── */
.window-pill {
    display: inline-block;
    width: 8px; height: 24px;
    border-radius: 3px;
    margin: 1px;
    vertical-align: middle;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e3055; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
RISK_COLORS = {
    'low':      '#22c55e',
    'moderate': '#eab308',
    'high':     '#ef4444'
}

def risk_level(score):
    if score < 35:   return 'low'
    if score < 62:   return 'moderate'
    return 'high'

def risk_label(score):
    lvl = risk_level(score)
    return {'low': 'LOW RISK', 'moderate': 'ELEVATED', 'high': 'HIGH RISK'}[lvl]

def load_artifacts():
    """Load trained model, scaler, and feature columns.
    EnsembleModel must be importable from model_utils for pickle to work."""
    try:
        with open('trained_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        return model, scaler, feature_cols, True
    except FileNotFoundError:
        return None, None, None, False
    except (AttributeError, ImportError) as e:
        # This happens if model_utils.py is not present
        st.sidebar.error(
            f"❌ Model load failed: {e}\n\n"
            "Make sure model_utils.py is in the same folder as dashboard.py, "
            "then re-run step1_train.py to regenerate trained_model.pkl."
        )
        return None, None, None, False


# ─────────────────────────────────────────────────────────────────────────────
# Physiological signal generators (simulation)
# ─────────────────────────────────────────────────────────────────────────────
def simulate_ecg_segment(duration_sec=5, fs=250, stress_level=0.5):
    """Generate synthetic ECG with realistic HRV based on stress level."""
    t = np.linspace(0, duration_sec, int(duration_sec * fs))
    # Heart rate varies with stress (60-100 bpm)
    hr   = 60 + 40 * stress_level
    rr   = 60 / hr
    noise = np.random.normal(0, 0.02, len(t))
    ecg = np.zeros(len(t))
    # QRS complex spikes
    beat_times = np.arange(0, duration_sec, rr * (1 + np.random.normal(0, 0.04 * (1 - stress_level))))
    for bt in beat_times:
        idx = int(bt * fs)
        if idx < len(ecg) - 10:
            ecg[idx:idx+5]   += np.array([0.1, 0.5, 1.5, -0.3, 0.1]) * (0.8 + 0.4 * np.random.randn())
    # P and T waves
    ecg = np.convolve(ecg, np.hanning(15)/np.hanning(15).sum(), mode='same')
    ecg += noise
    return t, ecg


def simulate_eda_segment(duration_sec=30, fs=4, stress_level=0.5):
    """Generate synthetic EDA (skin conductance) based on stress level."""
    t = np.linspace(0, duration_sec, int(duration_sec * fs))
    # Tonic: baseline SCL higher under stress
    scl_base = 1.5 + 5.0 * stress_level + np.random.normal(0, 0.3)
    tonic = scl_base + 0.1 * np.sin(2 * np.pi * 0.05 * t)
    # Phasic: SCR peaks
    phasic = np.zeros(len(t))
    n_peaks = int(2 + 8 * stress_level)
    for _ in range(n_peaks):
        peak_t = np.random.uniform(2, duration_sec - 3)
        amp    = np.random.uniform(0.1, 0.5 + 0.8 * stress_level)
        rise   = 1.5  # seconds
        decay  = 5.0
        for i, ti in enumerate(t):
            dt = ti - peak_t
            if 0 < dt < 15:
                phasic[i] += amp * (np.exp(-dt / decay) - np.exp(-dt / (rise * 0.3)))
    eda = tonic + np.clip(phasic, 0, None) + np.random.normal(0, 0.05, len(t))
    return t, np.clip(eda, 0, None)


def build_realtime_features(stress_level, window_id, prev_rmssd=None):
    """Generate a complete feature vector for one simulation window.
    Includes all features added in step0 (pnn20, tinn, sdsd, sd1/sd2,
    sampen, scr_energy, temp_*, fusion_* extended) so scaler.transform
    never receives missing columns."""
    sl = stress_level
    drift  = min(window_id * 0.015, 0.3)
    sl_eff = min(sl + drift, 1.0)

    rng     = np.random.default_rng(seed=window_id * 7 + int(sl * 100))
    mean_rr = float(np.clip(rng.normal(880 - 200*sl_eff, 40), 400, 1400))
    sdnn    = float(np.clip(rng.normal(65 - 35*sl_eff, 8), 5, 200))
    rmssd   = float(np.clip(rng.normal(55 - 38*sl_eff, 7), 5, 150))
    pnn50   = float(max(0, rng.normal(25 - 20*sl_eff, 4)))
    pnn20   = float(max(0, rng.normal(40 - 30*sl_eff, 6)))
    lf_hf   = float(np.clip(rng.normal(1.5 + 3.5*sl_eff, 0.5), 0.3, 12))
    lf      = float(max(10, rng.normal(300 + 400*sl_eff, 60)))
    hf      = float(max(10, rng.normal(280 - 150*sl_eff, 50)))
    eda_m   = float(np.clip(rng.normal(1.5 + 6.0*sl_eff, 0.8), 0, 20))
    eda_pk  = float(max(0, rng.poisson(2 + 9*sl_eff)))
    rr_slope= float(rng.normal(0.4 - 1.0*sl_eff, 0.15))
    total   = lf + hf
    sdsd    = float(np.clip(rng.normal(20*(1+sl_eff), 5), 1, 100))
    sd1     = rmssd / np.sqrt(2)
    sd2     = float(np.sqrt(max(2*sdnn**2 - sd1**2, 1e-6)))
    t_mean  = float(rng.normal(35.5 - sl_eff, 0.4))
    t_slope = float(rng.normal(-0.002*sl_eff, 0.001))

    feats = {
        # HRV time-domain
        'hrv_mean_rr':    mean_rr,
        'hrv_mean_hr':    60000 / mean_rr,
        'hrv_sdnn':       sdnn,
        'hrv_rmssd':      rmssd,
        'hrv_pnn50':      pnn50,
        'hrv_pnn20':      pnn20,
        'hrv_cv':         sdnn / mean_rr,
        'hrv_skewness':   float(rng.normal(0.1*sl_eff, 0.3)),
        'hrv_kurtosis':   float(rng.normal(0.3, 0.3)),
        'hrv_tinn':       float(rng.normal(350 - 100*sl_eff, 50)),
        'hrv_sdsd':       sdsd,
        # HRV frequency-domain
        'hrv_lf':         lf,
        'hrv_hf':         hf,
        'hrv_vlf':        float(rng.normal(150 + 100*sl_eff, 40)),
        'hrv_tp':         lf + hf + float(rng.normal(150, 40)),
        'hrv_lf_hf':      lf_hf,
        'hrv_lf_norm':    lf / total,
        'hrv_hf_norm':    hf / total,
        # HRV non-linear
        'hrv_sd1':        sd1,
        'hrv_sd2':        sd2,
        'hrv_sd1_sd2':    sd1 / max(sd2, 1e-6),
        'hrv_sampen':     float(rng.normal(2.0 - 0.6*sl_eff, 0.3)),
        # HRV trend
        'hrv_rr_slope':   rr_slope,
        'hrv_stability':  float(rng.normal(0.8 + 0.9*sl_eff, 0.2)),
        # EDA
        'eda_scl_mean':   eda_m,
        'eda_scl_std':    eda_m * float(rng.uniform(0.1, 0.25)),
        'eda_scl_range':  eda_m * float(rng.uniform(0.2, 0.4)),
        'eda_scl_min':    max(0, eda_m * float(rng.uniform(0.5, 0.8))),
        'eda_scl_max':    eda_m * float(rng.uniform(1.2, 1.6)),
        'eda_scr_mean':   float(rng.uniform(0.05, 0.4*sl_eff + 0.1)),
        'eda_scr_std':    float(rng.uniform(0.02, 0.15)),
        'eda_scr_max':    float(rng.uniform(0.1, 0.8*sl_eff + 0.2)),
        'eda_scr_energy': float(rng.uniform(0.2, 2.5*sl_eff + 0.3)),
        'eda_peak_count': eda_pk,
        'eda_peak_rate':  eda_pk,
        'eda_peak_amp_mean': float(rng.uniform(0.05, 0.4*sl_eff + 0.1)),
        'eda_peak_amp_std':  float(rng.uniform(0.02, 0.1)),
        'eda_peak_amp_max':  float(rng.uniform(0.1, 0.7*sl_eff + 0.2)),
        'eda_raw_mean':   eda_m + float(rng.normal(0, 0.15)),
        'eda_raw_std':    eda_m * float(rng.uniform(0.05, 0.15)),
        'eda_raw_skew':   float(rng.normal(0.4*sl_eff, 0.2)),
        'eda_raw_kurt':   float(rng.normal(0.5*sl_eff, 0.2)),
        'eda_trend_slope': float(rng.normal(0.008*sl_eff, 0.003)),
        # Temperature
        'temp_mean':      t_mean,
        'temp_std':       float(rng.uniform(0.05, 0.3)),
        'temp_min':       t_mean - float(rng.uniform(0.1, 0.4)),
        'temp_max':       t_mean + float(rng.uniform(0.1, 0.4)),
        'temp_range':     float(rng.uniform(0.2, 0.8)),
        'temp_slope':     t_slope,
        # Fusion
        'fusion_autonomic_balance':    lf_hf * (eda_m + 0.1),
        'fusion_parasympathetic_load': rmssd / (eda_m + 1.0),
        'fusion_stress_index':         (lf_hf + 1) / (rmssd / 10 + 1),
        'fusion_recovery_capacity':    sdnn / (lf_hf + 0.1),
        'fusion_scr_hrv_coupling':     float(rng.uniform(0.2, 2.0*sl_eff + 0.2)) / (rmssd + 1e-6),
        'fusion_lfhf_eda_product':     lf_hf * eda_m,
        'fusion_temp_hrv':             t_mean * rmssd,
        'fusion_temp_trend':           t_slope * lf_hf,
    }
    return feats, rmssd


# ─────────────────────────────────────────────────────────────────────────────
# Plotly chart helpers
# ─────────────────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,21,38,0.6)',
    font=dict(family='DM Mono, monospace', color='#8aa8d0', size=10),
    margin=dict(l=50, r=20, t=30, b=40),
    xaxis=dict(gridcolor='#1a2a45', linecolor='#1e3055', showgrid=True, zeroline=False),
    yaxis=dict(gridcolor='#1a2a45', linecolor='#1e3055', showgrid=True, zeroline=False),
)

def make_risk_gauge(score):
    """Radial gauge for burnout risk score."""
    lvl   = risk_level(score)
    color = RISK_COLORS[lvl]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={'reference': 50, 'increasing': {'color': '#ef4444'},
               'decreasing': {'color': '#22c55e'}, 'font': {'size': 13}},
        number={'font': {'size': 42, 'family': 'Syne, sans-serif', 'color': '#e8f0ff'},
                'suffix': ''},
        title={'text': f"BURNOUT RISK<br><span style='font-size:11px;color:{color};letter-spacing:2px'>{risk_label(score)}</span>",
               'font': {'size': 12, 'color': '#5a7ba8', 'family': 'DM Mono, monospace'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 0.5,
                     'tickcolor': '#2a4a70', 'tickfont': {'size': 9},
                     'tickvals': [0, 35, 62, 100],
                     'ticktext': ['0', '35', '62', '100']},
            'bar': {'color': color, 'thickness': 0.22},
            'bgcolor': '#0d1526',
            'borderwidth': 0,
            'steps': [
                {'range': [0,  35], 'color': 'rgba(34,197,94,0.08)'},
                {'range': [35, 62], 'color': 'rgba(234,179,8,0.08)'},
                {'range': [62, 100],'color': 'rgba(239,68,68,0.08)'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 2},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Mono, monospace', color='#8aa8d0'),
        height=240,
        margin=dict(l=20, r=20, t=40, b=10)
    )
    return fig


def make_risk_trend(history_df):
    """Area chart of risk score over time."""
    fig = go.Figure()
    if len(history_df) == 0:
        return fig

    t = list(range(len(history_df)))
    scores = history_df['risk_score'].values

    # Colored area segments
    for i in range(len(scores) - 1):
        c = RISK_COLORS[risk_level(scores[i])]
        fig.add_trace(go.Scatter(
            x=[t[i], t[i+1]], y=[scores[i], scores[i+1]],
            mode='lines', line=dict(color=c, width=2),
            showlegend=False, hoverinfo='skip'
        ))

    # Filled area
    fig.add_trace(go.Scatter(
        x=t, y=scores,
        fill='tozeroy',
        fillcolor='rgba(59,130,246,0.06)',
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False, hoverinfo='skip'
    ))

    # Threshold lines
    fig.add_hline(y=35, line_dash='dot', line_color='rgba(34,197,94,0.4)', line_width=1)
    fig.add_hline(y=62, line_dash='dot', line_color='rgba(239,68,68,0.4)', line_width=1)

    fig.add_annotation(x=0, y=35, text="LOW / MED",
                       showarrow=False, font=dict(size=8, color='rgba(34,197,94,0.5)'),
                       xanchor='left')
    fig.add_annotation(x=0, y=62, text="MED / HIGH",
                       showarrow=False, font=dict(size=8, color='rgba(239,68,68,0.5)'),
                       xanchor='left')

    fig.update_layout(
        **PLOT_LAYOUT,
        height=200,
        title=dict(text='Risk Score Timeline', font=dict(size=11), x=0),
        yaxis=dict(**PLOT_LAYOUT['yaxis'], range=[0, 105]),
        showlegend=False
    )
    return fig


def make_hrv_chart(history_df):
    """Dual-axis HRV chart: RMSSD and LF/HF ratio."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    t = list(range(len(history_df)))

    if len(history_df) == 0:
        return fig

    fig.add_trace(go.Scatter(
        x=t, y=history_df['rmssd'],
        name='RMSSD (ms)', mode='lines+markers',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=4),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=t, y=history_df['lf_hf'],
        name='LF/HF', mode='lines+markers',
        line=dict(color='#f97316', width=2, dash='dot'),
        marker=dict(size=4),
    ), secondary_y=True)

    fig.update_yaxes(title_text="RMSSD (ms)",
                     title_font=dict(size=9), secondary_y=False,
                     gridcolor='#1a2a45', linecolor='#1e3055')
    fig.update_yaxes(title_text="LF/HF ratio",
                     title_font=dict(size=9), secondary_y=True,
                     gridcolor='rgba(0,0,0,0)', linecolor='#1e3055')

    fig.update_layout(
        **PLOT_LAYOUT,
        height=220,
        title=dict(text='HRV Metrics', font=dict(size=11), x=0),
        legend=dict(orientation='h', y=1.1, x=0,
                    font=dict(size=9), bgcolor='rgba(0,0,0,0)')
    )
    return fig


def make_eda_chart(history_df):
    """EDA signal: mean SCL and peak rate."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    t = list(range(len(history_df)))

    if len(history_df) == 0:
        return fig

    fig.add_trace(go.Scatter(
        x=t, y=history_df['eda_mean'],
        name='Skin Conductance (μS)', mode='lines',
        fill='tozeroy', fillcolor='rgba(16,185,129,0.06)',
        line=dict(color='#10b981', width=2),
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=t, y=history_df['eda_peaks'],
        name='SCR peaks/min',
        marker_color='rgba(249,115,22,0.4)',
        marker_line_color='#f97316',
        marker_line_width=0.5,
    ), secondary_y=True)

    fig.update_yaxes(title_text="SCL (μS)",
                     title_font=dict(size=9), secondary_y=False,
                     gridcolor='#1a2a45')
    fig.update_yaxes(title_text="SCR peaks/min",
                     title_font=dict(size=9), secondary_y=True,
                     gridcolor='rgba(0,0,0,0)')

    fig.update_layout(
        **PLOT_LAYOUT,
        height=220,
        title=dict(text='EDA — Skin Conductance', font=dict(size=11), x=0),
        legend=dict(orientation='h', y=1.1, x=0,
                    font=dict(size=9), bgcolor='rgba(0,0,0,0)')
    )
    return fig


def make_feature_radar(feats_dict):
    """Radar chart of normalized key features."""
    # Normalize to 0-100 for display
    categories = ['HR', 'RMSSD↓', 'LF/HF↑', 'EDA↑', 'SCR Peaks↑', 'Stress Idx↑']
    hr_n    = (feats_dict.get('hrv_mean_hr',   70)  - 40) / 60 * 100
    rmssd_n = (1 - feats_dict.get('hrv_rmssd', 40) / 100) * 100
    lfhf_n  = feats_dict.get('hrv_lf_hf', 2) / 10 * 100
    eda_n   = feats_dict.get('eda_scl_mean', 3) / 12 * 100
    scr_n   = feats_dict.get('eda_peak_rate', 3) / 15 * 100
    si_n    = feats_dict.get('fusion_stress_index', 2) / 8 * 100

    vals = [np.clip(v, 0, 100) for v in [hr_n, rmssd_n, lfhf_n, eda_n, scr_n, si_n]]
    vals += [vals[0]]
    cats  = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats,
        fill='toself',
        fillcolor='rgba(239,68,68,0.12)',
        line=dict(color='#ef4444', width=1.5),
        marker=dict(size=5, color='#ef4444')
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(13,21,38,0.8)',
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor='#1e3055', tickfont=dict(size=7)),
            angularaxis=dict(gridcolor='#1e3055', tickfont=dict(size=9, color='#8aa8d0'))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=240,
        margin=dict(l=50, r=50, t=30, b=30),
        title=dict(text='Autonomic Profile', font=dict(size=11, color='#8aa8d0'), x=0.5)
    )
    return fig


def make_ecg_waveform(ecg_data, fs=250):
    """Live ECG waveform strip."""
    t = np.arange(len(ecg_data)) / fs
    fig = go.Figure(go.Scatter(
        x=t, y=ecg_data,
        mode='lines',
        line=dict(color='#22c55e', width=1.2),
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        height=150,
        title=dict(text='ECG Waveform (simulated)', font=dict(size=10), x=0),
        xaxis=dict(**PLOT_LAYOUT['xaxis'], title='Time (s)', title_font=dict(size=9)),
        yaxis=dict(**PLOT_LAYOUT['yaxis'], title='mV', title_font=dict(size=9)),
        showlegend=False
    )
    return fig


def make_eda_waveform(eda_data, fs=4):
    """Live EDA waveform strip."""
    t = np.arange(len(eda_data)) / fs
    fig = go.Figure(go.Scatter(
        x=t, y=eda_data,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(16,185,129,0.08)',
        line=dict(color='#10b981', width=1.2),
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        height=150,
        title=dict(text='EDA Waveform (simulated)', font=dict(size=10), x=0),
        xaxis=dict(**PLOT_LAYOUT['xaxis'], title='Time (s)', title_font=dict(size=9)),
        yaxis=dict(**PLOT_LAYOUT['yaxis'], title='μS', title_font=dict(size=9)),
        showlegend=False
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        'running':        False,
        'window_count':   0,
        'history':        [],
        'current_score':  0,
        'current_feats':  {},
        'prev_rmssd':     None,
        'rolling_state':  [],  # last N scores for BurnoutState tracking
        'ecg_buffer':     np.zeros(250 * 5),
        'eda_buffer':     np.zeros(4 * 30),
        'alert_log':      [],
        'model':          None,
        'scaler':         None,
        'feature_cols':   None,
        'model_loaded':   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# Load model once
if not st.session_state.model_loaded:
    model, scaler, fcols, ok = load_artifacts()
    if ok:
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.feature_cols = fcols
        st.session_state.model_loaded = True


# ─────────────────────────────────────────────────────────────────────────────
# BurnoutState: rolling window tracker
# ─────────────────────────────────────────────────────────────────────────────
ROLLING_N = 8  # number of windows to track for sustained risk

def update_burnout_state(new_score):
    """Track sustained risk. Returns smoothed score and sustained flag."""
    st.session_state.rolling_state.append(new_score)
    if len(st.session_state.rolling_state) > ROLLING_N:
        st.session_state.rolling_state.pop(0)

    smoothed = np.mean(st.session_state.rolling_state)
    sustained_high = (
        len(st.session_state.rolling_state) >= 5 and
        np.mean(st.session_state.rolling_state[-5:]) >= 62
    )
    return smoothed, sustained_high


# ─────────────────────────────────────────────────────────────────────────────
# Coping recommendations
# ─────────────────────────────────────────────────────────────────────────────
def get_recommendations(score, feats, sustained):
    recs = []
    lvl  = risk_level(score)
    rmssd   = feats.get('hrv_rmssd', 40)
    lf_hf   = feats.get('hrv_lf_hf', 2)
    eda_m   = feats.get('eda_scl_mean', 3)
    eda_pk  = feats.get('eda_peak_rate', 3)

    if lvl == 'low':
        recs = [
            ("🟢", "Autonomic balance is normal. Maintain your current activity rhythm."),
            ("💧", "Stay hydrated — optimal cognitive function begins with hydration."),
            ("📵", "Schedule a brief screen break every 90 minutes to preserve focus.")
        ]
    elif lvl == 'moderate':
        recs = [
            ("🌬️", "Elevated stress detected. Try box breathing: inhale 4s → hold 4s → exhale 4s → hold 4s."),
            ("⏸️",  f"RMSSD at {rmssd:.0f}ms (reduced parasympathetic tone). Take a 5-minute walk."),
            ("🔇",  "Reduce environmental stimulation — silence notifications for 20 minutes."),
            ("☕",  "Avoid caffeine for the next 2 hours; it will further elevate sympathetic activity.")
        ]
        if lf_hf > 3.0:
            recs.append(("📊", f"LF/HF ratio {lf_hf:.1f} indicates sympathetic dominance. Extended exhale breathing helps (4s in, 6-8s out)."))
    else:
        recs = [
            ("🚨", f"HIGH BURNOUT RISK (score: {score:.0f}/100). Immediate recovery intervention recommended."),
            ("😮‍💨", "4-7-8 breathing: inhale 4s, hold 7s, exhale 8s. Repeat 3 cycles immediately."),
            ("🛑",  "Pause current task. Cognitive performance under this autonomic load is significantly impaired."),
            ("🌿",  f"EDA level {eda_m:.1f}μS ({eda_pk:.0f} SCR peaks/min) confirms sustained physiological arousal."),
            ("🛏️",  "If this is end-of-day: prioritize 7-9h sleep. HRV recovery occurs primarily during deep sleep.")
        ]
        if sustained:
            recs.insert(1, ("⚠️", "Sustained high risk over 5+ consecutive windows. This indicates chronic load, not transient stress."))

    return recs


# ─────────────────────────────────────────────────────────────────────────────
# Simulation step
# ─────────────────────────────────────────────────────────────────────────────
def run_simulation_step(stress_level):
    """Run one simulation window and update session state."""
    wid  = st.session_state.window_count
    feats, rmssd = build_realtime_features(stress_level, wid, st.session_state.prev_rmssd)

    # Score prediction
    if st.session_state.model_loaded:
        fcols  = st.session_state.feature_cols
        x_row  = np.array([[feats.get(c, 0) for c in fcols]])
        x_sc   = st.session_state.scaler.transform(x_row)
        raw_score = float(st.session_state.model.predict_risk_score(x_sc)[0])
    else:
        # Heuristic score if model not loaded
        lf_hf = feats['hrv_lf_hf']
        eda_m = feats['eda_scl_mean']
        rmssd_v = feats['hrv_rmssd']
        raw_score = np.clip(
            (lf_hf / 8 * 40) + (eda_m / 10 * 30) + ((60 - rmssd_v) / 60 * 30),
            0, 100
        )

    smoothed, sustained = update_burnout_state(raw_score)

    # Waveforms
    _, ecg_wave = simulate_ecg_segment(5, 250, stress_level)
    _, eda_wave = simulate_eda_segment(30, 4, stress_level)

    # History record
    rec = {
        'window':     wid,
        'time':       datetime.now().strftime('%H:%M:%S'),
        'risk_score': smoothed,
        'rmssd':      feats['hrv_rmssd'],
        'lf_hf':      feats['hrv_lf_hf'],
        'eda_mean':   feats['eda_scl_mean'],
        'eda_peaks':  feats['eda_peak_rate'],
        'sustained':  sustained,
    }
    st.session_state.history.append(rec)
    if len(st.session_state.history) > 60:  # keep last 60 windows
        st.session_state.history.pop(0)

    st.session_state.current_score = smoothed
    st.session_state.current_feats = feats
    st.session_state.prev_rmssd    = rmssd
    st.session_state.ecg_buffer    = ecg_wave
    st.session_state.eda_buffer    = eda_wave
    st.session_state.window_count += 1

    # Alert log
    if smoothed >= 62:
        msg = f"[{rec['time']}] HIGH RISK — Score {smoothed:.0f} | RMSSD {feats['hrv_rmssd']:.0f}ms | LF/HF {feats['hrv_lf_hf']:.1f}"
        if not st.session_state.alert_log or st.session_state.alert_log[-1] != msg:
            st.session_state.alert_log.append(msg)
            if len(st.session_state.alert_log) > 20:
                st.session_state.alert_log.pop(0)

    return sustained


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;margin-bottom:20px'>
        <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#e8f0ff'>
            🫀 BurnoutSense
        </div>
        <div style='font-size:0.65rem;letter-spacing:2px;color:#5a7ba8;text-transform:uppercase'>
            Autonomic Monitor v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model status
    if st.session_state.model_loaded:
        st.success("✅ Model loaded")
    else:
        st.warning("⚠️ Model not found\nRun Step 1 first, or heuristic scoring will be used.")

    st.markdown("---")
    st.markdown('<div class="section-header">Simulation Controls</div>', unsafe_allow_html=True)

    stress_input = st.slider(
        "Simulated Stress Level",
        min_value=0.0, max_value=1.0, value=0.4, step=0.05,
        help="0 = relaxed baseline, 1 = maximum stress"
    )

    st.markdown(f"""
    <div style='font-size:0.7rem;color:#5a7ba8;margin-top:-8px;margin-bottom:12px'>
        Simulates autonomic response to sustained load.<br>
        This drives HRV and EDA signal generation.
    </div>
    """, unsafe_allow_html=True)

    update_speed = st.select_slider(
        "Update Interval",
        options=["Fast (1s)", "Normal (2s)", "Slow (4s)"],
        value="Normal (2s)"
    )
    speed_map = {"Fast (1s)": 1, "Normal (2s)": 2, "Slow (4s)": 4}
    sleep_sec = speed_map[update_speed]

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ START", use_container_width=True):
            st.session_state.running = True
    with col2:
        if st.button("⏹ STOP", use_container_width=True):
            st.session_state.running = False

    if st.button("↺ RESET", use_container_width=True):
        for k in ['history', 'window_count', 'current_score', 'current_feats',
                  'rolling_state', 'alert_log', 'prev_rmssd']:
            st.session_state[k] = [] if isinstance(st.session_state[k], list) else 0
        st.session_state.running = False

    st.markdown("---")
    st.markdown('<div class="section-header">Risk Thresholds</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem;line-height:1.8;color:#8aa8d0'>
        <span style='color:#22c55e'>●</span> Low Risk &nbsp;&nbsp; 0 – 34<br>
        <span style='color:#eab308'>●</span> Elevated &nbsp;&nbsp;&nbsp; 35 – 61<br>
        <span style='color:#ef4444'>●</span> High Risk &nbsp; 62 – 100
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Session Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.72rem;color:#8aa8d0;line-height:1.8'>
        Windows processed: <b style='color:#e8f0ff'>{st.session_state.window_count}</b><br>
        History depth: <b style='color:#e8f0ff'>{len(st.session_state.history)}</b><br>
        Rolling N: <b style='color:#e8f0ff'>{ROLLING_N} windows</b><br>
        Status: <b style='color:{"#22c55e" if st.session_state.running else "#5a7ba8"}'>
            {"● RUNNING" if st.session_state.running else "○ STOPPED"}
        </b>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
# Title bar
st.markdown(f"""
<div class="title-bar">
    <div style='font-size:2rem'>🫀</div>
    <div>
        <div class="title-main">BurnoutSense</div>
        <div class="title-sub">Physiological Autonomic Risk Monitor · ECG + EDA Multimodal Analysis</div>
    </div>
    <div style='margin-left:auto'>
        <span class="live-badge">{"● LIVE" if st.session_state.running else "○ IDLE"}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Run one simulation step if active
sustained = False
if st.session_state.running:
    sustained = run_simulation_step(stress_input)

score    = st.session_state.current_score
feats    = st.session_state.current_feats
history  = st.session_state.history
hist_df  = pd.DataFrame(history) if history else pd.DataFrame(
    columns=['window','time','risk_score','rmssd','lf_hf','eda_mean','eda_peaks','sustained'])

# ── Row 1: Gauge + key metrics + status ──
col_gauge, col_metrics, col_status = st.columns([1.2, 2.2, 1.6])

with col_gauge:
    st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
    st.plotly_chart(make_risk_gauge(score), use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_metrics:
    st.markdown('<div class="section-header">Biomarker Panel</div>', unsafe_allow_html=True)
    r1c1, r1c2, r1c3 = st.columns(3)
    r2c1, r2c2, r2c3 = st.columns(3)

    def metric_card(col, label, value, unit, delta_txt, delta_dir, accent):
        delta_class = {'up': 'delta-up', 'down': 'delta-down', 'flat': 'delta-flat'}[delta_dir]
        col.markdown(f"""
        <div class="metric-card" style="--accent-color:{accent}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-unit">{unit}</div>
            <div class="metric-delta {delta_class}">{delta_txt}</div>
        </div>
        """, unsafe_allow_html=True)

    rmssd_v   = feats.get('hrv_rmssd', 0)
    hr_v      = feats.get('hrv_mean_hr', 0)
    lf_hf_v   = feats.get('hrv_lf_hf', 0)
    sdnn_v    = feats.get('hrv_sdnn', 0)
    eda_v     = feats.get('eda_scl_mean', 0)
    eda_pk_v  = feats.get('eda_peak_rate', 0)

    metric_card(r1c1, "RMSSD",         f"{rmssd_v:.0f}",    "ms",
                "↓ parasympathetic" if rmssd_v < 30 else "✓ normal",
                'up' if rmssd_v < 30 else 'flat', '#3b82f6')
    metric_card(r1c2, "HEART RATE",    f"{hr_v:.0f}",       "bpm",
                "↑ elevated" if hr_v > 85 else "✓ normal",
                'up' if hr_v > 85 else 'flat', '#8b5cf6')
    metric_card(r1c3, "LF/HF RATIO",   f"{lf_hf_v:.2f}",   "ratio",
                "↑ sympathetic dominance" if lf_hf_v > 3 else "✓ balanced",
                'up' if lf_hf_v > 3 else 'flat', '#f97316')
    metric_card(r2c1, "SDNN",          f"{sdnn_v:.0f}",     "ms",
                "↓ low variability" if sdnn_v < 30 else "✓ normal",
                'up' if sdnn_v < 30 else 'flat', '#06b6d4')
    metric_card(r2c2, "SKIN CONDUCT.", f"{eda_v:.2f}",      "μS",
                "↑ aroused" if eda_v > 5 else "✓ calm",
                'up' if eda_v > 5 else 'flat', '#10b981')
    metric_card(r2c3, "SCR PEAKS",     f"{eda_pk_v:.1f}",   "/min",
                "↑ elevated" if eda_pk_v > 7 else "✓ normal",
                'up' if eda_pk_v > 7 else 'flat', '#f59e0b')

with col_status:
    lvl = risk_level(score)
    recs = get_recommendations(score, feats, sustained)

    status_html = {
        'low': f"""
        <div class="status-low">
            <div class="status-title">✓ Autonomic Balance Normal</div>
            <div class="status-body">HRV parameters within healthy range. Parasympathetic activity dominant. No intervention required.</div>
        </div>""",
        'moderate': f"""
        <div class="status-moderate">
            <div class="status-title">⚡ Elevated Sympathetic Activity</div>
            <div class="status-body">Reduced HRV and elevated EDA indicate physiological stress load. Monitor closely and consider recovery actions.</div>
        </div>""",
        'high': f"""
        <div class="status-high">
            <div class="status-title">🚨 High Burnout Risk Detected</div>
            <div class="status-body">Sustained autonomic dysregulation detected. Immediate intervention recommended. {"<br><b>Sustained over multiple windows.</b>" if sustained else ""}</div>
        </div>"""
    }[lvl]

    st.markdown(status_html, unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Adaptive Interventions</div>', unsafe_allow_html=True)

    for icon, txt in recs[:3]:
        st.markdown(f"""
        <div class="suggestion-card">
            <span class="suggestion-icon">{icon}</span>
            <span>{txt}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Row 2: Trend charts ──
col_trend, col_hrv, col_eda = st.columns(3)

with col_trend:
    st.plotly_chart(make_risk_trend(hist_df), use_container_width=True,
                    config={'displayModeBar': False})

with col_hrv:
    st.plotly_chart(make_hrv_chart(hist_df), use_container_width=True,
                    config={'displayModeBar': False})

with col_eda:
    st.plotly_chart(make_eda_chart(hist_df), use_container_width=True,
                    config={'displayModeBar': False})

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Row 3: Waveforms + Radar ──
col_waves, col_radar = st.columns([2.5, 1])

with col_waves:
    ecg_buf = st.session_state.ecg_buffer
    eda_buf = st.session_state.eda_buffer
    st.plotly_chart(make_ecg_waveform(ecg_buf), use_container_width=True,
                    config={'displayModeBar': False})
    st.plotly_chart(make_eda_waveform(eda_buf, 4), use_container_width=True,
                    config={'displayModeBar': False})

with col_radar:
    if feats:
        st.plotly_chart(make_feature_radar(feats), use_container_width=True,
                        config={'displayModeBar': False})

    # Extra recommendations
    if len(recs) > 3:
        st.markdown('<div class="section-header" style="margin-top:8px">More Actions</div>',
                    unsafe_allow_html=True)
        for icon, txt in recs[3:]:
            st.markdown(f"""
            <div class="suggestion-card">
                <span class="suggestion-icon">{icon}</span>
                <span>{txt}</span>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Row 4: Data table + Alert log ──
col_table, col_alerts = st.columns([2, 1])

with col_table:
    st.markdown('<div class="section-header">Window History</div>', unsafe_allow_html=True)
    if not hist_df.empty:
        disp = hist_df[['window','time','risk_score','rmssd','lf_hf',
                         'eda_mean','eda_peaks','sustained']].copy()
        disp.columns = ['Window', 'Time', 'Risk Score', 'RMSSD (ms)',
                         'LF/HF', 'EDA (μS)', 'SCR/min', 'Sustained']
        disp['Risk Score'] = disp['Risk Score'].round(1)
        disp['RMSSD (ms)'] = disp['RMSSD (ms)'].round(1)
        disp['LF/HF']      = disp['LF/HF'].round(2)
        disp['EDA (μS)']   = disp['EDA (μS)'].round(2)
        disp['SCR/min']    = disp['SCR/min'].round(1)

        def color_risk(val):
            if   val >= 62: return 'color: #ef4444; font-weight: bold'
            elif val >= 35: return 'color: #eab308'
            return 'color: #22c55e'

        styled = disp.tail(15).style.applymap(color_risk, subset=['Risk Score'])
        st.dataframe(styled, use_container_width=True, height=280)
    else:
        st.info("No data yet. Press START to begin simulation.")

with col_alerts:
    st.markdown('<div class="section-header">Alert Log</div>', unsafe_allow_html=True)
    if st.session_state.alert_log:
        alert_html = ""
        for a in reversed(st.session_state.alert_log[-10:]):
            alert_html += f"""
            <div style='background:#1a0505;border:1px solid #3c1a1a;border-radius:6px;
                        padding:8px 12px;margin-bottom:6px;font-size:0.68rem;
                        color:#fca5a5;line-height:1.4'>
                {a}
            </div>"""
        st.markdown(alert_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='color:#5a7ba8;font-size:0.75rem;padding:12px'>
            No high-risk alerts yet.<br>Alerts appear when risk score ≥ 62.
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align:center;margin-top:24px;padding-top:16px;
            border-top:1px solid #1e3055;font-size:0.65rem;
            color:#2a4a70;letter-spacing:1px'>
    BURNOUTSENSE · BIOMEDICAL INSTRUMENTATION PROJECT ·
    ECG + EDA MULTIMODAL AUTONOMIC MONITORING ·
    NOT FOR CLINICAL USE
</div>
""", unsafe_allow_html=True)

# ── Auto-refresh while running ──
if st.session_state.running:
    time.sleep(sleep_sec)
    st.rerun()
