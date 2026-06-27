"""
Streamlit GUI for EEGMAT Mental Workload Detection
---------------------------------------------------
Model selection via radio buttons:
  • Best Soft Voting Ensemble (.pkl)  → inference_utils pipeline
  • Bundle Model (.joblib)            → eegmat_ml_professional pipeline
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Pipeline A — bundle .joblib ─────────────────────────────────────────
from eegmat_ml_professional import (
    Config,
    extract_window_features,
    load_preprocess_edf,
    parse_subject_and_label,
)

# ── Pipeline B — raw .pkl ───────────────────────────────────────────────
from inference_utils import (
    load_eeg_only,
    crop_to_60_seconds,
    segment_raw,
    extract_features_from_segment,
)


# ─────────────────────────── Page Setup ────────────────────────────────

st.set_page_config(
    page_title="EEG Workload Detection",
    page_icon="🧠",
    layout="centered",
)


# ─────────────────────────── Model Paths ───────────────────────────────

MODELS = {
    "🤖 Best Soft Voting Ensemble  (.pkl)": {
        "path": "best_soft_voting_eeg_model.pkl",
        "type": "raw",
        "desc": "0.5–30 Hz · EEG channels only · first 60 sec · majority-vote",
    },
    "📦 Bundle Model  (.joblib)": {
        "path": "eegmat_ml_outputs/final_eegmat_ml_model.joblib",
        "type": "bundle",
        "desc": "1–45 Hz · 50 Hz notch · 128 Hz resample · full recording",
    },
}

# Pipeline A settings
WINDOW_SEC    = 4.0
STEP_SEC      = 2.0
LOW_CUT       = 1.0
HIGH_CUT      = 45.0
NOTCH_FREQ    = 50.0
RESAMPLE_RATE = 128.0


# ─────────────────────────── Loaders ───────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)


# ─────────────────────────── Pipeline A ────────────────────────────────

@st.cache_data(show_spinner=False)
def pipeline_a_preprocess(
    edf_path: str,
) -> Tuple[pd.DataFrame, Dict, np.ndarray, float, List[str]]:
    cfg = Config(
        data_dir=str(Path(edf_path).parent),
        window_sec=WINDOW_SEC,
        step_sec=STEP_SEC,
        l_freq=LOW_CUT,
        h_freq=HIGH_CUT,
        notch_freq=NOTCH_FREQ,
        resample=RESAMPLE_RATE,
    )
    raw      = load_preprocess_edf(Path(edf_path), cfg)
    data     = raw.get_data()
    sfreq    = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)

    win    = int(WINDOW_SEC * sfreq)
    step   = int(STEP_SEC   * sfreq)
    n_samp = data.shape[1]

    if n_samp < win:
        raise ValueError("EDF file is shorter than the window length (4 s).")

    rows = []
    for wid, start in enumerate(range(0, n_samp - win + 1, step)):
        seg   = data[:, start : start + win]
        feats = extract_window_features(seg, sfreq, ch_names)
        feats["window_id"] = wid
        feats["start_sec"] = start / sfreq
        feats["end_sec"]   = (start + win) / sfreq
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    info = dict(
        channels=len(ch_names),
        sfreq=sfreq,
        duration_sec=n_samp / sfreq,
        windows=len(feat_df),
        samples=n_samp,
    )
    return feat_df, info, data, sfreq, ch_names


def pipeline_a_predict(
    bundle: Dict,
    feat_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, float, int, float]:
    model  = bundle["model"]
    f_cols = bundle["feature_cols"]

    aligned = pd.DataFrame(0.0, index=feat_df.index, columns=f_cols)
    common  = [c for c in f_cols if c in feat_df.columns]
    aligned.loc[:, common] = feat_df[common].astype(float)
    X = aligned.to_numpy(dtype=float)

    prob       = (model.predict_proba(X)[:, 1]
                  if hasattr(model, "predict_proba")
                  else model.predict(X).astype(float))
    win_pred   = (prob >= 0.5).astype(int)
    file_prob  = float(np.mean(prob))
    file_pred  = int(file_prob >= 0.5)
    confidence = file_prob if file_pred == 1 else 1.0 - file_prob

    pred_df = feat_df[["window_id", "start_sec", "end_sec"]].copy()
    pred_df["prob_workload"] = prob
    pred_df["prediction"]    = np.where(win_pred == 1, "workload", "baseline")

    return pred_df, file_prob, file_pred, confidence


# ─────────────────────────── Pipeline B ────────────────────────────────

@st.cache_data(show_spinner=False)
def pipeline_b_preprocess(
    edf_path: str,
) -> Tuple[np.ndarray, int, np.ndarray, float, List[str], Dict]:
    raw      = load_eeg_only(edf_path)
    raw      = crop_to_60_seconds(raw)
    sfreq    = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)
    raw_data = raw.get_data() * 1e6

    segments, fs = segment_raw(raw)

    info = dict(
        channels=len(ch_names),
        sfreq=sfreq,
        duration_sec=float(raw.times[-1]),
        windows=len(segments),
        samples=raw_data.shape[1],
    )
    return segments, fs, raw_data, sfreq, ch_names, info


def pipeline_b_predict(
    model,
    segments: np.ndarray,
    fs: int,
) -> Tuple[pd.DataFrame, float, int, float, Dict]:
    X = np.array([extract_features_from_segment(seg, fs) for seg in segments])

    seg_preds = model.predict(X)
    probs     = (model.predict_proba(X)[:, 1]
                 if hasattr(model, "predict_proba")
                 else seg_preds.astype(float))

    low_count  = int(np.sum(seg_preds == 0))
    high_count = int(np.sum(seg_preds == 1))
    file_pred  = 1 if high_count > low_count else 0
    file_prob  = float(np.mean(probs))
    confidence = file_prob if file_pred == 1 else 1.0 - file_prob

    pred_df = pd.DataFrame({
        "window_id":     np.arange(len(seg_preds)),
        "start_sec":     np.arange(len(seg_preds)) * 2.0,
        "end_sec":       np.arange(len(seg_preds)) * 2.0 + 4.0,
        "prob_workload": probs,
        "prediction":    np.where(seg_preds == 1, "workload", "baseline"),
    })

    summary = dict(
        low_segments=low_count,
        high_segments=high_count,
        high_percentage=(high_count / len(seg_preds)) * 100,
        total_segments=len(seg_preds),
    )
    return pred_df, file_prob, file_pred, confidence, summary


# ─────────────────────────── Plots ─────────────────────────────────────

def plot_dual_eeg(
    data: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    pred_df: pd.DataFrame,
    max_channels: int = 5,
    seconds: float = 8.0,
):
    win_samples = int(4.0 * sfreq)
    n_ch        = min(max_channels, data.shape[0])

    high_segs, low_segs = [], []
    for _, row in pred_df.iterrows():
        start = int(row["start_sec"] * sfreq)
        end   = min(start + win_samples, data.shape[1])
        seg   = data[:n_ch, start:end]
        (high_segs if row["prediction"] == "workload" else low_segs).append(seg)

    def build(segs):
        return np.concatenate(segs, axis=1) if segs else np.zeros((n_ch, win_samples))

    def draw(ax, sig, title):
        n   = min(sig.shape[1], int(seconds * sfreq))
        t   = np.arange(n) / sfreq
        off = 0.0
        for i in range(n_ch):
            s     = sig[i, :n] - np.mean(sig[i, :n])
            scale = np.std(s) + 1e-12
            ax.plot(t, s / scale + off, label=ch_names[i])
            off += 4
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Normalized Amplitude")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    draw(ax1, build(high_segs), f"High Mental Workload — First {seconds:g} s")
    draw(ax2, build(low_segs),  f"Baseline / Low Workload — First {seconds:g} s")
    fig.tight_layout()
    return fig


def plot_segment_bar(summary: Dict):
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = ["Low Workload", "High Workload"]
    counts = [summary["low_segments"], summary["high_segments"]]
    colors = ["#4CAF50", "#F44336"]
    bars   = ax.bar(labels, counts, color=colors, width=0.5)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(count),
            ha="center", va="bottom", fontweight="bold",
        )
    ax.set_ylabel("Number of Segments")
    ax.set_title("Segment Voting Results")
    ax.set_ylim(0, max(counts) * 1.25 + 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ─────────────────────────── Main UI ───────────────────────────────────

st.title("🧠 EEG Mental Workload Detection")
st.caption("Select a model, upload an EDF file, then run classification.")
st.markdown("---")

# ── Step 1 — Model selection via radio buttons ──────────────────────────
st.subheader("Step 1 — Select Model")

selected_label = st.radio(
    "Choose model:",
    options=list(MODELS.keys()),
    index=0,
)

selected = MODELS[selected_label]

# Show model details card
st.markdown(
    f"""
    <div style="
        background:#f0f2f6;
        border-left:4px solid #4f8bf9;
        border-radius:6px;
        padding:10px 16px;
        margin-bottom:12px;
        font-size:0.9rem;
    ">
        <b>File:</b> <code>{selected['path']}</code><br>
        <b>Pipeline:</b> {selected['desc']}
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Step 2 — EDF upload ─────────────────────────────────────────────────
st.subheader("Step 2 — Upload EDF File")
edf_file = st.file_uploader("Upload EDF File", type=["edf"], label_visibility="collapsed")

# ── Step 3 — Run ────────────────────────────────────────────────────────
st.subheader("Step 3 — Classify")
run_btn = st.button(
    "🚀 Run Prediction",
    type="primary",
    use_container_width=True,
    disabled=edf_file is None,
)

st.markdown("---")

# ── Run ─────────────────────────────────────────────────────────────────
if run_btn and edf_file is not None:
    try:
        model_path = selected["path"]
        model_type = selected["type"]

        if not Path(model_path).exists():
            st.error(f"Model file not found: `{model_path}`\nMake sure it is in the project folder.")
            st.stop()

        obj = load_model(model_path)

        # Save EDF to temp file (shared by both pipelines)
        tmp_dir  = Path(tempfile.mkdtemp(prefix="eegmat_gui_"))
        edf_path = tmp_dir / edf_file.name
        edf_path.write_bytes(edf_file.read())

        with st.spinner("Processing EEG signal and running prediction..."):
            if model_type == "bundle":
                feat_df, info, raw_data, sfreq, ch_names = pipeline_a_preprocess(str(edf_path))
                pred_df, file_prob, file_pred, confidence = pipeline_a_predict(obj, feat_df)
                segment_summary = None

            else:
                segments, fs, raw_data, sfreq, ch_names, info = pipeline_b_preprocess(str(edf_path))
                pred_df, file_prob, file_pred, confidence, segment_summary = pipeline_b_predict(obj, segments, fs)

        # ── Ground truth from filename ───────────────────────────────
        try:
            subject, true_label, file_condition = parse_subject_and_label(edf_path)
        except Exception:
            subject, true_label, file_condition = "Unknown", -1, "unknown"

        pred_label = "workload" if file_pred == 1 else "baseline"

        # ── Result banner ────────────────────────────────────────────
        st.subheader("Result")
        msg = (
            f"🔴 **High Workload / Mental Arithmetic** — Confidence: {confidence*100:.1f}%"
            if file_pred == 1 else
            f"🟢 **Baseline / Low Workload** — Confidence: {confidence*100:.1f}%"
        )
        (st.error if file_pred == 1 else st.success)(msg)

        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", "High Workload" if file_pred == 1 else "Baseline")
        col2.metric("Confidence", f"{confidence*100:.1f}%")
        col3.metric("Windows",    info["windows"])

        # ── Segment voting (Pipeline B only) ─────────────────────────
        if segment_summary:
            st.subheader("📊 Segment Voting")
            c1, c2, c3 = st.columns(3)
            c1.metric("High Segments", segment_summary["high_segments"])
            c2.metric("Low Segments",  segment_summary["low_segments"])
            c3.metric("High %",        f"{segment_summary['high_percentage']:.1f}%")
            st.pyplot(plot_segment_bar(segment_summary), clear_figure=True)

        # ── EDF info ─────────────────────────────────────────────────
        with st.expander("EDF Information", expanded=True):
            st.write(f"**Subject:** {subject}")
            st.write(f"**Sampling Rate:** {info['sfreq']:.1f} Hz")
            st.write(f"**Channels:** {info['channels']}")
            st.write(f"**Duration:** {info['duration_sec']:.1f} sec")
            if true_label in [0, 1]:
                st.write(f"**Ground Truth (filename):** {'workload' if true_label==1 else 'baseline'}")

        # ── Dual EEG plot ─────────────────────────────────────────────
        st.subheader("🧬 EEG Signal: High Workload vs Baseline")
        st.pyplot(
            plot_dual_eeg(raw_data, sfreq, ch_names, pred_df,
                          max_channels=5, seconds=8.0),
            clear_figure=True,
        )

        # ── Download report ───────────────────────────────────────────
        report = pd.DataFrame([{
            "file":           edf_file.name,
            "model":          selected_label,
            "subject":        subject,
            "true_condition": file_condition,
            "prediction":     pred_label,
            "prob_workload":  file_prob,
            "confidence":     confidence,
            "n_windows":      info["windows"],
            "sfreq":          info["sfreq"],
            "channels":       info["channels"],
            "duration_sec":   info["duration_sec"],
            **(segment_summary or {}),
        }])
        st.download_button(
            "⬇️ Download Report CSV",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="eegmat_prediction_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as exc:
        st.error(f"Error: {exc}")
        st.exception(exc)

else:
    st.info("Select a model, upload an EDF file, then click **Run Prediction**.")

st.markdown("---")
st.caption(
    "PKL model: 0.5–30 Hz · EEG only · first 60 sec  |  "
    "Bundle model: 1–45 Hz · 50 Hz notch · 128 Hz resample · full recording"
)