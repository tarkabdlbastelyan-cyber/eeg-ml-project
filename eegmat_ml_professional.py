"""
Professional Machine Learning pipeline for EEGMAT Mental Workload classification.
Dataset: PhysioNet EEG During Mental Arithmetic Tasks (EEGMAT)
Task: binary classification: rest/baseline (_1.edf) vs mental arithmetic/workload (_2.edf)

Main anti-overfitting rules used here:
1) Group-aware evaluation by subject: no windows from the same subject appear in both train and test.
2) Scaling/feature selection/model fitting are inside sklearn Pipeline, so they are fitted only on training folds.
3) Nested CV option for hyperparameter tuning without leaking validation/test folds.
4) Window-level probabilities are aggregated to subject-file level for a fairer final report.

Install:
    pip install numpy pandas scipy scikit-learn mne joblib matplotlib xgboost lightgbm

Download data manually from PhysioNet, then run:
    python eegmat_ml_professional.py --data_dir "PATH_TO_EEGMAT_FOLDER" --model auto --nested 1

Example data folder should contain files like:
    Subject00_1.edf, Subject00_2.edf, ..., subject-info.csv
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis

import mne
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# EEGMAT common 10-20 channels from the 23-channel recording.
# If a channel is missing, the script simply uses the available intersection.
PREFERRED_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
    "C3", "C4", "Cz", "T3", "T4", "T5", "T6",
    "P3", "P4", "Pz", "O1", "O2",
]


@dataclass
class Config:
    data_dir: str
    out_dir: str = "eegmat_ml_outputs"
    window_sec: float = 4.0
    step_sec: float = 2.0
    l_freq: float = 1.0
    h_freq: float = 45.0
    notch_freq: float = 50.0
    resample: Optional[float] = 128.0
    use_subject_level_report: bool = True
    model: str = "auto"  # auto, xgb, lgbm, svm, rf, extratrees, ensemble
    nested: int = 1
    random_state: int = 42
    max_subjects: Optional[int] = None


def parse_subject_and_label(path: Path) -> Tuple[str, int, str]:
    """Return subject id, label, and condition name from EEGMAT filename."""
    m = re.search(r"Subject(\d+)_(\d+)\.edf$", path.name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Unexpected EDF filename: {path.name}")
    subject = f"Subject{int(m.group(1)):02d}"
    condition = int(m.group(2))
    # EEGMAT naming convention: _1 = before/rest, _2 = during mental arithmetic.
    label = 0 if condition == 1 else 1
    condition_name = "baseline" if label == 0 else "workload"
    return subject, label, condition_name


def clean_channel_names(raw: mne.io.BaseRaw) -> None:
    mapping = {}
    for ch in raw.ch_names:
        cleaned = ch.strip().replace("EEG ", "").replace("-Ref", "").replace(".", "")
        mapping[ch] = cleaned
    raw.rename_channels(mapping)


def load_preprocess_edf(path: Path, cfg: Config) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    clean_channel_names(raw)

    available = [ch for ch in PREFERRED_CHANNELS if ch in raw.ch_names]
    if len(available) >= 8:
        raw.pick_channels(available, ordered=True)
    else:
        # fallback: keep EEG-like channels only if preferred names were not found
        raw.pick_types(eeg=True, exclude=[])

    # Filtering is conservative; EEGMAT files are already artifact-cleaned according to dataset docs.
    raw.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, fir_design="firwin", verbose=False)
    if cfg.notch_freq:
        raw.notch_filter(freqs=[cfg.notch_freq], verbose=False)
    if cfg.resample:
        raw.resample(cfg.resample, verbose=False)

    # Average reference is standard for many EEG ML pipelines.
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def hjorth_parameters(x: np.ndarray) -> Tuple[float, float, float]:
    dx = np.diff(x)
    ddx = np.diff(dx)
    var0 = np.var(x) + 1e-12
    var1 = np.var(dx) + 1e-12
    var2 = np.var(ddx) + 1e-12
    activity = var0
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / (mobility + 1e-12)
    return float(activity), float(mobility), float(complexity)


def spectral_entropy(psd: np.ndarray) -> float:
    p = psd / (np.sum(psd) + 1e-12)
    return float(-np.sum(p * np.log2(p + 1e-12)) / np.log2(len(p) + 1e-12))


def bandpower(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def extract_window_features(window: np.ndarray, sfreq: float, ch_names: List[str]) -> Dict[str, float]:
    """
    window shape: channels x samples.
    Returns robust time, frequency, ratio, nonlinear, asymmetry, and connectivity features.
    """
    feats: Dict[str, float] = {}
    eps = 1e-12

    # Per-channel features
    band_values_by_channel = {}
    for ci, ch in enumerate(ch_names):
        x = window[ci].astype(float)
        x = x - np.mean(x)

        feats[f"{ch}_mean"] = float(np.mean(x))
        feats[f"{ch}_std"] = float(np.std(x))
        feats[f"{ch}_var"] = float(np.var(x))
        feats[f"{ch}_rms"] = float(np.sqrt(np.mean(x**2)))
        feats[f"{ch}_skew"] = float(skew(x))
        feats[f"{ch}_kurtosis"] = float(kurtosis(x))
        feats[f"{ch}_ptp"] = float(np.ptp(x))
        feats[f"{ch}_zero_cross"] = float(np.mean(np.diff(np.signbit(x)) != 0))

        activity, mobility, complexity = hjorth_parameters(x)
        feats[f"{ch}_hjorth_activity"] = activity
        feats[f"{ch}_hjorth_mobility"] = mobility
        feats[f"{ch}_hjorth_complexity"] = complexity

        freqs, psd = welch(x, fs=sfreq, nperseg=min(len(x), int(2 * sfreq)))
        total_power = bandpower(freqs, psd, 1.0, 45.0) + eps
        feats[f"{ch}_total_power"] = total_power
        feats[f"{ch}_spectral_entropy"] = spectral_entropy(psd)

        channel_bands = {}
        for band_name, (lo, hi) in BANDS.items():
            bp = bandpower(freqs, psd, lo, hi)
            channel_bands[band_name] = bp
            feats[f"{ch}_{band_name}_abs"] = bp
            feats[f"{ch}_{band_name}_rel"] = bp / total_power
        band_values_by_channel[ch] = channel_bands

        feats[f"{ch}_theta_alpha"] = channel_bands["theta"] / (channel_bands["alpha"] + eps)
        feats[f"{ch}_beta_alpha"] = channel_bands["beta"] / (channel_bands["alpha"] + eps)
        feats[f"{ch}_theta_beta"] = channel_bands["theta"] / (channel_bands["beta"] + eps)
        feats[f"{ch}_engagement_index"] = channel_bands["beta"] / (channel_bands["alpha"] + channel_bands["theta"] + eps)

    # Global band summaries across channels
    for band_name in BANDS:
        vals = np.array([band_values_by_channel[ch][band_name] for ch in ch_names], dtype=float)
        feats[f"global_{band_name}_mean"] = float(np.mean(vals))
        feats[f"global_{band_name}_std"] = float(np.std(vals))
        feats[f"global_{band_name}_max"] = float(np.max(vals))

    # Left-right asymmetry for common pairs: log power difference.
    pairs = [("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8"), ("C3", "C4"), ("T3", "T4"), ("T5", "T6"), ("P3", "P4"), ("O1", "O2")]
    for left, right in pairs:
        if left in ch_names and right in ch_names:
            for band_name in BANDS:
                lp = band_values_by_channel[left][band_name] + eps
                rp = band_values_by_channel[right][band_name] + eps
                feats[f"asym_{left}_{right}_{band_name}"] = float(np.log(lp) - np.log(rp))

    # Lightweight connectivity: upper triangle correlation features summarized, not all pairs.
    if window.shape[0] >= 2:
        corr = np.corrcoef(window)
        upper = corr[np.triu_indices_from(corr, k=1)]
        upper = np.nan_to_num(upper, nan=0.0, posinf=0.0, neginf=0.0)
        feats["conn_corr_mean"] = float(np.mean(upper))
        feats["conn_corr_std"] = float(np.std(upper))
        feats["conn_corr_abs_mean"] = float(np.mean(np.abs(upper)))
        feats["conn_corr_max"] = float(np.max(upper))
        feats["conn_corr_min"] = float(np.min(upper))

    return feats


def create_feature_table(cfg: Config) -> pd.DataFrame:
    data_dir = Path(cfg.data_dir)
    edf_files = sorted(data_dir.glob("Subject*_*.edf"))
    if not edf_files:
        raise FileNotFoundError(f"No EEGMAT EDF files found in: {data_dir}")

    subjects_seen = []
    rows = []
    for path in edf_files:
        subject, label, condition_name = parse_subject_and_label(path)
        if cfg.max_subjects is not None:
            if subject not in subjects_seen:
                subjects_seen.append(subject)
            if len(subjects_seen) > cfg.max_subjects:
                continue

        print(f"[INFO] Processing {path.name} | {subject} | {condition_name}")
        raw = load_preprocess_edf(path, cfg)
        data = raw.get_data()  # channels x samples
        sfreq = float(raw.info["sfreq"])
        ch_names = list(raw.ch_names)

        win = int(cfg.window_sec * sfreq)
        step = int(cfg.step_sec * sfreq)
        n_samples = data.shape[1]
        n_windows = max(0, 1 + (n_samples - win) // step)
        if n_windows == 0:
            print(f"[WARN] Skipping {path.name}: too short")
            continue

        for wi, start in enumerate(range(0, n_samples - win + 1, step)):
            segment = data[:, start : start + win]
            feats = extract_window_features(segment, sfreq, ch_names)
            feats["subject"] = subject
            feats["file"] = path.name
            feats["window_id"] = wi
            feats["label"] = label
            feats["condition"] = condition_name
            rows.append(feats)

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def get_model_and_grid(name: str, random_state: int):
    name = name.lower()

    if name == "xgb" and XGBClassifier is not None:
        clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        grid = {
            "clf__n_estimators": [100, 250],
            "clf__max_depth": [2, 3, 4],
            "clf__learning_rate": [0.03, 0.07],
            "clf__min_child_weight": [1, 5],
            "clf__reg_lambda": [1, 5, 10],
        }
        return clf, grid

    if name == "lgbm" and LGBMClassifier is not None:
        clf = LGBMClassifier(
            objective="binary",
            random_state=random_state,
            n_jobs=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
        )
        grid = {
            "clf__n_estimators": [100, 250],
            "clf__num_leaves": [7, 15, 31],
            "clf__learning_rate": [0.03, 0.07],
            "clf__min_child_samples": [5, 10, 20],
            "clf__reg_lambda": [1, 5, 10],
        }
        return clf, grid

    if name == "svm":
        clf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)
        grid = {
            "clf__C": [0.3, 1, 3, 10],
            "clf__gamma": ["scale", 0.01, 0.03, 0.1],
        }
        return clf, grid

    if name == "rf":
        clf = RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1)
        grid = {
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [3, 5, 8, None],
            "clf__min_samples_leaf": [2, 4, 8],
            "clf__max_features": ["sqrt", 0.5],
        }
        return clf, grid

    if name == "extratrees":
        clf = ExtraTreesClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1)
        grid = {
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [3, 5, 8, None],
            "clf__min_samples_leaf": [2, 4, 8],
            "clf__max_features": ["sqrt", 0.5],
        }
        return clf, grid

    if name == "ensemble":
        estimators = []
        if XGBClassifier is not None:
            estimators.append(("xgb", XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=5, random_state=random_state, n_jobs=-1)))
        estimators.append(("svm", SVC(kernel="rbf", C=1, gamma="scale", probability=True, class_weight="balanced", random_state=random_state)))
        estimators.append(("et", ExtraTreesClassifier(n_estimators=400, max_depth=5, min_samples_leaf=4, max_features="sqrt", class_weight="balanced", random_state=random_state, n_jobs=-1)))
        clf = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
        grid = {"select__k": [80, 150, 250, "all"]}
        return clf, grid

    # auto fallback: prefer XGBoost, then LightGBM, then ExtraTrees
    if name == "auto":
        if XGBClassifier is not None:
            return get_model_and_grid("xgb", random_state)
        if LGBMClassifier is not None:
            return get_model_and_grid("lgbm", random_state)
        return get_model_and_grid("extratrees", random_state)

    raise ValueError(f"Unknown model={name}, or required package is not installed.")


def build_pipeline(model_name: str, random_state: int) -> Tuple[Pipeline, Dict]:
    clf, grid = get_model_and_grid(model_name, random_state)
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=mutual_info_classif, k=150)),
            ("clf", clf),
        ]
    )
    # Add feature selection choices to non-ensemble models too.
    if "select__k" not in grid:
        grid = {"select__k": [80, 150, 250, "all"], **grid}
    return pipe, grid


def aggregate_by_file(y_true: np.ndarray, proba: np.ndarray, files: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for f in np.unique(files):
        idx = files == f
        rows.append((f, int(np.round(np.mean(y_true[idx]))), float(np.mean(proba[idx]))))
    out = pd.DataFrame(rows, columns=["file", "label", "proba"])
    y_file = out["label"].to_numpy()
    p_file = out["proba"].to_numpy()
    pred_file = (p_file >= 0.5).astype(int)
    return y_file, pred_file, p_file


def evaluate(cfg: Config, df: pd.DataFrame) -> Dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_cols = ["subject", "file", "window_id", "label", "condition"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    groups = df["subject"].to_numpy()
    files = df["file"].to_numpy()

    outer = LeaveOneGroupOut()

    all_true, all_pred, all_prob, all_file = [], [], [], []
    fold_reports = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y, groups), start=1):
        test_subject = np.unique(groups[test_idx])[0]
        print(f"\n[CV] Fold {fold:02d} | test subject = {test_subject}")

        pipe, grid = build_pipeline(cfg.model, cfg.random_state)

        if cfg.nested:
            inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=cfg.random_state)
            search = GridSearchCV(
                pipe,
                param_grid=grid,
                scoring="f1_macro",
                cv=inner,
                n_jobs=-1,
                refit=True,
                verbose=0,
            )
            search.fit(X[train_idx], y[train_idx], groups=groups[train_idx])
            model = search.best_estimator_
            best_params = search.best_params_
        else:
            model = pipe.fit(X[train_idx], y[train_idx])
            best_params = {}

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X[test_idx])[:, 1]
        else:
            # fallback; SVC has probability=True so this rarely triggers
            pred_temp = model.predict(X[test_idx])
            prob = pred_temp.astype(float)
        pred = (prob >= 0.5).astype(int)

        acc = accuracy_score(y[test_idx], pred)
        mf1 = f1_score(y[test_idx], pred, average="macro")
        bacc = balanced_accuracy_score(y[test_idx], pred)
        try:
            auc = roc_auc_score(y[test_idx], prob)
        except Exception:
            auc = np.nan

        print(f"[CV] acc={acc:.3f} | macro_f1={mf1:.3f} | bal_acc={bacc:.3f} | auc={auc:.3f}")
        fold_reports.append({
            "fold": fold,
            "test_subject": test_subject,
            "accuracy": acc,
            "macro_f1": mf1,
            "balanced_accuracy": bacc,
            "roc_auc": auc,
            "best_params": best_params,
        })

        all_true.extend(y[test_idx].tolist())
        all_pred.extend(pred.tolist())
        all_prob.extend(prob.tolist())
        all_file.extend(files[test_idx].tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_prob = np.array(all_prob)
    all_file = np.array(all_file)

    window_metrics = {
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "macro_f1": float(f1_score(all_true, all_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(all_true, all_pred)),
        "roc_auc": float(roc_auc_score(all_true, all_prob)),
        "confusion_matrix": confusion_matrix(all_true, all_pred).tolist(),
        "classification_report": classification_report(all_true, all_pred, target_names=["baseline", "workload"], output_dict=True),
    }

    y_file, pred_file, p_file = aggregate_by_file(all_true, all_prob, all_file)
    file_metrics = {
        "accuracy": float(accuracy_score(y_file, pred_file)),
        "macro_f1": float(f1_score(y_file, pred_file, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_file, pred_file)),
        "roc_auc": float(roc_auc_score(y_file, p_file)),
        "confusion_matrix": confusion_matrix(y_file, pred_file).tolist(),
        "classification_report": classification_report(y_file, pred_file, target_names=["baseline", "workload"], output_dict=True),
    }

    results = {
        "config": asdict(cfg),
        "n_windows": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "n_features": int(len(feature_cols)),
        "window_level_metrics": window_metrics,
        "file_level_metrics": file_metrics,
        "fold_reports": fold_reports,
    }

    pd.DataFrame(fold_reports).to_csv(out_dir / "loso_fold_results.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    pred_df = pd.DataFrame({
        "file": all_file,
        "y_true": all_true,
        "y_pred": all_pred,
        "prob_workload": all_prob,
    })
    pred_df.to_csv(out_dir / "window_predictions.csv", index=False)

    return results


def train_final_model(cfg: Config, df: pd.DataFrame) -> None:
    """Train final model on all data after CV, for later inference/deployment."""
    out_dir = Path(cfg.out_dir)
    meta_cols = ["subject", "file", "window_id", "label", "condition"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    groups = df["subject"].to_numpy()

    pipe, grid = build_pipeline(cfg.model, cfg.random_state)
    if cfg.nested:
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=inner, n_jobs=-1, refit=True)
        search.fit(X, y, groups=groups)
        final_model = search.best_estimator_
        final_params = search.best_params_
    else:
        final_model = pipe.fit(X, y)
        final_params = {}

    joblib.dump({
        "model": final_model,
        "feature_cols": feature_cols,
        "config": asdict(cfg),
        "final_params": final_params,
    }, out_dir / "final_eegmat_ml_model.joblib")
    print(f"[DONE] Final model saved to: {out_dir / 'final_eegmat_ml_model.joblib'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="eegmat_ml_outputs")
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "xgb", "lgbm", "svm", "rf", "extratrees", "ensemble"])
    parser.add_argument("--nested", type=int, default=1)
    parser.add_argument("--window_sec", type=float, default=4.0)
    parser.add_argument("--step_sec", type=float, default=2.0)
    parser.add_argument("--resample", type=float, default=128.0)
    parser.add_argument("--max_subjects", type=int, default=None, help="For quick debugging only. Use all subjects for real results.")
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        model=args.model,
        nested=args.nested,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        resample=args.resample,
        max_subjects=args.max_subjects,
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_path = out_dir / "features.parquet"
    if features_path.exists():
        print(f"[INFO] Loading cached features: {features_path}")
        df = pd.read_parquet(features_path)
    else:
        df = create_feature_table(cfg)
        df.to_parquet(features_path, index=False)
        df.to_csv(out_dir / "features.csv", index=False)
        print(f"[DONE] Features saved: {features_path}")

    print("\n[INFO] Dataset summary")
    print(df.groupby(["subject", "condition"]).size().head(10))
    print(f"Subjects: {df['subject'].nunique()} | Windows: {len(df)} | Columns: {len(df.columns)}")

    results = evaluate(cfg, df)
    train_final_model(cfg, df)

    print("\n========== FINAL RESULTS ==========")
    print("Window-level:", results["window_level_metrics"])
    print("File-level:", results["file_level_metrics"])
    print(f"Outputs saved in: {cfg.out_dir}")


if __name__ == "__main__":
    main()
