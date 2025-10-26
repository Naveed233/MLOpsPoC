#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monitor_drift.py
- Reads data/predictions/predictions.jsonl (written by api_server.py)
- Builds/updates a baseline profile (feature distributions)
- Computes drift:
    * PSI for numeric features
    * Jensen-Shannon distance for categorical features
- Computes API latency stats (p50/p95)
- Optionally joins outcomes (data/actuals/actuals.csv) by external_id to compute MAE overall and by ward
- Writes a JSON summary + CSV details in reports/

Usage examples:
  # 1) Create a baseline from the last 30 days of logs
  python src/monitor_drift.py --create-baseline --baseline-window-days 30

  # 2) Compare last 7 days against baseline
  python src/monitor_drift.py --recent-window-days 7

  # 3) Include MAE if you have outcomes in data/actuals/actuals.csv
  python src/monitor_drift.py --recent-window-days 7 --with-actuals
"""

from __future__ import annotations
import os
import json
import argparse
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_LOG = os.path.join(BASE_DIR, "data", "predictions", "predictions.jsonl")
ACTUALS_CSV = os.path.join(BASE_DIR, "data", "actuals", "actuals.csv")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
BASELINE_JSON = os.path.join(REPORT_DIR, "baseline_profile.json")

os.makedirs(REPORT_DIR, exist_ok=True)

NUMERIC_FEATURES = [
    "effective_area_m2",
    "building_age_years",
    "coverage_ratio",
    "floor_area_ratio",
    "is_tokyo",
]
CATEGORICAL_FEATURES = [
    "prefecture",
    "city_ward",
    "property_type",
    "building_structure",
    "age_bucket",
]

# ------------------- Utilities -------------------


def _read_jsonl(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"[ERROR] Predictions log not found: {path}")
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    if not recs:
        raise SystemExit("[ERROR] No prediction records found.")
    df = pd.DataFrame(recs)
    # Flatten request dict
    if "request" in df.columns:
        req = pd.json_normalize(df["request"])
        req.columns = [c.split(".")[-1] for c in req.columns]
        df = pd.concat([df.drop(columns=["request"]), req], axis=1)
    # Parse timestamp
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce")
    df = df.dropna(subset=["ts_utc"]).sort_values("ts_utc")
    return df


def _time_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    end = df["ts_utc"].max()
    start = end - pd.Timedelta(days=days)
    return df[(df["ts_utc"] > start) & (df["ts_utc"] <= end)].copy()


def _hist_numeric(s: pd.Series, bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return np.zeros(bins), np.linspace(0, 1, bins + 1)
    hist, edges = np.histogram(s, bins=bins)
    p = hist / max(hist.sum(), 1)
    return p, edges


def _freq_categorical(s: pd.Series) -> Dict[str, float]:
    s = s.dropna().astype(str)
    if s.empty:
        return {}
    counts = s.value_counts()
    p = (counts / counts.sum()).to_dict()
    return p


def _psi(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Population Stability Index for numeric histograms p (baseline) vs q (recent)
    """
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return float(((q - p) * np.log(q / p)).sum())


def _jsd(p_map: Dict[str, float], q_map: Dict[str, float], eps: float = 1e-12) -> float:
    """
    Jensen-Shannon distance for categorical distributions
    """
    keys = sorted(set(p_map) | set(q_map))
    p = np.array([p_map.get(k, 0.0) for k in keys], dtype=float)
    q = np.array([q_map.get(k, 0.0) for k in keys], dtype=float)
    p = p / max(p.sum(), 1)
    q = q / max(q.sum(), 1)
    # jensenshannon returns distance in [0,1]
    return float(jensenshannon(p + eps, q + eps))


def _latency_stats(df: pd.DataFrame) -> Dict[str, float]:
    if "latency_ms" not in df.columns:
        return {}
    s = pd.to_numeric(df["latency_ms"], errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "count": int(len(s)),
        "p50_ms": float(np.percentile(s, 50)),
        "p95_ms": float(np.percentile(s, 95)),
        "max_ms": float(np.max(s)),
    }


def _load_actuals() -> pd.DataFrame:
    if not os.path.exists(ACTUALS_CSV):
        return pd.DataFrame()
    df = pd.read_csv(ACTUALS_CSV)
    # expected columns: external_id, sale_price_yen, ts_utc(optional), city_ward(optional)
    if "external_id" not in df.columns or "sale_price_yen" not in df.columns:
        return pd.DataFrame()
    return df


# ------------------- Baseline Profile -------------------


def create_baseline(df_win: pd.DataFrame, path: str) -> None:
    """
    Build baseline distributions for listed features from df_win window.
    """
    profile = {"created_at_utc": datetime.utcnow().isoformat(), "features": {}}

    for col in NUMERIC_FEATURES:
        if col in df_win.columns:
            p, edges = _hist_numeric(df_win[col])
            profile["features"][col] = {
                "type": "numeric",
                "hist": p.tolist(),
                "edges": edges.tolist(),
            }

    for col in CATEGORICAL_FEATURES:
        if col in df_win.columns:
            freq = _freq_categorical(df_win[col])
            profile["features"][col] = {"type": "categorical", "freq": freq}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    print(f"[OK] Baseline saved: {path}")


def load_baseline(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(
            f"[ERROR] Baseline profile not found: {path}\n"
            f"Create it first with: python src/monitor_drift.py --create-baseline"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------- Drift Computation -------------------


def compute_drift(baseline: dict, df_recent: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    summary = {"numeric_alerts": 0, "categorical_alerts": 0}
    # thresholds (industry rough heuristics)
    PSI_WARN, PSI_ALERT = 0.1, 0.25
    JSD_WARN, JSD_ALERT = 0.15, 0.3

    for feat, meta in baseline.get("features", {}).items():
        if meta["type"] == "numeric":
            p = np.array(meta["hist"], dtype=float)
            # rebuild q with the same bin edges
            edges = np.array(meta["edges"], dtype=float)
            s = pd.to_numeric(df_recent.get(feat), errors="coerce").dropna()
            if s.empty:
                psi = float("nan")
            else:
                hist, _ = np.histogram(s, bins=edges)
                q = hist / max(hist.sum(), 1)
                psi = _psi(p, q)
            status = "OK"
            if np.isfinite(psi):
                if psi >= PSI_ALERT:
                    status = "ALERT"
                    summary["numeric_alerts"] += 1
                elif psi >= PSI_WARN:
                    status = "WARN"
            rows.append(
                {
                    "feature": feat,
                    "type": "numeric",
                    "metric": "PSI",
                    "value": psi,
                    "status": status,
                }
            )

        elif meta["type"] == "categorical":
            base_freq = meta["freq"]
            recent_freq = (
                _freq_categorical(df_recent.get(feat)) if feat in df_recent.columns else {}
            )
            jsd = _jsd(base_freq, recent_freq) if base_freq or recent_freq else float("nan")
            status = "OK"
            if np.isfinite(jsd):
                if jsd >= JSD_ALERT:
                    status = "ALERT"
                    summary["categorical_alerts"] += 1
                elif jsd >= JSD_WARN:
                    status = "WARN"
            rows.append(
                {
                    "feature": feat,
                    "type": "categorical",
                    "metric": "JSD",
                    "value": jsd,
                    "status": status,
                }
            )

    return pd.DataFrame(rows), summary


# ------------------- MAE with Actuals (optional) -------------------


def compute_mae_with_actuals(
    df_recent: pd.DataFrame, actuals: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Join on 'external_id' to compute MAE overall and by ward.
    You must include external_id in API requests to use this.
    """
    if "external_id" not in df_recent.columns:
        return pd.DataFrame(), {}
    df_recent = df_recent.dropna(subset=["external_id"]).copy()
    if df_recent.empty or actuals.empty:
        return pd.DataFrame(), {}

    join = pd.merge(df_recent, actuals, on="external_id", how="inner", suffixes=("", "_actual"))
    join = join.dropna(subset=["sale_price_yen", "prediction_yen"])
    if join.empty:
        return pd.DataFrame(), {}

    join["abs_err"] = (join["sale_price_yen"] - join["prediction_yen"]).abs()
    overall = {"MAE": float(join["abs_err"].mean()), "matches": int(len(join))}
    by_ward = (
        join.groupby("city_ward")["abs_err"]
        .mean()
        .sort_values()
        .reset_index()
        .rename(columns={"abs_err": "MAE"})
    )
    return by_ward, overall


# ------------------- Main -------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--create-baseline",
        action="store_true",
        help="Create or overwrite baseline from baseline window",
    )
    ap.add_argument("--baseline-window-days", type=int, default=30, help="Days for baseline window")
    ap.add_argument("--recent-window-days", type=int, default=7, help="Days for recent window")
    ap.add_argument(
        "--with-actuals", action="store_true", help="Join with outcomes if available to compute MAE"
    )
    args = ap.parse_args()

    df = _read_jsonl(PRED_LOG)

    # Latency stats on recent window
    recent = _time_window(df, args.recent_window_days)
    latency = _latency_stats(recent)

    if args.create_baseline:
        base = _time_window(df, args.baseline_window_days)
        if base.empty:
            raise SystemExit("[ERROR] No records in baseline window.")
        create_baseline(base, BASELINE_JSON)
        # fallthrough to also compute drift on the same run

    baseline = load_baseline(BASELINE_JSON)
    drift_df, drift_summary = compute_drift(baseline, recent)

    # Optional MAE
    mae_by_ward_df, mae_overall = pd.DataFrame(), {}
    if args.with_actuals:
        actuals = _load_actuals()
        mae_by_ward_df, mae_overall = compute_mae_with_actuals(recent, actuals)

    # Write reports
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    drift_csv = os.path.join(REPORT_DIR, f"drift_{ts}.csv")
    drift_df.to_csv(drift_csv, index=False, encoding="utf-8")

    summary = {
        "generated_at_utc": ts,
        "recent_window_days": args.recent_window_days,
        "latency": latency,
        "drift_summary": drift_summary,
        "drift_csv": drift_csv,
        "mae_overall": mae_overall,
        "mae_by_ward_rows": int(len(mae_by_ward_df)) if not mae_by_ward_df.empty else 0,
    }
    summary_json = os.path.join(REPORT_DIR, f"summary_{ts}.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not mae_by_ward_df.empty:
        mae_csv = os.path.join(REPORT_DIR, f"mae_by_ward_{ts}.csv")
        mae_by_ward_df.to_csv(mae_csv, index=False, encoding="utf-8")
        summary["mae_by_ward_csv"] = mae_csv
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] Monitoring report written:")
    print("  ", summary_json)
    print("  ", drift_csv)
    if "mae_by_ward_csv" in summary:
        print("  ", summary["mae_by_ward_csv"])
    if latency:
        print(  # noqa: E501
            f"  Latency p50={latency.get('p50_ms', 'n/a'):.0f}ms  p95={latency.get('p95_ms', 'n/a'):.0f}ms  max={latency.get('max_ms', 'n/a'):.0f}ms"
        )


if __name__ == "__main__":
    main()
