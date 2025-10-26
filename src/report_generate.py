#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_generate.py
Builds a single HTML report with:
- Model MAE timeline (models/history/*/metadata.json)
- Current best metrics (models/training_summary.json)
- Drift summary + per-feature metrics (reports/summary_*.json + drift_*.csv)
- Prediction log analysis: volume over time, latency histogram, price histogram (data/predictions/predictions.jsonl)

Run:
  python src/report_generate.py
"""

from __future__ import annotations
import os, io, json, base64, glob
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
HIST_DIR = os.path.join(MODELS_DIR, "history")
PROD_DIR = os.path.join(MODELS_DIR, "production")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PRED_LOG = os.path.join(BASE_DIR, "data", "predictions", "predictions.jsonl")

os.makedirs(REPORTS_DIR, exist_ok=True)

# ------- helpers -------


def _b64_png(fig) -> str:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(bio.getvalue()).decode("ascii")


def _load_json_safely(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _latest_file(pattern: str) -> str | None:
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _read_predictions() -> pd.DataFrame:
    if not os.path.exists(PRED_LOG):
        return pd.DataFrame()
    rows = []
    with open(PRED_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                rows.append(rec)
            except Exception:
                pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # flatten request
    if "request" in df.columns:
        req = pd.json_normalize(df["request"])
        req.columns = [c.split(".")[-1] for c in req.columns]
        df = pd.concat([df.drop(columns=["request"]), req], axis=1)
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce")
        df = df.dropna(subset=["ts_utc"]).sort_values("ts_utc")
    return df


# ------- charts -------


def chart_model_history() -> Tuple[str, pd.DataFrame]:
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(HIST_DIR, "*_*"))):
        meta = _load_json_safely(os.path.join(run_dir, "metadata.json"))
        ts = meta.get("saved_at_utc")
        mae = (meta.get("test_metrics") or {}).get("MAE")
        tag = meta.get("tag", "unknown")
        if ts and mae is not None:
            rows.append({"saved_at_utc": ts, "MAE": float(mae), "tag": tag})
    if not rows:
        fig = plt.figure()
        plt.text(0.1, 0.5, "No model history found", fontsize=12)
        plt.axis("off")
        return _b64_png(fig), pd.DataFrame()

    df = pd.DataFrame(rows)
    df["saved_at_utc"] = pd.to_datetime(df["saved_at_utc"])
    df = df.sort_values("saved_at_utc")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(df["saved_at_utc"], df["MAE"], marker="o")
    ax.set_title("Model MAE over Time (test set)")
    ax.set_xlabel("Saved at (UTC)")
    ax.set_ylabel("MAE (yen)")
    fig.autofmt_xdate()
    return _b64_png(fig), df


def chart_latency(df_pred: pd.DataFrame) -> str:
    if df_pred.empty or "latency_ms" not in df_pred.columns:
        fig = plt.figure()
        plt.text(0.1, 0.5, "No latency data available", fontsize=12)
        plt.axis("off")
        return _b64_png(fig)
    s = pd.to_numeric(df_pred["latency_ms"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(s, bins=30)
    ax.set_title("Latency Distribution (ms)")
    ax.set_xlabel("ms")
    ax.set_ylabel("count")
    return _b64_png(fig)


def chart_volume(df_pred: pd.DataFrame) -> str:
    if df_pred.empty or "ts_utc" not in df_pred.columns:
        fig = plt.figure()
        plt.text(0.1, 0.5, "No prediction timestamps", fontsize=12)
        plt.axis("off")
        return _b64_png(fig)
    df = df_pred.copy()
    df["date"] = df["ts_utc"].dt.date
    daily = df.groupby("date").size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(daily["date"], daily["count"], marker="o")
    ax.set_title("Daily Prediction Volume")
    ax.set_xlabel("date")
    ax.set_ylabel("requests")
    fig.autofmt_xdate()
    return _b64_png(fig)


def chart_price_hist(df_pred: pd.DataFrame) -> str:
    if df_pred.empty or "prediction_yen" not in df_pred.columns:
        fig = plt.figure()
        plt.text(0.1, 0.5, "No predictions yet", fontsize=12)
        plt.axis("off")
        return _b64_png(fig)
    s = pd.to_numeric(df_pred["prediction_yen"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(s, bins=30)
    ax.set_title("Predicted Price Histogram (yen)")
    ax.set_xlabel("yen")
    ax.set_ylabel("count")
    return _b64_png(fig)


def chart_drift(drift_csv_path: str | None) -> Tuple[str, pd.DataFrame]:
    if not drift_csv_path or not os.path.exists(drift_csv_path):
        fig = plt.figure()
        plt.text(0.1, 0.5, "No drift CSV found", fontsize=12)
        plt.axis("off")
        return _b64_png(fig), pd.DataFrame()
    df = pd.read_csv(drift_csv_path)
    # value by feature
    fig, ax = plt.subplots(figsize=(8, 4))
    dd = df.sort_values("value")
    ax.barh(dd["feature"].astype(str), dd["value"].astype(float))
    ax.set_title("Drift metric by feature (PSI/JSD)")
    ax.set_xlabel("metric value")
    return _b64_png(fig), df


# ------- main -------


def main():
    # training summary & prod meta
    training_summary = _load_json_safely(os.path.join(MODELS_DIR, "training_summary.json"))
    prod_meta = _load_json_safely(os.path.join(PROD_DIR, "metadata.json"))

    # model MAE timeline
    history_img, history_df = chart_model_history()

    # latest monitoring summary/drift
    latest_summary = _latest_file(os.path.join(REPORTS_DIR, "summary_*.json"))
    drift_csv = None
    latency_line = ""
    drift_alerts = ""
    if latest_summary:
        summary = _load_json_safely(latest_summary)
        drift_csv = summary.get("drift_csv")
        lat = summary.get("latency", {})
        latency_line = f"Latency p50={lat.get('p50_ms','n/a')}, p95={lat.get('p95_ms','n/a')}, max={lat.get('max_ms','n/a')}"
        dsum = summary.get("drift_summary", {})
        drift_alerts = f"Numeric alerts={dsum.get('numeric_alerts',0)}, Categorical alerts={dsum.get('categorical_alerts',0)}"
    drift_img, drift_df = chart_drift(drift_csv)

    # predictions log charts
    df_pred = _read_predictions()
    vol_img = chart_volume(df_pred)
    lat_img = chart_latency(df_pred)
    price_img = chart_price_hist(df_pred)

    # HTML assemble
    now = datetime.utcnow().isoformat()
    best = training_summary.get("best_model", "n/a")
    best_metrics = training_summary.get("best_metrics", {})
    prod_tag = prod_meta.get("tag", "unknown")
    prod_saved = prod_meta.get("saved_at_utc", "unknown")

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Price Engine Report</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
 h1,h2 {{ margin: 0.2em 0; }}
 .card {{ border: 1px solid #ddd; padding: 16px; margin: 16px 0; border-radius: 8px; }}
 img {{ max-width: 100%; height: auto; }}
 code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
 table {{ border-collapse: collapse; }}
 th, td {{ padding: 6px 10px; border: 1px solid #e5e5e5; }}
 a {{ color: #0366d6; text-decoration: none; }}
 a:hover {{ text-decoration: underline; }}
 a code {{ color: #0366d6; cursor: pointer; }}
 a code:hover {{ background: #e1f5fe; }}
</style>
</head>
<body>
<h1>Real-Estate Price Engine — Analysis Report</h1>
<p>Generated at (UTC): <b>{now}</b></p>

<div class="card">
  <h2>Production Model</h2>
  <p>Tag: <b>{prod_tag}</b>, Saved at (UTC): <b>{prod_saved}</b></p>
  <p>Best in last training run: <b>{best}</b> with MAE={best_metrics.get('MAE','n/a')}, RMSE={best_metrics.get('RMSE','n/a')}, MAPE={best_metrics.get('MAPE','n/a')}%</p>
</div>

<div class="card">
  <h2>Model Performance Timeline</h2>
  <img src="data:image/png;base64,{history_img}" />
</div>

<div class="card">
  <h2>Prediction Volume & Latency</h2>
  <p>{latency_line}</p>
  <img src="data:image/png;base64,{vol_img}" />
  <img src="data:image/png;base64,{lat_img}" />
</div>

<div class="card">
  <h2>Prediction Value Distribution</h2>
  <img src="data:image/png;base64,{price_img}" />
</div>

<div class="card">
  <h2>Data Drift</h2>
  <p>{drift_alerts}</p>
  <img src="data:image/png;base64,{drift_img}" />
  <p>Details: {f'<a href="../{drift_csv}" target="_blank"><code>{drift_csv}</code></a>' if drift_csv else '<code>n/a</code>'}</p>
</div>

<div class="card">
  <h2>Artifacts</h2>
  <ul>
    <li>Training summary: <a href="../models/training_summary.json" target="_blank"><code>models/training_summary.json</code></a></li>
    <li>Production model: <a href="../models/production/metadata.json" target="_blank"><code>models/production/metadata.json</code></a></li>
    <li>Production pipeline: <a href="../models/production/model.joblib" target="_blank" download><code>models/production/model.joblib</code></a></li>
    <li>Prediction logs: <a href="../data/predictions/predictions.jsonl" target="_blank"><code>data/predictions/predictions.jsonl</code></a></li>
    <li>Actuals: <a href="../data/actuals/actuals.csv" target="_blank"><code>data/actuals/actuals.csv</code></a></li>
    <li>Baseline profile: <a href="baseline_profile.json" target="_blank"><code>reports/baseline_profile.json</code></a></li>
  </ul>
  <p><small>💡 Click any link to view the file. JSON files will open in browser, others may download.</small></p>
</div>

</body>
</html>
"""
    out_html = os.path.join(REPORTS_DIR, "price_engine_report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] Report written: {out_html}")


if __name__ == "__main__":
    main()
