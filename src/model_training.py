#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_training.py
- Train baseline Ridge and XGBoost regressors on processed MLIT data
- MLflow tracking (params, metrics, artifacts)
- Save timestamped model artifacts under models/history/
- Update models/production/model.joblib to best-by-MAE

Run:
  python src/model_training.py
Optional:
  MLFLOW_TRACKING_URI=mlruns  mlflow ui
"""

from __future__ import annotations
import os
import sys
import time
import json
import shutil
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

import joblib
import mlflow

# Try to use xgboost if available
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ---------------- paths ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
TRAIN_CSV = os.path.join(PROC_DIR, "processed_train.csv")
TEST_CSV  = os.path.join(PROC_DIR, "processed_test.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
HIST_DIR   = os.path.join(MODELS_DIR, "history")
PROD_DIR   = os.path.join(MODELS_DIR, "production")
os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(PROD_DIR, exist_ok=True)

EXPERIMENT_NAME = "price_estimation_mlit"


# ---------------- helpers ----------------
def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise SystemExit(f"[ERROR] Processed CSVs not found. Run data_prep.py first.\nPaths:\n  {train_path}\n  {test_path}")
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    return train, test


def pick_features(df_cols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Choose columns that typically exist after our data_prep (real MLIT).
    We guard against missing columns by intersecting with actual df columns.
    """
    categorical_candidates = [
        "prefecture",
        "city_ward",
        "property_type",
        "building_structure",
        "age_bucket",
    ]
    numeric_candidates = [
        "effective_area_m2",
        "building_age_years",
        "coverage_ratio",
        "floor_area_ratio",
        "is_tokyo",
    ]
    cats = [c for c in categorical_candidates if c in df_cols]
    nums = [c for c in numeric_candidates if c in df_cols]
    return cats, nums


def build_pipeline(model_name: str,
                   numeric_features: List[str],
                   categorical_features: List[str]):
    """
    Returns a sklearn Pipeline with preprocessing + estimator.
    """
    numeric_tf = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_tf = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_features),
            ("cat", categorical_tf, categorical_features),
        ],
        remainder="drop"
    )

    if model_name == "ridge":
        est = Ridge(alpha=2.0, random_state=42)
    elif model_name == "xgb":
        if not HAS_XGB:
            raise RuntimeError("xgboost not installed")
        est = XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=4,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=0,
            tree_method="hist",
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    pipe = Pipeline(steps=[("preproc", preproc), ("model", est)])
    return pipe


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None)))) * 100.0
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def log_run_to_mlflow(model_name: str, run_params: Dict, run_metrics: Dict, artifacts_dir: str, model_pipeline):
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=model_name):
        # params
        for k, v in run_params.items():
            mlflow.log_param(k, v)
        # metrics
        for k, v in run_metrics.items():
            mlflow.log_metric(k, float(v))
        # artifact: save model inside run (sklearn model)
        local_model_path = os.path.join(artifacts_dir, f"{model_name}_pipeline.joblib")
        joblib.dump(model_pipeline, local_model_path)
        mlflow.log_artifact(local_model_path, artifact_path="model")
        # save a small README per run
        readme_path = os.path.join(artifacts_dir, "MODEL_CARD.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# {model_name} model card\n\n")
            f.write("Inputs: numeric & categorical features as encoded in the pipeline\n")
            f.write("Target: sale_price_yen\n")
            f.write("Metrics: logged in this MLflow run\n")
        mlflow.log_artifact(readme_path, artifact_path="model_card")


def save_as_history_and_maybe_promote(tag: str, model_pipeline, test_metrics: Dict[str, float]) -> str:
    """
    Save model to models/history/<timestamp>_<tag>/model.joblib
    If best MAE so far, copy to models/production/model.joblib and write a small metadata.json
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(HIST_DIR, f"{ts}_{tag}")
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, "model.joblib")
    joblib.dump(model_pipeline, model_path)

    meta = {
        "tag": tag,
        "saved_at_utc": ts,
        "test_metrics": test_metrics,
    }
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Decide promotion by comparing MAE vs current production (if any)
    prod_meta_path = os.path.join(PROD_DIR, "metadata.json")
    promote = True
    if os.path.exists(prod_meta_path):
        try:
            with open(prod_meta_path, "r", encoding="utf-8") as f:
                prod_meta = json.load(f)
            current_mae = float(prod_meta.get("test_metrics", {}).get("MAE", np.inf))
            if not np.isfinite(current_mae):
                current_mae = np.inf
        except Exception:
            current_mae = np.inf
        promote = test_metrics["MAE"] < current_mae

    if promote:
        # update production pointer
        shutil.copyfile(model_path, os.path.join(PROD_DIR, "model.joblib"))
        with open(prod_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return run_dir


def train_one(model_name: str,
              X_train: pd.DataFrame, y_train: np.ndarray,
              X_test: pd.DataFrame, y_test: np.ndarray,
              cats: List[str], nums: List[str]) -> Tuple[Pipeline, Dict[str, float]]:
    pipe = build_pipeline(model_name, numeric_features=nums, categorical_features=cats)

    # Fit
    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predict & metrics
    y_pred = pipe.predict(X_test)
    m = metrics(y_test, y_pred)
    m["TrainSeconds"] = train_time
    return pipe, m


def main():
    # allow overriding paths
    global TRAIN_CSV, TEST_CSV
    if len(sys.argv) >= 3:
        TRAIN_CSV = sys.argv[1]
        TEST_CSV = sys.argv[2]

    # MLflow default local store if not set
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = os.path.join(BASE_DIR, "mlruns")

    # load
    train_df, test_df = load_data(TRAIN_CSV, TEST_CSV)

    # define target and features
    target = "sale_price_yen"
    if target not in train_df.columns:
        raise SystemExit(f"[ERROR] Target column '{target}' missing in processed CSV.")

    cats, nums = pick_features(train_df.columns.tolist())
    if not nums and not cats:
        raise SystemExit("[ERROR] No usable features found. Check processed CSV column names.")

    # drop rows with missing target
    train_df = train_df[train_df[target].notna()].copy()
    test_df  = test_df[test_df[target].notna()].copy()

    X_train = train_df[cats + nums]
    y_train = train_df[target].astype(float).values
    X_test  = test_df[cats + nums]
    y_test  = test_df[target].astype(float).values

    # For MLflow artifacts
    run_artifacts_dir = os.path.join(HIST_DIR, "__tmp_artifacts__")
    os.makedirs(run_artifacts_dir, exist_ok=True)

    results = []

    # 1) Ridge baseline
    ridge_params = {"alpha": 2.0, "model": "ridge"}
    ridge_model, ridge_metrics = train_one("ridge", X_train, y_train, X_test, y_test, cats, nums)
    log_run_to_mlflow("ridge", ridge_params, ridge_metrics, run_artifacts_dir, ridge_model)
    ridge_dir = save_as_history_and_maybe_promote("ridge", ridge_model, ridge_metrics)
    results.append(("ridge", ridge_metrics, ridge_dir))

    # 2) XGBoost (if available)
    if HAS_XGB:
        xgb_params = {
            "n_estimators": 600,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_lambda": 2.0,
            "model": "xgb",
        }
        xgb_model, xgb_metrics = train_one("xgb", X_train, y_train, X_test, y_test, cats, nums)
        log_run_to_mlflow("xgb", xgb_params, xgb_metrics, run_artifacts_dir, xgb_model)
        xgb_dir = save_as_history_and_maybe_promote("xgb", xgb_model, xgb_metrics)
        results.append(("xgb", xgb_metrics, xgb_dir))
    else:
        print("[WARN] xgboost not installed, skipping XGBRegressor")

    # Which won?
    results.sort(key=lambda x: x[1]["MAE"])
    best_name, best_metrics, best_dir = results[0]
    print("\n[SUMMARY] Test-set metrics")
    for name, m, d in results:
        print(f"  {name:5s}  MAE={m['MAE']:.0f}  RMSE={m['RMSE']:.0f}  MAPE={m['MAPE']:.2f}%  -> {d}")

    # Write summary json
    summary = {
        "best_model": best_name,
        "best_metrics": best_metrics,
        "trained_at_utc": datetime.utcnow().isoformat(),
        "features": {"categorical": cats, "numeric": nums},
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "production_pointer": os.path.join(PROD_DIR, "model.joblib"),
    }
    with open(os.path.join(MODELS_DIR, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # cleanup tmp artifacts dir
    shutil.rmtree(run_artifacts_dir, ignore_errors=True)

    print("\n[OK] Training finished.")
    print(f"Best: {best_name}  MAE={best_metrics['MAE']:.0f}  RMSE={best_metrics['RMSE']:.0f}  MAPE={best_metrics['MAPE']:.2f}%")
    print(f"Production pointer: {summary['production_pointer']}")
    print("MLflow tracking dir:", os.environ.get("MLFLOW_TRACKING_URI", "<unset>"))
    print("Inspect runs with:   mlflow ui   (then open http://127.0.0.1:5000)")
    

if __name__ == "__main__":
    main()

