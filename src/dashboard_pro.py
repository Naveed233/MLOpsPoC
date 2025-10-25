#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dashboard_pro.py
PRODUCTION-GRADE Interactive Streamlit dashboard

NEW FEATURES:
✅ Model versioning with git commit hashes
✅ Automated retraining triggers based on drift thresholds
✅ System resource metrics (CPU, memory, cost)
✅ CI/CD deployment timeline
✅ Model explainability (feature importance)
"""

import os
import json
import glob
import subprocess
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psutil  # System metrics
import joblib  # Load model for feature importance

# Config
st.set_page_config(page_title="🚀 Production Price Engine", page_icon="🏠", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
HIST_DIR = os.path.join(MODELS_DIR, "history")
PROD_DIR = os.path.join(MODELS_DIR, "production")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PRED_LOG = os.path.join(BASE_DIR, "data", "predictions", "predictions.jsonl")
ACTUALS_CSV = os.path.join(BASE_DIR, "data", "actuals", "actuals.csv")

# Thresholds for automated retraining
PSI_THRESHOLD = 0.1
JSD_THRESHOLD = 0.1
MAE_DEGRADATION_THRESHOLD = 1.2  # 20% worse than baseline

#######################
# HELPER FUNCTIONS
#######################

@st.cache_data
def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_predictions():
    """Load prediction logs"""
    if not os.path.exists(PRED_LOG):
        return pd.DataFrame()
    
    predictions = []
    with open(PRED_LOG, 'r') as f:
        for line in f:
            try:
                predictions.append(json.loads(line))
            except:
                pass
    
    if not predictions:
        return pd.DataFrame()
    
    df = pd.DataFrame(predictions)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"])
    
    # Extract nested request fields
    if "request" in df.columns:
        req_df = pd.json_normalize(df["request"])
        for col in req_df.columns:
            df[col] = req_df[col]
    
    return df

@st.cache_data
def load_model_history():
    """Load model training history"""
    history_dirs = sorted(glob.glob(os.path.join(HIST_DIR, "*")))
    
    records = []
    for d in history_dirs:
        meta_path = os.path.join(d, "metadata.json")
        if os.path.exists(meta_path):
            records.append(load_json(meta_path))
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    if "saved_at_utc" in df.columns:
        df["saved_at_utc"] = pd.to_datetime(df["saved_at_utc"])
    if "test_metrics" in df.columns:
        for metric in ["MAE", "RMSE", "MAPE"]:
            df[metric] = df["test_metrics"].apply(lambda x: x.get(metric, 0) if isinstance(x, dict) else 0)
    
    return df

@st.cache_data
def load_drift_reports():
    """Load drift monitoring reports"""
    drift_files = sorted(glob.glob(os.path.join(REPORTS_DIR, "drift_*.csv")))
    
    if not drift_files:
        return pd.DataFrame()
    
    all_reports = []
    for f in drift_files:
        df = pd.read_csv(f)
        # Extract timestamp from filename: drift_20251024T123456Z.csv
        ts = os.path.basename(f).replace("drift_", "").replace(".csv", "")
        df["report_ts"] = ts
        all_reports.append(df)
    
    combined = pd.concat(all_reports, ignore_index=True)
    combined["report_ts"] = pd.to_datetime(combined["report_ts"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    
    # Group by report to get summary stats
    summary = combined.groupby("report_ts").agg({
        "value": ["mean", "max"],
        "status": lambda x: (x == "ALERT").sum()
    }).reset_index()
    summary.columns = ["report_ts", "mean_drift", "max_drift", "num_alerts"]
    
    # Also keep full detail
    combined["numeric_alerts"] = combined.apply(lambda x: 1 if x["type"] == "numeric" and x["status"] == "ALERT" else 0, axis=1)
    combined["categorical_alerts"] = combined.apply(lambda x: 1 if x["type"] == "categorical" and x["status"] == "ALERT" else 0, axis=1)
    
    return combined

@st.cache_data
def load_actuals():
    """Load actual outcomes"""
    if os.path.exists(ACTUALS_CSV):
        return pd.read_csv(ACTUALS_CSV)
    return pd.DataFrame()

def get_system_metrics():
    """Get current system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "cpu_percent": cpu_percent,
        "memory_used_gb": memory.used / (1024**3),
        "memory_total_gb": memory.total / (1024**3),
        "memory_percent": memory.percent,
        "disk_used_gb": disk.used / (1024**3),
        "disk_total_gb": disk.total / (1024**3),
        "disk_percent": disk.percent,
    }

def get_git_commit():
    """Get current git commit hash (short)"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return "no-git"

def get_feature_importance():
    """Extract feature importance from production model"""
    try:
        model_path = os.path.join(PROD_DIR, "model.joblib")
        if not os.path.exists(model_path):
            return None
        
        pipeline = joblib.load(model_path)
        
        # Extract the final estimator
        if hasattr(pipeline, 'named_steps'):
            model = pipeline.named_steps.get('model') or pipeline.named_steps.get('regressor')
        else:
            model = pipeline
        
        # Get feature names from preprocessor
        if hasattr(pipeline, 'named_steps') and 'preprocessor' in pipeline.named_steps:
            preprocessor = pipeline.named_steps['preprocessor']
            feature_names = []
            
            if hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = preprocessor.get_feature_names_out()
            elif hasattr(preprocessor, 'transformers_'):
                for name, trans, cols in preprocessor.transformers_:
                    if name == 'drop':
                        continue
                    if hasattr(trans, 'get_feature_names_out'):
                        feature_names.extend(trans.get_feature_names_out(cols))
                    else:
                        feature_names.extend(cols)
        
        # Get importances
        importances = None
        if hasattr(model, 'feature_importances_'):  # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):  # Linear models
            importances = np.abs(model.coef_)
        
        if importances is not None and len(feature_names) == len(importances):
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
        
    except Exception as e:
        st.sidebar.error(f"Error loading feature importance: {e}")
    
    return None

def check_retraining_triggers(df_drift, df_history, df_pred, actuals):
    """Check if automated retraining should be triggered"""
    triggers = []
    
    # 1. Check drift thresholds
    if not df_drift.empty:
        latest_drift = df_drift.sort_values('report_ts').tail(8)  # Last 8 features
        
        psi_violations = latest_drift[(latest_drift['metric'] == 'PSI') & (latest_drift['value'] > PSI_THRESHOLD)]
        jsd_violations = latest_drift[(latest_drift['metric'] == 'JSD') & (latest_drift['value'] > JSD_THRESHOLD)]
        
        if not psi_violations.empty:
            triggers.append(f"🔴 PSI drift detected in {len(psi_violations)} features")
        if not jsd_violations.empty:
            triggers.append(f"🔴 JSD drift detected in {len(jsd_violations)} features")
    
    # 2. Check MAE degradation (if we have actuals)
    if not actuals.empty and not df_pred.empty and 'external_id' in df_pred.columns:
        merged = pd.merge(df_pred, actuals, on='external_id', how='inner', suffixes=('', '_actual'))
        if not merged.empty and 'sale_price_yen' in merged.columns:
            current_mae = np.mean(np.abs(merged['prediction_yen'] - merged['sale_price_yen']))
            
            if not df_history.empty:
                baseline_mae = df_history.iloc[0]['MAE']  # First model MAE
                
                if current_mae > baseline_mae * MAE_DEGRADATION_THRESHOLD:
                    triggers.append(f"🔴 MAE degraded: ¥{current_mae/1e6:.1f}M vs baseline ¥{baseline_mae/1e6:.1f}M")
    
    # 3. Check prediction volume spike
    if not df_pred.empty and 'ts_utc' in df_pred.columns:
        daily_counts = df_pred.groupby(df_pred['ts_utc'].dt.date).size()
        if len(daily_counts) > 1:
            recent_avg = daily_counts.tail(3).mean()
            historical_avg = daily_counts.mean()
            
            if recent_avg > historical_avg * 2:
                triggers.append(f"⚠️ Prediction volume spike: {recent_avg:.0f} vs {historical_avg:.0f} avg")
    
    return triggers

def estimate_cost_per_1k(latency_ms):
    """Estimate cost per 1000 predictions (simplified)"""
    # Assumptions:
    # - AWS Lambda: $0.20 per 1M requests + $0.0000166667 per GB-second
    # - Average 512MB memory, ~latency_ms duration
    compute_cost_per_request = (0.0000166667 * 0.5 * (latency_ms / 1000))
    request_cost_per_request = 0.20 / 1_000_000
    
    total_per_request = compute_cost_per_request + request_cost_per_request
    cost_per_1k = total_per_1k = total_per_request * 1000
    
    return round(cost_per_1k, 4)

#######################
# MAIN DASHBOARD
#######################

def main():
    st.title("🚀 Real Estate Price Engine - Production Dashboard")
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    # Load data
    df_pred_all = load_predictions()
    df_history = load_model_history()
    df_drift = load_drift_reports()
    actuals_all = load_actuals()
    
    # Apply filters
    df_pred = df_pred_all.copy()
    
    # Date filter
    if not df_pred.empty and "ts_utc" in df_pred.columns:
        min_date = df_pred["ts_utc"].min().date()
        max_date = df_pred["ts_utc"].max().date()
        default_start = max(min_date, max_date - timedelta(days=7))
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date
        )
        if len(date_range) == 2:
            df_pred = df_pred[(df_pred["ts_utc"].dt.date >= date_range[0]) & 
                             (df_pred["ts_utc"].dt.date <= date_range[1])]
    
    # Ward filter
    if not df_pred.empty and "city_ward" in df_pred.columns:
        wards = ["All"] + sorted(df_pred["city_ward"].dropna().unique().tolist())
        selected_ward = st.sidebar.selectbox("City/Ward", wards)
        if selected_ward != "All":
            df_pred = df_pred[df_pred["city_ward"] == selected_ward]
    
    # Property type filter
    if not df_pred.empty and "property_type" in df_pred.columns:
        prop_types = ["All"] + sorted(df_pred["property_type"].dropna().unique().tolist())
        selected_type = st.sidebar.selectbox("Property Type", prop_types)
        if selected_type != "All":
            df_pred = df_pred[df_pred["property_type"] == selected_type]
    
    #######################
    # NEW: SYSTEM STATUS BAR
    #######################
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    sys_metrics = get_system_metrics()
    
    with col1:
        st.metric("🖥️ CPU Usage", f"{sys_metrics['cpu_percent']:.1f}%")
    
    with col2:
        st.metric("💾 Memory", f"{sys_metrics['memory_used_gb']:.1f}/{sys_metrics['memory_total_gb']:.0f}GB")
    
    with col3:
        git_commit = get_git_commit()
        st.metric("🔖 Version", git_commit[:7] if len(git_commit) > 7 else git_commit)
    
    with col4:
        if not df_pred.empty and 'latency_ms' in df_pred.columns:
            avg_latency = df_pred['latency_ms'].mean()
            cost_per_1k = estimate_cost_per_1k(avg_latency)
            st.metric("💰 Cost/1K", f"${cost_per_1k:.4f}")
        else:
            st.metric("💰 Cost/1K", "N/A")
    
    #######################
    # NEW: RETRAINING TRIGGERS
    #######################
    st.markdown("---")
    st.header("🔄 Automated Retraining Status")
    
    triggers = check_retraining_triggers(df_drift, df_history, df_pred_all, actuals_all)
    
    if triggers:
        st.warning("⚠️ **Retraining Recommended**")
        for trigger in triggers:
            st.write(f"  {trigger}")
        st.button("🚀 Trigger Retraining Pipeline", type="primary", disabled=True, help="Would trigger CI/CD pipeline in production")
    else:
        st.success("✅ All metrics within acceptable ranges. No retraining needed.")
        st.caption(f"Thresholds: PSI < {PSI_THRESHOLD}, JSD < {JSD_THRESHOLD}, MAE < {MAE_DEGRADATION_THRESHOLD}x baseline")
    
    #######################
    # PREDICTION METRICS
    #######################
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Filtered Predictions", len(df_pred), 
                 help="Number of predictions matching current filters")
    
    with col2:
        if not df_pred.empty and "latency_ms" in df_pred.columns:
            p50 = df_pred["latency_ms"].median()
            p95 = df_pred["latency_ms"].quantile(0.95)
            st.metric("Latency p50/p95", f"{p50:.1f}/{p95:.1f}ms")
        else:
            st.metric("Latency p50/p95", "N/A")
    
    with col3:
        if not df_pred.empty and "prediction_yen" in df_pred.columns:
            avg_pred = df_pred["prediction_yen"].mean()
            st.metric("Avg Prediction", f"¥{avg_pred/1e6:.1f}M")
        else:
            st.metric("Avg Prediction", "N/A")
    
    with col4:
        if not df_drift.empty:
            latest_report = df_drift.sort_values('report_ts').tail(8)
            total_alerts = (latest_report["status"] == "ALERT").sum()
            st.metric("Drift Alerts", total_alerts, 
                     delta="⚠️" if total_alerts > 0 else "✅")
        else:
            st.metric("Drift Alerts", "N/A")
    
    #######################
    # MODEL PERFORMANCE
    #######################
    st.markdown("---")
    st.header("📈 Model Performance")
    
    # Calculate MAE on filtered data if actuals available
    actuals_for_performance = load_actuals()
    if not actuals_for_performance.empty and "external_id" in df_pred.columns:
        merged_perf = pd.merge(df_pred, actuals_for_performance, on="external_id", how="inner", suffixes=("", "_actual"))
        if not merged_perf.empty and "sale_price_yen" in merged_perf.columns and "prediction_yen" in merged_perf.columns:
            merged_perf["abs_error"] = (merged_perf["sale_price_yen"] - merged_perf["prediction_yen"]).abs()
            merged_perf["pct_error"] = (merged_perf["abs_error"] / merged_perf["sale_price_yen"]) * 100
            
            st.info(f"📊 **Live Performance Metrics** (based on {len(merged_perf)} matched actuals in filtered data)")
            col_live1, col_live2, col_live3, col_live4 = st.columns(4)
            
            with col_live1:
                filtered_mae = merged_perf["abs_error"].mean()
                st.metric("Filtered MAE", f"¥{filtered_mae/1e6:.2f}M")
            
            with col_live2:
                filtered_rmse = np.sqrt((merged_perf["abs_error"] ** 2).mean())
                st.metric("Filtered RMSE", f"¥{filtered_rmse/1e6:.2f}M")
            
            with col_live3:
                filtered_mape = merged_perf["pct_error"].mean()
                st.metric("Filtered MAPE", f"{filtered_mape:.2f}%")
            
            with col_live4:
                filtered_median_err = merged_perf["abs_error"].median()
                st.metric("Median Error", f"¥{filtered_median_err/1e6:.2f}M")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if not df_history.empty:
            st.subheader("MAE Timeline")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_history["saved_at_utc"], df_history["MAE"], marker="o", linewidth=2)
            ax.set_ylabel("MAE (¥)")
            ax.set_xlabel("Model Version")
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            st.pyplot(fig)
        else:
            st.info("No model history available")
    
    with col2:
        st.subheader("Current Production Model")
        st.caption("📌 Static metrics from training")
        prod_meta = load_json(os.path.join(PROD_DIR, "metadata.json"))
        if prod_meta:
            st.write(f"**Tag:** {prod_meta.get('tag', 'unknown')}")
            st.write(f"**Version:** {prod_meta.get('commit_hash', get_git_commit())[:7]}")
            st.write(f"**Saved:** {prod_meta.get('saved_at_utc', 'unknown')[:16]}")
            metrics = prod_meta.get("test_metrics", {})
            st.write(f"**Test MAE:** ¥{metrics.get('MAE', 0)/1e6:.2f}M")
            st.write(f"**Test RMSE:** ¥{metrics.get('RMSE', 0)/1e6:.2f}M")
            st.write(f"**Test MAPE:** {metrics.get('MAPE', 0):.2f}%")
        else:
            st.info("No production model metadata")
    
    #######################
    # NEW: FEATURE IMPORTANCE
    #######################
    st.markdown("---")
    st.header("🧠 Model Explainability")
    
    feature_importance_df = get_feature_importance()
    
    if feature_importance_df is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Top 10 Feature Importances")
            top_features = feature_importance_df.head(10)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(top_features['feature'], top_features['importance'])
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Top Features")
            st.dataframe(
                feature_importance_df.head(10).style.format({'importance': '{:.4f}'}),
                use_container_width=True
            )
    else:
        st.info("💡 Feature importance not available. Train an XGBoost model for better explainability.")
    
    #######################
    # NEW: CI/CD DEPLOYMENT HISTORY
    #######################
    st.markdown("---")
    st.header("🚢 Deployment History")
    
    if not df_history.empty:
        st.subheader("Last 5 Deployments")
        
        deployment_data = []
        for _, row in df_history.tail(5).iterrows():
            deployment_data.append({
                "Timestamp": row.get('saved_at_utc', 'N/A'),
                "Model": row.get('tag', 'N/A'),
                "MAE (¥M)": f"{row.get('MAE', 0)/1e6:.2f}",
                "MAPE (%)": f"{row.get('MAPE', 0):.2f}",
                "Status": "✅ Deployed"
            })
        
        st.dataframe(pd.DataFrame(deployment_data), use_container_width=True)
        
        # Deployment timeline chart
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.scatter(df_history["saved_at_utc"], df_history["MAE"]/1e6, s=100, c=range(len(df_history)), cmap='viridis')
        ax.plot(df_history["saved_at_utc"], df_history["MAE"]/1e6, alpha=0.5)
        ax.set_ylabel("MAE (¥M)")
        ax.set_xlabel("Deployment Time")
        ax.set_title("Model Performance Over Deployments")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        st.pyplot(fig)
    else:
        st.info("No deployment history available")
    
    #######################
    # PREDICTION ANALYSIS
    #######################
    st.markdown("---")
    st.header("🔍 Prediction Analysis")
    
    if not df_pred.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Daily Volume")
            daily = df_pred.groupby(df_pred["ts_utc"].dt.date).size().reset_index(name="count")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(daily["ts_utc"], daily["count"], marker="o", linewidth=2)
            ax.set_ylabel("Requests")
            ax.set_xlabel("Date")
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Latency Distribution")
            if "latency_ms" in df_pred.columns:
                fig, ax = plt.subplots(figsize=(8, 4))
                latency = df_pred["latency_ms"].dropna()
                ax.hist(latency, bins=30, edgecolor="black", alpha=0.7)
                ax.axvline(latency.median(), color='r', linestyle='--', label=f'Median: {latency.median():.1f}ms')
                ax.set_xlabel("Latency (ms)")
                ax.set_ylabel("Count")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price Distribution")
            if "prediction_yen" in df_pred.columns:
                fig, ax = plt.subplots(figsize=(8, 4))
                prices = df_pred["prediction_yen"].dropna() / 1e6
                ax.hist(prices, bins=30, edgecolor="black", alpha=0.7)
                ax.set_xlabel("Predicted Price (¥M)")
                ax.set_ylabel("Count")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        with col2:
            st.subheader("Predictions by Ward (Top 10)")
            if "city_ward" in df_pred.columns:
                ward_counts = df_pred["city_ward"].value_counts().head(10)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(ward_counts.index, ward_counts.values)
                ax.set_xlabel("Count")
                ax.set_ylabel("Ward")
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig)
    
    #######################
    # DRIFT MONITORING
    #######################
    st.markdown("---")
    st.header("⚠️ Drift Monitoring")
    
    if not df_drift.empty:
        latest_drift = df_drift.sort_values('report_ts').tail(8)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Drift by Feature")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(latest_drift['feature'], latest_drift['value'])
            ax.set_xlabel("Drift Metric Value")
            ax.set_ylabel("Feature")
            ax.axvline(PSI_THRESHOLD, color='orange', linestyle='--', label=f'Threshold ({PSI_THRESHOLD})')
            ax.invert_yaxis()
            ax.legend()
            ax.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Drift Status")
            ok_count = (latest_drift['status'] == 'OK').sum()
            alert_count = (latest_drift['status'] == 'ALERT').sum()
            
            st.metric("✅ OK", ok_count)
            if alert_count > 0:
                st.metric("🔴 ALERT", alert_count)
            
            st.caption(f"Last checked: {latest_drift['report_ts'].max()}")
        
        st.subheader("Details")
        st.dataframe(latest_drift[['feature', 'type', 'metric', 'value', 'status']], use_container_width=True)
    else:
        st.info("No drift reports available. Run: python src/monitor_drift.py")
    
    #######################
    # ACTUAL VS PREDICTED
    #######################
    st.markdown("---")
    st.header("✅ Actual vs Predicted (MAE by Ward)")
    
    actuals = load_actuals()
    if not actuals.empty and "external_id" in df_pred.columns and "external_id" in actuals.columns:
        merged = pd.merge(df_pred, actuals, on="external_id", how="inner", suffixes=("_pred", "_actual"))
        if not merged.empty and "sale_price_yen" in merged.columns and "prediction_yen" in merged.columns:
            merged["abs_error"] = (merged["sale_price_yen"] - merged["prediction_yen"]).abs()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("MAE by Ward")
                if "city_ward" in merged.columns:
                    mae_by_ward = merged.groupby("city_ward")["abs_error"].mean().sort_values()
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.barh(mae_by_ward.index, mae_by_ward.values / 1e6)
                    ax.set_xlabel("MAE (¥M)")
                    ax.set_ylabel("Ward")
                    ax.grid(True, alpha=0.3, axis='x')
                    st.pyplot(fig)
                else:
                    st.info("💡 Add 'city_ward' column to predictions for ward-level MAE analysis")
            
            with col2:
                st.subheader("Overall Metrics")
                overall_mae = merged["abs_error"].mean()
                st.metric("Overall MAE", f"¥{overall_mae/1e6:.2f}M")
                st.metric("Matched Records", len(merged))
                
                if "city_ward" in merged.columns:
                    mae_by_ward = merged.groupby("city_ward")["abs_error"].mean().sort_values()
                    st.subheader("Top/Bottom Wards")
                    st.write("**Best (Lowest MAE):**")
                    for ward, mae in mae_by_ward.head(3).items():
                        st.write(f"• {ward}: ¥{mae/1e6:.2f}M")
                    st.write("**Worst (Highest MAE):**")
                    for ward, mae in mae_by_ward.tail(3).items():
                        st.write(f"• {ward}: ¥{mae/1e6:.2f}M")
        else:
            st.info("No matched predictions with actuals yet")
    else:
        st.info("No actuals available. Add data to data/actuals/actuals.csv to see MAE analysis.")
    
    # Footer
    st.markdown("---")
    st.caption(f"Dashboard refreshed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Version: {get_git_commit()[:7]}")

if __name__ == "__main__":
    main()

