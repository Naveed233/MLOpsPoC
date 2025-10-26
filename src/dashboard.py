#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dashboard.py
Interactive Streamlit dashboard for real-estate price engine monitoring

Features:
- Model performance timeline
- Drift monitoring with alerts
- Prediction volume and latency analysis
- MAE by ward (when actuals available)
- Interactive filters: date range, ward, property type

Run:
  streamlit run src/dashboard.py
"""

import os
import json
import glob
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Config
st.set_page_config(page_title="Price Engine Dashboard", page_icon="🏠", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
HIST_DIR = os.path.join(MODELS_DIR, "history")
PROD_DIR = os.path.join(MODELS_DIR, "production")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PRED_LOG = os.path.join(BASE_DIR, "data", "predictions", "predictions.jsonl")
ACTUALS_CSV = os.path.join(BASE_DIR, "data", "actuals", "actuals.csv")


# Helper functions
@st.cache_data
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@st.cache_data
def load_predictions():
    if not os.path.exists(PRED_LOG):
        return pd.DataFrame()
    rows = []
    with open(PRED_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "request" in df.columns:
        req = pd.json_normalize(df["request"])
        req.columns = [c.split(".")[-1] for c in req.columns]
        df = pd.concat([df.drop(columns=["request"]), req], axis=1)
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce")
        df = df.dropna(subset=["ts_utc"]).sort_values("ts_utc")
    return df


@st.cache_data
def load_model_history():
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(HIST_DIR, "*_*"))):
        meta = load_json(os.path.join(run_dir, "metadata.json"))
        ts = meta.get("saved_at_utc")
        metrics = meta.get("test_metrics", {})
        tag = meta.get("tag", "unknown")
        if ts and metrics.get("MAE") is not None:
            rows.append(
                {
                    "saved_at_utc": pd.to_datetime(ts),
                    "MAE": float(metrics.get("MAE")),
                    "RMSE": float(metrics.get("RMSE", 0)),
                    "MAPE": float(metrics.get("MAPE", 0)),
                    "tag": tag,
                }
            )
    return pd.DataFrame(rows).sort_values("saved_at_utc") if rows else pd.DataFrame()


@st.cache_data
def load_drift_reports():
    summaries = glob.glob(os.path.join(REPORTS_DIR, "summary_*.json"))
    if not summaries:
        return pd.DataFrame()
    rows = []
    for path in sorted(summaries):
        summary = load_json(path)
        ts = summary.get("generated_at_utc")
        if ts:
            rows.append(
                {
                    "timestamp": pd.to_datetime(ts),
                    "numeric_alerts": summary.get("drift_summary", {}).get("numeric_alerts", 0),
                    "categorical_alerts": summary.get("drift_summary", {}).get(
                        "categorical_alerts", 0
                    ),
                    "latency_p50": summary.get("latency", {}).get("p50_ms", 0),
                    "latency_p95": summary.get("latency", {}).get("p95_ms", 0),
                }
            )
    return pd.DataFrame(rows).sort_values("timestamp") if rows else pd.DataFrame()


@st.cache_data
def load_actuals():
    if not os.path.exists(ACTUALS_CSV):
        return pd.DataFrame()
    return pd.read_csv(ACTUALS_CSV)


def get_latest_drift_csv():
    pattern = os.path.join(REPORTS_DIR, "drift_*.csv")
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


# Main dashboard
def main():
    st.title("🏠 Real Estate Price Engine Dashboard")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("Filters")

    # Load data
    df_pred = load_predictions()
    df_history = load_model_history()
    df_drift = load_drift_reports()

    # Date filter
    if not df_pred.empty and "ts_utc" in df_pred.columns:
        min_date = df_pred["ts_utc"].min().date()
        max_date = df_pred["ts_utc"].max().date()
        # Ensure default start date doesn't go below min_date
        default_start = max(min_date, max_date - timedelta(days=7))
        date_range = st.sidebar.date_input(
            "Date Range", value=(default_start, max_date), min_value=min_date, max_value=max_date
        )
        if len(date_range) == 2:
            df_pred = df_pred[
                (df_pred["ts_utc"].dt.date >= date_range[0])
                & (df_pred["ts_utc"].dt.date <= date_range[1])
            ]

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

    # Metrics row - FILTERED DATA
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Filtered Predictions",
            len(df_pred),
            help="Number of predictions matching current filters",
        )

    with col2:
        if not df_pred.empty and "latency_ms" in df_pred.columns:
            p50 = df_pred["latency_ms"].median()
            st.metric("Latency p50", f"{p50:.1f}ms", help="Median latency for filtered predictions")
        else:
            st.metric("Latency p50", "N/A")

    with col3:
        if not df_pred.empty and "prediction_yen" in df_pred.columns:
            avg_pred = df_pred["prediction_yen"].mean()
            min_pred = df_pred["prediction_yen"].min()
            max_pred = df_pred["prediction_yen"].max()
            st.metric(
                "Avg Prediction",
                f"¥{avg_pred/1e6:.1f}M",
                help=f"Range: ¥{min_pred/1e6:.1f}M - ¥{max_pred/1e6:.1f}M",
            )
        else:
            st.metric("Avg Prediction", "N/A")

    with col4:
        if not df_drift.empty:
            latest_alerts = df_drift.iloc[-1]
            total_alerts = int(
                latest_alerts["numeric_alerts"] + latest_alerts["categorical_alerts"]
            )
            st.metric(
                "Drift Alerts",
                total_alerts,
                delta="⚠️" if total_alerts > 0 else "✅",
                help="Global drift status (not filtered)",
            )
        else:
            st.metric("Drift Alerts", "N/A")

    st.markdown("---")

    # Model Performance Section
    st.header("📈 Model Performance")

    # Calculate MAE on filtered data if actuals available
    actuals_for_performance = load_actuals()
    if not actuals_for_performance.empty and "external_id" in df_pred.columns:
        merged_perf = pd.merge(
            df_pred,
            actuals_for_performance,
            on="external_id",
            how="inner",
            suffixes=("", "_actual"),
        )
        if (
            not merged_perf.empty
            and "sale_price_yen" in merged_perf.columns
            and "prediction_yen" in merged_perf.columns
        ):
            merged_perf["abs_error"] = (
                merged_perf["sale_price_yen"] - merged_perf["prediction_yen"]
            ).abs()
            merged_perf["pct_error"] = (
                merged_perf["abs_error"] / merged_perf["sale_price_yen"]
            ) * 100

            # Show live metrics based on FILTERED data
            st.info(
                f"📊 **Live Performance Metrics** (based on {len(merged_perf)} matched actuals in filtered data)"
            )
            col_live1, col_live2, col_live3, col_live4 = st.columns(4)

            with col_live1:
                filtered_mae = merged_perf["abs_error"].mean()
                st.metric(
                    "Filtered MAE",
                    f"¥{filtered_mae/1e6:.2f}M",
                    help="Mean Absolute Error on filtered predictions",
                )

            with col_live2:
                filtered_rmse = np.sqrt((merged_perf["abs_error"] ** 2).mean())
                st.metric(
                    "Filtered RMSE",
                    f"¥{filtered_rmse/1e6:.2f}M",
                    help="Root Mean Squared Error on filtered predictions",
                )

            with col_live3:
                filtered_mape = merged_perf["pct_error"].mean()
                st.metric(
                    "Filtered MAPE",
                    f"{filtered_mape:.2f}%",
                    help="Mean Absolute Percentage Error on filtered predictions",
                )

            with col_live4:
                filtered_median_err = merged_perf["abs_error"].median()
                st.metric(
                    "Median Error",
                    f"¥{filtered_median_err/1e6:.2f}M",
                    help="Median absolute error on filtered predictions",
                )

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
        st.caption("📌 Static metrics from training (not affected by filters)")
        prod_meta = load_json(os.path.join(PROD_DIR, "metadata.json"))
        if prod_meta:
            st.write(f"**Tag:** {prod_meta.get('tag', 'unknown')}")
            st.write(f"**Saved:** {prod_meta.get('saved_at_utc', 'unknown')}")
            metrics = prod_meta.get("test_metrics", {})
            st.write(f"**Test MAE:** ¥{metrics.get('MAE', 0)/1e6:.2f}M")
            st.write(f"**Test RMSE:** ¥{metrics.get('RMSE', 0)/1e6:.2f}M")
            st.write(f"**Test MAPE:** {metrics.get('MAPE', 0):.2f}%")
            st.caption(f"Evaluated on {13737} train / {1263} test samples")
        else:
            st.info("No production model metadata")

    st.markdown("---")

    # Prediction Analysis Section
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
                ax.axvline(
                    latency.median(),
                    color="r",
                    linestyle="--",
                    label=f"Median: {latency.median():.1f}ms",
                )
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
                ax.grid(True, alpha=0.3, axis="x")
                st.pyplot(fig)
    else:
        st.info("No prediction data available for selected filters")

    st.markdown("---")

    # Drift Monitoring Section
    st.header("⚠️ Drift Monitoring")

    drift_csv = get_latest_drift_csv()
    if drift_csv:
        df_drift_detail = pd.read_csv(drift_csv)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Drift by Feature")
            fig, ax = plt.subplots(figsize=(10, 6))
            sorted_df = df_drift_detail.sort_values("value")
            colors = [
                "red" if s == "ALERT" else "orange" if s == "WARN" else "green"
                for s in sorted_df["status"]
            ]
            ax.barh(sorted_df["feature"], sorted_df["value"], color=colors, alpha=0.7)
            ax.set_xlabel("Drift Metric Value")
            ax.set_ylabel("Feature")
            ax.grid(True, alpha=0.3, axis="x")
            st.pyplot(fig)

        with col2:
            st.subheader("Drift Status")
            status_counts = df_drift_detail["status"].value_counts()
            for status, count in status_counts.items():
                if status == "ALERT":
                    st.error(f"🔴 {status}: {count}")
                elif status == "WARN":
                    st.warning(f"🟡 {status}: {count}")
                else:
                    st.success(f"🟢 {status}: {count}")

            st.subheader("Details")
            st.dataframe(df_drift_detail, use_container_width=True, height=200)
    else:
        st.info("No drift reports available. Run monitor_drift.py first.")

    st.markdown("---")

    # MAE by Ward Section (if actuals available)
    st.header("📊 Actual vs Predicted (MAE by Ward)")

    actuals = load_actuals()
    if not actuals.empty and "external_id" in df_pred.columns and "external_id" in actuals.columns:
        merged = pd.merge(
            df_pred, actuals, on="external_id", how="inner", suffixes=("_pred", "_actual")
        )
        if (
            not merged.empty
            and "sale_price_yen" in merged.columns
            and "prediction_yen" in merged.columns
        ):
            merged["abs_error"] = (merged["sale_price_yen"] - merged["prediction_yen"]).abs()

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("MAE by Ward")
                # Check if city_ward column exists in merged data
                if "city_ward" in merged.columns:
                    mae_by_ward = merged.groupby("city_ward")["abs_error"].mean().sort_values()
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.barh(mae_by_ward.index, mae_by_ward.values / 1e6)
                    ax.set_xlabel("MAE (¥M)")
                    ax.set_ylabel("Ward")
                    ax.grid(True, alpha=0.3, axis="x")
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
    st.caption(f"Dashboard refreshed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
