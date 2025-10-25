#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_prep.py (real-data first)
- Prefer official MLIT transaction CSVs in data/raw/
- Clean & engineer features for JP residential valuation
- Time-aware train/test split (last 6 months of data → test)
- Save processed CSVs + a data dictionary

Run:
  python src/data_prep.py
"""

from __future__ import annotations
import os
import re
import json
from glob import glob
from datetime import datetime
from typing import Tuple, Optional

import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
TRAIN_OUT = os.path.join(PROC_DIR, "processed_train.csv")
TEST_OUT  = os.path.join(PROC_DIR, "processed_test.csv")
DICT_OUT  = os.path.join(PROC_DIR, "data_dictionary.json")

os.makedirs(PROC_DIR, exist_ok=True)

# ---------------- Utilities ----------------
QMAP = {"1": (1, 3), "2": (4, 6), "3": (7, 9), "4": (10, 12)}

def parse_period_to_date(period_val: str) -> pd.Timestamp:
    """
    MLIT period often looks like: '2023年第3四半期'
    We map to the quarter midpoint month for a reasonable timestamp.
    """
    if pd.isna(period_val):
        return pd.NaT
    s = str(period_val)
    # try to extract yyyy and q
    # examples: '2023年第3四半期', '2024年第1四半期', or sometimes '20233'
    m = re.search(r'(?P<y>20\d{2}).*?(?P<q>[1-4])', s)
    if not m:
        # fallback if string like '20233'
        m = re.match(r'(?P<y>20\d{2})(?P<q>[1-4])', s)
    if not m:
        return pd.NaT
    y = int(m.group("y"))
    q = m.group("q")
    m_start, m_end = QMAP.get(q, (1, 3))
    # Use quarter midpoint month (second month of the quarter)
    midpoint_month = m_start + 1
    return pd.Timestamp(year=y, month=midpoint_month, day=15)

def coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def pick_first_existing(paths) -> Optional[str]:
    for p in paths:
        files = sorted(glob(p))
        if files:
            return files[0]
    return None

def normalize_mlit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a variety of MLIT CSV shapes to a unified raw schema
    Fields we try to pull:
      Prefecture, Municipality, DistrictName, TradePrice, Area, TotalFloorArea,
      BuildingYear, Structure, CoverageRatio, FloorAreaRatio, CityPlanning, Period, Type
    """
    # Column name map candidates
    col_map_candidates = {
        "prefecture": ["Prefecture", "都道府県名", "都道府県"],
        "municipality": ["Municipality", "市区町村名", "市区町村"],
        "district": ["DistrictName", "地区名", "町名"],
        "price": ["TradePrice", "取引価格(総額)", "取引価格"],
        "area": ["Area", "面積(㎡)", "土地面積(㎡)", "面積"],
        "tfa": ["TotalFloorArea", "延床面積(㎡)", "建物延床面積(㎡)"],
        "byear": ["BuildingYear", "建築年", "建築年（西暦）"],
        "structure": ["Structure", "建物の構造", "構造"],
        "cov": ["CoverageRatio", "建ぺい率(%)", "建ぺい率"],
        "far": ["FloorAreaRatio", "容積率(%)", "容積率"],
        "cityplan": ["CityPlanning", "都市計画"],
        "period": ["Period", "取引時期", "時期"],
        "type": ["Type", "種類", "取引の事情等"],  # 'Type' varies a lot
    }

    def pick(df, keys):
        for k in keys:
            if k in df.columns:
                return df[k]
        return pd.Series([None]*len(df))

    out = pd.DataFrame({
        "prefecture": pick(df, col_map_candidates["prefecture"]),
        "city_ward": pick(df, col_map_candidates["municipality"]),
        "district_name": pick(df, col_map_candidates["district"]),
        "sale_price_yen": coerce_num(pick(df, col_map_candidates["price"])),
        "area_m2": coerce_num(pick(df, col_map_candidates["area"])),
        "total_floor_area_m2": coerce_num(pick(df, col_map_candidates["tfa"])),
        "building_year": coerce_num(pick(df, col_map_candidates["byear"])),
        "building_structure": pick(df, col_map_candidates["structure"]),
        "coverage_ratio": coerce_num(pick(df, col_map_candidates["cov"])),
        "floor_area_ratio": coerce_num(pick(df, col_map_candidates["far"])),
        "city_planning": pick(df, col_map_candidates["cityplan"]),
        "period_raw": pick(df, col_map_candidates["period"]),
        "type_raw": pick(df, col_map_candidates["type"]),
    })
    return out

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    # transaction_date from period
    df["transaction_date"] = df["period_raw"].apply(parse_period_to_date)

    # building age
    df["building_age_years"] = np.where(
        df["building_year"].notna() & df["transaction_date"].notna(),
        df["transaction_date"].dt.year - df["building_year"].astype("Int64"),
        np.nan
    )
    df.loc[df["building_age_years"] < 0, "building_age_years"] = np.nan

    # property_type (best-effort from type_raw and areas)
    def infer_prop_type(row):
        t = str(row.get("type_raw") or "").lower()
        # very rough heuristics
        if "土地" in t or "land" in t:
            return "土地"
        if "マンション" in t or "condo" in t or "共同住宅" in t or (pd.notna(row["total_floor_area_m2"]) and row["total_floor_area_m2"] > 0 and (pd.isna(row["area_m2"]) or row["area_m2"] < 10)):
            return "マンション"
        if "戸建" in t or "一戸建" in t or "detached" in t:
            return "戸建て"
        # fallback by shapes
        if pd.notna(row["total_floor_area_m2"]) and row["total_floor_area_m2"] > 0:
            return "マンション"
        if pd.notna(row["area_m2"]) and row["area_m2"] > 0 and pd.isna(row["total_floor_area_m2"]):
            return "土地"
        return "戸建て"

    df["property_type"] = df.apply(infer_prop_type, axis=1)

    # unify structure labels a bit
    def map_structure(s):
        s = "" if pd.isna(s) else str(s)
        if any(k in s for k in ["RC", "鉄筋コンクリート"]):
            return "RC"
        if any(k in s for k in ["SRC", "鉄骨鉄筋コンクリート"]):
            return "SRC"
        if any(k in s for k in ["木", "木造", "W"]):
            return "木造"
        if any(k in s for k in ["鉄骨", "S"]):
            return "鉄骨"
        return s or None

    df["building_structure"] = df["building_structure"].apply(map_structure)

    # choose a size feature that best represents value driver
    df["effective_area_m2"] = np.where(
        df["property_type"] == "マンション",
        df["total_floor_area_m2"],
        df["area_m2"]
    )
    # guardrails
    df["effective_area_m2"] = df["effective_area_m2"].clip(lower=10)

    # engineered buckets
    df["age_bucket"] = pd.cut(
        df["building_age_years"],
        bins=[-1,5,10,20,999],
        labels=["0-5","5-10","10-20","20+"]
    )
    df["is_tokyo"] = (df["prefecture"].astype(str).str.contains("東京")).astype(int)

    # price per sqm
    df["price_per_sqm"] = df["sale_price_yen"] / df["effective_area_m2"]
    return df

def temporal_split(df: pd.DataFrame, months_test: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    # Drop rows without transaction_date
    df = df[df["transaction_date"].notna()].copy()
    max_date = df["transaction_date"].max()
    cutoff = max_date - pd.DateOffset(months=months_test)
    train = df[df["transaction_date"] <= cutoff].copy()
    test  = df[df["transaction_date"] > cutoff].copy()
    return train, test, cutoff.date().isoformat()

# ---------------- Main flow ----------------
def main():
    # 1) Locate a raw MLIT CSV
    mlit_normalized = pick_first_existing([os.path.join(RAW_DIR, "mlit_trades_normalized_*.csv")])
    mlit_any = pick_first_existing([
        os.path.join(RAW_DIR, "mlit_trades_*.csv"),
        os.path.join(RAW_DIR, "mlit_trades_manual_*.csv"),
        os.path.join(RAW_DIR, "*.csv"),
    ])

    if mlit_normalized:
        print(f"[INFO] Using normalized MLIT CSV: {mlit_normalized}")
        df_raw = pd.read_csv(mlit_normalized)
        # Ensure expected field names
        expected_cols = {"Prefecture","Municipality","DistrictName","TradePrice","Area","TotalFloorArea","BuildingYear","Structure","CoverageRatio","FloorAreaRatio","CityPlanning","Period","Type"}
        if not expected_cols.issubset(set(df_raw.columns)):
            # If our earlier fetch script produced lower-case names, handle that
            df_raw.columns = [c.strip() for c in df_raw.columns]
    elif mlit_any:
        print(f"[INFO] Using MLIT-like CSV: {mlit_any}")
        df_raw = pd.read_csv(mlit_any)
    else:
        raise SystemExit(
            "[ERROR] No MLIT CSV found in data/raw/.\n"
            "Fetch first with:\n"
            "  python src/fetch_mlit.py 東京 20231 20244\n"
            "or save a CSV manually into data/raw/."
        )

    # 2) Normalize to our internal raw schema
    df = normalize_mlit_columns(df_raw)

    # 3) Minimal filtering & cleaning
    # Keep obvious valid rows
    before = len(df)
    df = df[(df["sale_price_yen"].notna()) & (df["sale_price_yen"] > 0)]
    df = df[df["prefecture"].notna() & df["city_ward"].notna()]
    after = len(df)
    print(f"[CLEAN] Rows kept: {after:,} / {before:,}")

    # 4) Derive features
    df = derive_features(df)

    # 5) More guards
    df = df[df["effective_area_m2"].notna() & (df["effective_area_m2"] > 0)]
    df = df[df["price_per_sqm"].notna() & (df["price_per_sqm"] > 10000)]  # discard noise
    # ensure transaction_date exists
    df = df[df["transaction_date"].notna()]

    # 6) Split
    train, test, cutoff = temporal_split(df, months_test=6)

    # 7) Save
    train.to_csv(TRAIN_OUT, index=False, encoding="utf-8")
    test.to_csv(TEST_OUT, index=False, encoding="utf-8")

    # 8) Data dictionary
    dictionary = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "source": "MLIT transactions (normalized)",
        "cutoff_date_for_test": cutoff,
        "rows": {"train": len(train), "test": len(test)},
        "columns": {col: str(dtype) for col, dtype in zip(train.columns, train.dtypes)},
        "inference_policy": "Exclude sale_price_yen, transaction_date, and future-dated fields at inference."
    }
    with open(DICT_OUT, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Processed → {PROC_DIR}")
    print(f"     Train rows: {len(train):,}  Test rows: {len(test):,}  Cutoff: {cutoff}")
    for col in ["sale_price_yen","effective_area_m2","building_age_years","price_per_sqm"]:
        s = train[col].describe()
        p95 = train[col].quantile(0.95)
        print(f"     {col:22s} min {s['min']:.0f}  p50 {s['50%']:.0f}  p95 {p95:.0f}  max {s['max']:.0f}")

if __name__ == "__main__":
    main()
