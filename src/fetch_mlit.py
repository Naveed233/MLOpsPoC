#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_mlit.py
Fetch official MLIT real estate transaction data using j_realty_api

Usage:
  python src/fetch_mlit.py <prefecture> <from_period> <to_period>

Example:
  python src/fetch_mlit.py 東京 20231 20244
  python src/fetch_mlit.py 千葉 20231 20244

Period format: YYYYQ (e.g., 20231 = 2023 Q1, 20244 = 2024 Q4)
"""

import sys
import os
import pandas as pd
from j_realty_api.j_realty_jp import PropTransactions, CityCode

# Prefecture code mapping
PREF_CODES = {
    "北海道": "01", "青森": "02", "岩手": "03", "宮城": "04", "秋田": "05",
    "山形": "06", "福島": "07", "茨城": "08", "栃木": "09", "群馬": "10",
    "埼玉": "11", "千葉": "12", "東京": "13", "神奈川": "14", "新潟": "15",
    "富山": "16", "石川": "17", "福井": "18", "山梨": "19", "長野": "20",
    "岐阜": "21", "静岡": "22", "愛知": "23", "三重": "24", "滋賀": "25",
    "京都": "26", "大阪": "27", "兵庫": "28", "奈良": "29", "和歌山": "30",
    "鳥取": "31", "島根": "32", "岡山": "33", "広島": "34", "山口": "35",
    "徳島": "36", "香川": "37", "愛媛": "38", "高知": "39", "福岡": "40",
    "佐賀": "41", "長崎": "42", "熊本": "43", "大分": "44", "宮崎": "45",
    "鹿児島": "46", "沖縄": "47"
}

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)


def fetch_data(prefecture: str, from_period: int, to_period: int):
    """Fetch MLIT data for specified prefecture and period range"""
    
    # Get prefecture code
    pref_clean = prefecture.replace("都", "").replace("府", "").replace("県", "")
    pref_code = PREF_CODES.get(pref_clean)
    
    if not pref_code:
        print(f"[ERROR] Unknown prefecture: {prefecture}")
        print(f"Available: {', '.join(PREF_CODES.keys())}")
        sys.exit(1)
    
    print(f"[INFO] Fetching data for {prefecture} (code: {pref_code})")
    print(f"[INFO] Period range: {from_period} to {to_period}")
    
    # Get all city codes for this prefecture
    print(f"[FETCH] Getting city codes for {prefecture}...")
    try:
        city_lookup = CityCode(prefecture)
        cities_json = city_lookup.city_json
        
        if not cities_json or 'data' not in cities_json:
            print("[ERROR] Could not retrieve city codes")
            sys.exit(1)
        
        cities = cities_json['data']
        print(f"[OK] Found {len(cities)} cities/wards in {prefecture}")
    except Exception as e:
        print(f"[ERROR] Failed to get city codes: {e}")
        sys.exit(1)
    
    # Fetch data for each city
    all_data = []
    for i, city_info in enumerate(cities, 1):
        city_code = city_info.get('id', '')
        city_name = city_info.get('name', 'Unknown')
        
        print(f"[{i}/{len(cities)}] {city_name} (code: {city_code})...", end=" ", flush=True)
        
        try:
            # Fetch transactions for this city
            prop_trans = PropTransactions(
                pref_code=pref_code,
                city_code=city_code,
                from_dt=from_period,
                to_dt=to_period
            )
            
            df = prop_trans.get_data()
            
            if df is not None and len(df) > 0:
                all_data.append(df)
                print(f"✓ {len(df):,} rows")
            else:
                print("✗ no data")
        except Exception as e:
            print(f"✗ error: {e}")
    
    if not all_data:
        print("[ERROR] No data fetched")
        sys.exit(1)
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n[OK] Total rows fetched: {len(combined):,}")
    
    # Save raw data
    raw_path = os.path.join(RAW_DIR, f"mlit_trades_{pref_code}_{from_period}_{to_period}.csv")
    combined.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] Raw data → {raw_path}")
    
    # Normalize column names for easier processing
    normalized = combined.copy()
    
    # Common column name mappings
    rename_map = {
        "都道府県名": "Prefecture",
        "市区町村名": "Municipality", 
        "地区名": "DistrictName",
        "最寄駅：名称": "NearestStation",
        "最寄駅：距離（分）": "StationDistanceMin",
        "取引価格（総額）": "TradePrice",
        "坪単価": "PricePerTsubo",
        "間取り": "Layout",
        "面積（㎡）": "Area",
        "土地の形状": "LandShape",
        "間口": "Frontage",
        "延床面積（㎡）": "TotalFloorArea",
        "建築年": "BuildingYear",
        "建物の構造": "Structure",
        "用途": "Usage",
        "今後の利用目的": "FutureUsage",
        "前面道路：方位": "RoadDirection",
        "前面道路：種類": "RoadType",
        "前面道路：幅員（m）": "RoadWidth",
        "都市計画": "CityPlanning",
        "建ぺい率（％）": "CoverageRatio",
        "容積率（％）": "FloorAreaRatio",
        "取引時期": "Period",
        "改装": "Renovation",
        "取引の事情等": "Type",
    }
    
    normalized.rename(columns=rename_map, inplace=True)
    
    norm_path = os.path.join(RAW_DIR, f"mlit_trades_normalized_{pref_code}_{from_period}_{to_period}.csv")
    normalized.to_csv(norm_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] Normalized → {norm_path}")
    
    # Quick stats
    if "TradePrice" in normalized.columns:
        prices = pd.to_numeric(normalized["TradePrice"], errors="coerce")
        valid_prices = prices.dropna()
        if len(valid_prices) > 0:
            print(f"\n[STATS] Price range: ¥{valid_prices.min():,.0f} - ¥{valid_prices.max():,.0f}")
            print(f"        Median: ¥{valid_prices.median():,.0f}")
            print(f"        Valid price records: {len(valid_prices):,} / {len(normalized):,}")
    
    print("\n[DONE] Ready to run: python src/data_prep.py")


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    
    prefecture = sys.argv[1]
    from_period = int(sys.argv[2])
    to_period = int(sys.argv[3])
    
    fetch_data(prefecture, from_period, to_period)


if __name__ == "__main__":
    main()
