#!/usr/bin/env python3
"""
Fetch REAL MLIT data using the direct API endpoint
Run this when you have network access to www.land.mlit.go.jp
"""
import requests
import pandas as pd
from datetime import datetime

# Major Tokyo/Osaka city codes
# Format: 13XXX (Tokyo), 27XXX (Osaka), 14XXX (Kanagawa)
CITY_CODES = {
    "13101": "千代田区",
    "13102": "中央区", 
    "13103": "港区",
    "13104": "新宿区",
    "13105": "文京区",
    "13113": "渋谷区",
    "27128": "大阪市中央区",
    "27102": "大阪市北区",
    "14100": "横浜市",
}

def fetch_mlit_data(city_code, from_quarter, to_quarter):
    """
    Fetch MLIT real estate transaction data
    
    Args:
        city_code: 5-digit city code (e.g., "13101" for Chiyoda-ku)
        from_quarter: Start quarter (e.g., "20241" = Q1 2024)
        to_quarter: End quarter (e.g., "20242" = Q2 2024)
    
    Returns:
        DataFrame of transactions
    """
    url = f"https://www.land.mlit.go.jp/webland/api/TradeListSearch?from={from_quarter}&to={to_quarter}&city={city_code}"
    
    print(f"Fetching {CITY_CODES.get(city_code, city_code)}...")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if "data" in data and len(data["data"]) > 0:
            df = pd.DataFrame(data["data"])
            print(f"  ✅ Fetched {len(df)} transactions")
            return df
        else:
            print(f"  ⚠️  No data available")
            return pd.DataFrame()
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return pd.DataFrame()

def main():
    print("=" * 70)
    print("🌐 FETCHING REAL MLIT DATA")
    print("=" * 70)
    
    # Fetch last 2 quarters (Q1-Q2 2024)
    all_data = []
    
    for city_code, city_name in CITY_CODES.items():
        df = fetch_mlit_data(
            city_code=city_code,
            from_quarter="20241",  # Q1 2024
            to_quarter="20242"     # Q2 2024
        )
        
        if not df.empty:
            df["city_code"] = city_code
            df["city_name"] = city_name
            all_data.append(df)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        
        print("\n" + "=" * 70)
        print("📊 RESULTS")
        print("=" * 70)
        print(f"Total transactions: {len(combined):,}")
        print(f"Cities covered: {len(CITY_CODES)}")
        print(f"\nColumns available: {len(combined.columns)}")
        print(combined.columns.tolist())
        
        # Save to CSV
        output_file = f"data/raw/mlit_real_{datetime.now().strftime('%Y%m%d')}.csv"
        combined.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n💾 Saved to: {output_file}")
        print(f"\n🔍 Sample data:")
        print(combined.head(3).to_string())
        
        print("\n" + "=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print("1. Map MLIT columns to your schema in src/data_prep.py")
        print("2. Run: python src/data_prep.py")
        print("3. Run: python src/model_training.py")
        print("4. Your dashboard will now show REAL data!")
    else:
        print("\n❌ No data fetched. Check network connection.")

if __name__ == "__main__":
    main()

