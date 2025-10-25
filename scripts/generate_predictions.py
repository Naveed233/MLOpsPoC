#!/usr/bin/env python3
"""
Generate synthetic prediction logs for dashboard visualization
"""
import requests
import random
import time
from datetime import datetime, timedelta
import json

# Configuration
API_URL = "http://localhost:8000/predict_price"
NUM_PREDICTIONS = 100

# Real Tokyo/Osaka wards
LOCATIONS = [
    {"prefecture": "東京都", "city_ward": "渋谷区"},
    {"prefecture": "東京都", "city_ward": "新宿区"},
    {"prefecture": "東京都", "city_ward": "港区"},
    {"prefecture": "東京都", "city_ward": "千代田区"},
    {"prefecture": "東京都", "city_ward": "中央区"},
    {"prefecture": "東京都", "city_ward": "世田谷区"},
    {"prefecture": "東京都", "city_ward": "目黒区"},
    {"prefecture": "大阪府", "city_ward": "中央区"},
    {"prefecture": "大阪府", "city_ward": "北区"},
    {"prefecture": "神奈川県", "city_ward": "横浜市西区"},
    {"prefecture": "神奈川県", "city_ward": "横浜市中区"},
    {"prefecture": "千葉県", "city_ward": "千葉市中央区"},
]

PROPERTY_TYPES = ["マンション", "戸建て", "土地"]
STRUCTURES = ["RC", "SRC", "木造", "鉄骨造"]

def generate_realistic_property():
    """Generate realistic property parameters"""
    location = random.choice(LOCATIONS)
    prop_type = random.choice(PROPERTY_TYPES)
    
    # Realistic ranges based on property type
    if prop_type == "マンション":
        area = random.uniform(30, 120)  # 30-120 m²
        age = random.uniform(0, 50)     # 0-50 years
        coverage = random.uniform(40, 80)
        floor_ratio = random.uniform(200, 600)
        structure = random.choice(["RC", "SRC"])
    elif prop_type == "戸建て":
        area = random.uniform(60, 200)  # 60-200 m²
        age = random.uniform(0, 40)
        coverage = random.uniform(30, 70)
        floor_ratio = random.uniform(100, 300)
        structure = random.choice(["木造", "鉄骨造", "RC"])
    else:  # 土地
        area = random.uniform(50, 300)
        age = 0
        coverage = random.uniform(40, 80)
        floor_ratio = random.uniform(100, 400)
        structure = "その他"
    
    return {
        **location,
        "property_type": prop_type,
        "building_structure": structure,
        "building_age_years": round(age, 1),
        "effective_area_m2": round(area, 1),
        "coverage_ratio": round(coverage, 1),
        "floor_area_ratio": round(floor_ratio, 1),
    }

def main():
    print("=" * 70)
    print("🏠 GENERATING SYNTHETIC PREDICTIONS")
    print("=" * 70)
    
    successful = 0
    failed = 0
    
    for i in range(NUM_PREDICTIONS):
        property_data = generate_realistic_property()
        
        try:
            response = requests.post(API_URL, json=property_data, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                successful += 1
                
                if (i + 1) % 10 == 0:
                    print(f"✅ Generated {i + 1}/{NUM_PREDICTIONS} predictions...")
                    print(f"   Latest: {property_data['city_ward']} - ¥{result['prediction_yen']/1e6:.1f}M")
            else:
                failed += 1
                print(f"❌ Failed: {response.status_code}")
        
        except requests.exceptions.ConnectionError:
            print("\n❌ Error: API server not running!")
            print("Start it with: uvicorn src.api_server:app --reload")
            break
        except Exception as e:
            failed += 1
            print(f"❌ Error: {e}")
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed:     {failed}")
    print(f"\n💾 Predictions saved to: data/predictions/predictions.jsonl")
    print(f"🔄 Refresh your dashboard to see updated graphs!")

if __name__ == "__main__":
    main()

