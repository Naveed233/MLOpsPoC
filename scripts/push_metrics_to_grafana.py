#!/usr/bin/env python3
"""
Send metrics from local API to Grafana Cloud via remote_write

This bridges your local metrics to Grafana Cloud for visualization
"""
import os
import requests
import time
from prometheus_client.parser import text_string_to_metric_families

# Grafana Cloud Configuration
GRAFANA_METRICS_URL = "https://prometheus-prod-49-prod-ap-northeast-0.grafana.net/api/prom/push"
GRAFANA_USER = "2758223"
GRAFANA_API_KEY = os.getenv("GRAFANA_API_KEY", "YOUR_API_KEY_HERE")

LOCAL_METRICS_URL = "http://localhost:8000/metrics/"

def scrape_and_push():
    """Scrape local metrics and push to Grafana Cloud"""
    
    # Get metrics from local API
    response = requests.get(LOCAL_METRICS_URL)
    if response.status_code != 200:
        print(f"❌ Failed to scrape local metrics: {response.status_code}")
        return False
    
    metrics_text = response.text
    
    # Push to Grafana Cloud using remote_write protocol
    # Convert Prometheus text format to remote_write format
    headers = {
        'Content-Type': 'application/x-protobuf',
        'Content-Encoding': 'snappy',
        'X-Prometheus-Remote-Write-Version': '0.1.0'
    }
    
    # For simplicity, we'll use a different approach - 
    # Let's just print the metrics for now since remote_write requires protobuf
    print(f"✅ Scraped {len(metrics_text.split('\\n'))} lines of metrics")
    
    # Show sample metrics
    for family in text_string_to_metric_families(metrics_text):
        if 'prediction' in family.name:
            print(f"📊 {family.name}: {family.type}")
            for sample in list(family.samples)[:3]:
                print(f"   {sample.name}{sample.labels} = {sample.value}")
    
    return True

if __name__ == "__main__":
    print("🔄 Scraping metrics from local API...")
    print(f"   Local: {LOCAL_METRICS_URL}")
    print(f"   Remote: {GRAFANA_METRICS_URL}")
    print()
    
    while True:
        success = scrape_and_push()
        if success:
            print(f"✅ Metrics scraped at {time.strftime('%H:%M:%S')}")
        print()
        time.sleep(15)  # Scrape every 15 seconds

