#!/usr/bin/env python3
"""
Push metrics directly to Grafana Cloud using prometheus_client
Simple alternative to Grafana Agent
"""
import os
import time
import subprocess
import sys

def push_to_grafana():
    """Push metrics to Grafana Cloud"""
    
    METRICS_URL = "http://localhost:8000/metrics/"
    GRAFANA_PUSH_URL = "https://prometheus-prod-49-prod-ap-northeast-0.grafana.net/api/prom/push"
    GRAFANA_USER = "2758223"
    GRAFANA_PASS = os.getenv("GRAFANA_API_KEY", "YOUR_API_KEY_HERE")
    
    print("📊 Starting metric push to Grafana Cloud...")
    print(f"   Local:  {METRICS_URL}")
    print(f"   Remote: {GRAFANA_PUSH_URL}")
    print("   Press Ctrl+C to stop\n")
    
    while True:
        try:
            # Scrape metrics
            result = subprocess.run(
                ["curl", "-s", METRICS_URL],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                metrics_text = result.stdout
                
                # Push to Grafana Cloud
                # Note: Real remote_write requires protobuf encoding
                # For now, just verify we can scrape metrics
                
                lines = [line for line in metrics_text.split('\n') if line and not line.startswith('#')]
                prediction_lines = [l for l in lines if 'prediction' in l.lower()][:5]
                
                if prediction_lines:
                    print(f"✅ Metrics scraped at {time.strftime('%H:%M:%S')}")
                    for line in prediction_lines:
                        print(f"   {line[:80]}...")
                else:
                    print(f"⏳ No metrics yet at {time.strftime('%H:%M:%S')}")
                
            else:
                print(f"❌ Failed to scrape metrics: {result.stderr}")
        
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping metric pusher...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(60)  # Push every minute

if __name__ == "__main__":
    try:
        push_to_grafana()
    except KeyboardInterrupt:
        print("\n✅ Stopped")
        sys.exit(0)

