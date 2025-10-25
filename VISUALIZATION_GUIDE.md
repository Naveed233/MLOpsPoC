# 📊 Visualization & Analysis Guide

## Quick Start

### 1. Generate Static HTML Report
Creates a single, shareable HTML file with all charts and metrics.

```bash
python src/report_generate.py
# Open: reports/price_engine_report.html
```

**Includes:**
- Model MAE timeline
- Prediction volume & latency charts
- Price distribution histograms
- Drift metrics visualization
- Production model metrics

### 2. Launch Interactive Dashboard
Real-time monitoring dashboard with filters and drill-downs.

```bash
streamlit run src/dashboard.py
# Opens in browser at http://localhost:8501
```

**Features:**
- Date range filtering
- Ward/city filtering
- Property type filtering
- Real-time metrics
- Interactive charts
- MAE by ward analysis (when actuals available)

## Verification & Actuals Workflow

### Adding Actual Sale Prices

1. **Capture external_id in predictions:**
   ```json
   {
     "external_id": "listing-12345",
     "prefecture": "東京都",
     "city_ward": "渋谷区",
     ...
   }
   ```

2. **Record outcomes in `data/actuals/actuals.csv`:**
   ```csv
   external_id,sale_price_yen,ts_utc,city_ward
   listing-12345,78500000,2025-10-24T12:00:00Z,渋谷区
   listing-12346,71200000,2025-10-24T12:30:00Z,中央区
   ```

3. **Run MAE analysis:**
   ```bash
   python src/monitor_drift.py --recent-window-days 7 --with-actuals
   ```

4. **View in dashboard:**
   - MAE by ward visualization
   - Best/worst performing regions
   - Overall accuracy metrics

### Best Practices for Actuals

✅ **DO:**
- Always include `external_id` in API requests
- Update actuals weekly as sales close
- Segment MAE analysis by ward and property type
- Set alerts for wards with high MAE (>20% of median price)
- Feed outcomes back into training data

❌ **DON'T:**
- Mix property types when calculating MAE
- Compare predictions >6 months old to actuals
- Ignore regional variations in error rates

## Monitoring Schedule

### Daily
```bash
# Quick health check
curl http://localhost:8000/health
curl http://localhost:8000/version

# Check latest metrics
tail -10 data/predictions/predictions.jsonl
```

### Weekly
```bash
# Drift monitoring
python src/monitor_drift.py --recent-window-days 7

# Generate report
python src/report_generate.py

# Review in dashboard
streamlit run src/dashboard.py
```

### Monthly
```bash
# Comprehensive analysis with actuals
python src/monitor_drift.py --recent-window-days 30 --with-actuals

# Consider retraining if:
# - Drift alerts > 3 features
# - MAE increased >15% in any major ward
# - Latency p95 > 100ms
```

## Dashboard Sections

### 1. Model Performance
- **MAE Timeline**: Track model quality over training runs
- **Current Production**: Active model metrics
- Identify: Model degradation, successful improvements

### 2. Prediction Analysis
- **Daily Volume**: Request traffic patterns
- **Latency Distribution**: API performance
- **Price Distribution**: Value range coverage
- **By Ward**: Geographic distribution
- Identify: Usage patterns, performance issues

### 3. Drift Monitoring
- **Drift by Feature**: PSI/JSD metrics per feature
- **Status Breakdown**: OK/WARN/ALERT counts
- **Details Table**: Feature-level analysis
- Identify: Data shift, model decay

### 4. MAE by Ward
- **Ward Comparison**: Regional accuracy
- **Best/Worst**: Performance extremes
- **Overall Metrics**: System-wide accuracy
- Identify: Regional bias, data quality issues

## CloudWatch Integration (Production)

Export metrics to CloudWatch for ops monitoring:

```python
import boto3
import json

cloudwatch = boto3.client('cloudwatch', region_name='ap-northeast-1')

# Read latest monitoring report
with open('reports/summary_*.json') as f:
    metrics = json.load(f)

# Push to CloudWatch
cloudwatch.put_metric_data(
    Namespace='RealEstateML/Production',
    MetricData=[
        {
            'MetricName': 'API_Latency_P95',
            'Value': metrics['latency']['p95_ms'],
            'Unit': 'Milliseconds'
        },
        {
            'MetricName': 'Drift_NumericAlerts',
            'Value': metrics['drift_summary']['numeric_alerts'],
            'Unit': 'Count'
        },
        {
            'MetricName': 'Drift_CategoricalAlerts',
            'Value': metrics['drift_summary']['categorical_alerts'],
            'Unit': 'Count'
        },
        {
            'MetricName': 'MAE_Overall',
            'Value': metrics.get('mae_overall', {}).get('MAE', 0),
            'Unit': 'None'
        },
    ]
)
```

## Grafana Dashboard (Production)

Create dashboards for:
1. **SLA Metrics**: Latency p50/p95/p99, error rate, availability
2. **Model Metrics**: Drift alerts, MAE by region, prediction volume
3. **Business Metrics**: Avg predicted price, prediction distribution, regional coverage

## Alerting Thresholds

| Metric | WARN | ALERT | Action |
|--------|------|-------|--------|
| Latency p95 | >50ms | >100ms | Scale up, optimize |
| Drift (PSI) | >0.1 | >0.25 | Review data, consider retrain |
| Drift (JSD) | >0.15 | >0.3 | Review data, consider retrain |
| MAE increase | >10% | >20% | Investigate, retrain |
| Error rate | >1% | >5% | Check logs, rollback if needed |

## Report Sharing

### Static Report
```bash
# Generate report
python src/report_generate.py

# Share via email/Slack
cp reports/price_engine_report.html /path/to/shared/folder/
```

### Dashboard Screenshots
```bash
# In Streamlit, use built-in screenshot or:
streamlit run src/dashboard.py --server.headless true --server.port 8501

# Then use headless browser for automated screenshots
```

### API for Metrics
Expose monitoring metrics via API endpoint:

```python
@app.get("/metrics/summary")
def get_metrics_summary():
    summary = load_latest_summary()
    return {
        "latency_p95": summary['latency']['p95_ms'],
        "drift_alerts": sum(summary['drift_summary'].values()),
        "mae_overall": summary.get('mae_overall', {}).get('MAE'),
    }
```

## Troubleshooting

### No charts in HTML report
- Ensure matplotlib is installed: `pip install matplotlib`
- Check that models/history/ has metadata.json files
- Verify predictions.jsonl exists and has data

### Dashboard not loading
- Check Streamlit installation: `pip install streamlit`
- Ensure port 8501 is available
- Clear cache: `streamlit cache clear`

### MAE not showing
- Verify actuals.csv has `external_id` and `sale_price_yen` columns
- Check that predictions have `external_id` field
- Run drift monitor with `--with-actuals` flag

### Drift always shows 0
- Need baseline first: `python src/monitor_drift.py --create-baseline`
- Ensure recent window has different data from baseline
- Check that features exist in predictions.jsonl

