# 📊 Grafana + Prometheus Monitoring Setup

## 🎯 What You Get

Production-grade monitoring stack with:
- **Grafana**: Professional dashboards (used by Netflix, Uber, PayPal)
- **Prometheus**: Time-series metrics collection
- **Node Exporter**: System metrics (CPU, memory, disk)
- **Custom ML Metrics**: Predictions, latency, prices, drift

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd /Users/naveedmaqbool/Desktop/real-estate-mlops
source .venv/bin/activate
pip install prometheus-client==0.19.0 psutil==5.9.6
```

### 2. Start the Monitoring Stack
```bash
docker-compose up -d
```

This starts 4 services:
- **API** (port 8000) - ML inference with metrics
- **Prometheus** (port 9090) - Metrics collection
- **Grafana** (port 3000) - Dashboards
- **Node Exporter** (port 9100) - System metrics

### 3. Access Dashboards

**Grafana Dashboard:**
- URL: http://localhost:3000
- Username: `admin`
- Password: `admin123`
- Dashboard: "🏠 Real Estate ML Monitoring" (auto-loaded)

**Prometheus:**
- URL: http://localhost:9090
- Query metrics directly

**ML API:**
- URL: http://localhost:8000/docs
- Metrics endpoint: http://localhost:8000/metrics

---

## 📈 Metrics Exposed

### ML-Specific Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `prediction_requests_total` | Counter | Total predictions by status, prefecture, property type |
| `prediction_latency_seconds` | Histogram | Prediction latency distribution |
| `prediction_price_yen` | Histogram | Predicted price distribution |
| `model_load_time_seconds` | Gauge | Time to load model |
| `model_info` | Gauge | Model metadata (tag, saved_at) |

### System Metrics (Node Exporter)

| Metric | Description |
|--------|-------------|
| `node_cpu_seconds_total` | CPU usage by mode |
| `node_memory_*` | Memory usage stats |
| `node_disk_*` | Disk I/O and usage |
| `node_network_*` | Network traffic |

---

## 🎨 Grafana Dashboard Panels

### Top Row - KPIs
1. **Predictions/Min**: Current request rate
2. **Latency p50**: Median response time
3. **Median Predicted Price**: Typical prediction
4. **Success Rate**: % of successful predictions

### Middle Row - Time Series
5. **Prediction Request Rate**: Requests/sec over time (by status)
6. **Prediction Latency**: p50, p95, p99 over time
7. **Predictions by Prefecture**: Geographic distribution
8. **Predicted Price Distribution**: Price percentiles

### Bottom Row - System Health
9. **System CPU Usage**: Server CPU utilization
10. **System Memory**: Used vs available memory
11. **Model Load Time**: Time to load model on startup

---

## 🔍 Example Queries

### In Prometheus (http://localhost:9090)

**Total predictions in last hour:**
```promql
sum(increase(prediction_requests_total{status="success"}[1h]))
```

**95th percentile latency:**
```promql
histogram_quantile(0.95, sum(rate(prediction_latency_seconds_bucket[5m])) by (le))
```

**Predictions by prefecture:**
```promql
sum by(prefecture) (rate(prediction_requests_total{status="success"}[5m]))
```

**Average predicted price:**
```promql
rate(prediction_price_yen_sum[5m]) / rate(prediction_price_yen_count[5m])
```

**Error rate:**
```promql
sum(rate(prediction_requests_total{status="error"}[5m])) / sum(rate(prediction_requests_total[5m]))
```

---

## 🎤 Interview Demo Flow

### 1. Start with Grafana Overview (30 sec)
> "This is our production monitoring stack using Grafana and Prometheus—the same tools used by companies like Uber and Shopify. It provides real-time visibility into ML model performance and system health."

### 2. Show KPI Panels (15 sec)
> "At the top, we track key metrics: prediction rate, latency, median price, and success rate. These update in real-time as requests come in."

### 3. Highlight Latency Tracking (20 sec)
> "This panel shows p50, p95, and p99 latency over time. We maintain sub-10ms p95 latency, which is critical for real-time user experiences. If latency degrades, we can see it immediately and investigate."

### 4. Geographic Analysis (15 sec)
> "We track predictions by prefecture to understand regional demand patterns. Tokyo consistently has the highest volume, as expected for Japan's largest real estate market."

### 5. System Health (15 sec)
> "The bottom row monitors infrastructure health: CPU, memory, and model load time. This helps us right-size our compute resources and catch issues before they impact users."

### 6. Custom Metrics (20 sec)
> "These metrics are custom-instrumented in our FastAPI service using Prometheus client libraries. Every prediction increments counters and records histograms, giving us complete observability."

### 7. Alerting (10 sec) [Show Prometheus alerts]
> "We can configure alerts in Prometheus—for example, if p95 latency exceeds 50ms or error rate goes above 1%, it sends notifications to Slack or PagerDuty."

---

## 🛠️ Configuration Files

```
real-estate-mlops/
├── docker-compose.yml                    # Orchestrates all services
├── infra/
│   ├── prometheus.yml                    # Prometheus config
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/
│       │   │   └── prometheus.yml        # Auto-configure Prometheus
│       │   └── dashboards/
│       │       └── default.yml           # Auto-load dashboards
│       └── dashboards/
│           └── ml_monitoring.json        # ML dashboard definition
└── src/
    └── api_server_grafana.py            # Instrumented API
```

---

## 🚀 Advanced: Custom Alerts

Create `infra/prometheus_alerts.yml`:

```yaml
groups:
  - name: ml_alerts
    interval: 30s
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m])) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency detected"
          description: "p95 latency is {{ $value }}s (threshold: 50ms)"
      
      - alert: HighErrorRate
        expr: sum(rate(prediction_requests_total{status="error"}[5m])) / sum(rate(prediction_requests_total[5m])) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }}% (threshold: 1%)"
      
      - alert: LowPredictionVolume
        expr: sum(rate(prediction_requests_total[5m])) < 0.1
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "Low prediction volume"
          description: "Only {{ $value }} predictions/sec in last 10 minutes"
```

Then update `docker-compose.yml`:
```yaml
prometheus:
  ...
  volumes:
    - ./infra/prometheus_alerts.yml:/etc/prometheus/alerts.yml
  command:
    - '--config.file=/etc/prometheus/prometheus.yml'
    - '--alerting.rules=/etc/prometheus/alerts.yml'
```

---

## 📊 Generate Load for Demo

Make 100 predictions to populate Grafana:

```bash
# In terminal 1: Start the stack
docker-compose up

# In terminal 2: Generate load
cd /Users/naveedmaqbool/Desktop/real-estate-mlops
source .venv/bin/activate
python scripts/generate_predictions.py
```

Or use Apache Bench:
```bash
ab -n 1000 -c 10 -p request.json -T application/json http://localhost:8000/predict_price
```

---

## 🔧 Troubleshooting

### Grafana shows "No data"
1. Check Prometheus is scraping: http://localhost:9090/targets
2. Verify API is exposing metrics: http://localhost:8000/metrics
3. Check API logs: `docker-compose logs api`

### Prometheus can't reach API
- Ensure all services are in the same Docker network
- Check `prometheus.yml` has correct target: `api:8000`

### Dashboard not auto-loading
- Check provisioning config: `infra/grafana/provisioning/dashboards/default.yml`
- Manually import: Grafana → Dashboards → Import → Upload `ml_monitoring.json`

---

## 🎯 Key Differences: Streamlit vs Grafana

| Feature | Streamlit Dashboard | Grafana |
|---------|-------------------|---------|
| **Use Case** | Data science exploration | Production monitoring |
| **Real-time** | Refresh on demand | Live updates (15s) |
| **Metrics** | Batch logs (JSONL) | Time-series (Prometheus) |
| **Alerting** | ❌ None | ✅ Email, Slack, PagerDuty |
| **Scalability** | Single instance | Multi-tenant, HA |
| **Industry Use** | Prototypes, demos | Production systems |

**Both are valuable:**
- Use **Streamlit** for exploratory analysis, model debugging, stakeholder demos
- Use **Grafana** for production monitoring, SRE dashboards, alerting

---

## 🌟 Why Grafana Impresses Interviewers

1. **Industry Standard**: Shows you know production tools, not just Jupyter
2. **Observability**: Demonstrates SRE mindset (monitoring, alerting, SLOs)
3. **Scalability**: Grafana handles millions of metrics/sec
4. **Custom Metrics**: Shows you can instrument code (not just use logs)
5. **Time-Series**: Real-time monitoring > batch reporting

**Saying:** "We use Grafana + Prometheus" instantly signals production experience.

---

## 📚 Learn More

- **Grafana Docs**: https://grafana.com/docs/
- **Prometheus Docs**: https://prometheus.io/docs/
- **PromQL Tutorial**: https://prometheus.io/docs/prometheus/latest/querying/basics/
- **Grafana Best Practices**: https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/best-practices/

---

## 🎉 You Now Have

✅ Production-grade monitoring stack  
✅ Real-time ML metrics (latency, volume, prices)  
✅ System metrics (CPU, memory, disk)  
✅ Professional Grafana dashboard  
✅ Prometheus alerting rules  
✅ Docker-based deployment  
✅ **Instant credibility in ML/SRE interviews!**

---

**Next:** Generate some predictions and watch Grafana come alive! 🚀

