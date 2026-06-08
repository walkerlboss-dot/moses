# Monitoring API — Moses

> **Real-time production monitoring for deployed humanoid robots.**

---

## ProductionMonitor

```python
from moses.monitoring.production import ProductionMonitor
```

Real-time metrics, health checks, anomaly detection, alerting.

### Constructor

```python
ProductionMonitor(
    config: MonitorConfig,
    alert_channels: list[AlertChannel],
)
```

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `ingest()` | `metrics_dict` | — | Ingest metrics from robot |
| `get_health()` | — | `HealthStatus` | Overall health score (0-100) |
| `get_metrics()` | `name`, `window` | `list` | Metrics time series |
| `check_anomaly()` | — | `list` | Detected anomalies |
| `alert()` | `severity`, `message` | — | Send alert |

### Health Checks

| Check | Threshold | Action |
|-------|-----------|--------|
| Joint limits | >95% of limit | Warning |
| Motor temp | >80°C | Critical |
| Battery | <20% | Warning |
| Sensor sanity | NaN detected | Critical |
| Control freq | <90 Hz | Warning |

### Anomaly Detection (3 layers)

1. **Rule-based:** Hard thresholds
2. **Statistical:** Z-score on rolling windows
3. **ML:** IsolationForest for complex patterns

### Alert Channels

| Channel | Config | Cooldown |
|---------|--------|----------|
| Telegram | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | 5 min |
| Email | `SMTP_SERVER`, `SMTP_USER` | 10 min |
| PagerDuty | `PAGERDUTY_KEY` | 1 min |

### Example

```python
monitor = ProductionMonitor(
    config=MonitorConfig(),
    alert_channels=[TelegramAlert(), EmailAlert()],
)

# Ingest metrics from robot
monitor.ingest({
    "joint_positions": robot.get_joint_positions(),
    "motor_temps": robot.get_motor_temps(),
    "battery_level": robot.get_battery(),
})

# Check health
health = monitor.get_health()
if health.status == "critical":
    monitor.alert("CRITICAL", f"Health score: {health.score}")
```

---

## DriftDetector

```python
from moses.monitoring.drift import DriftDetector
```

Detects data, concept, and performance drift.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `detect_data_drift()` | `new_data`, `reference` | `dict` | KS-test + PSI |
| `detect_concept_drift()` | `predictions`, `ground_truth` | `dict` | Error rate comparison |
| `detect_performance_drift()` | `metrics`, `baseline` | `dict` | Rolling accuracy degradation |
| `should_retrain()` | — | `bool` | Trigger retraining |

### Drift Types

| Type | Detection | Threshold | Action |
|------|-----------|-----------|--------|
| Data drift | KS-test p < 0.05 | PSI > 0.2 | Log |
| Concept drift | Error rate +10% | Window comparison | Alert |
| Performance drift | Accuracy -5% | Rolling baseline | Retrain |

### Retraining Trigger

```python
detector = DriftDetector(config)

if detector.should_retrain():
    logger.info("Drift detected. Triggering retraining.")
    pipeline.run(trigger="degradation")
```

---

## DashboardApp

```python
from moses.monitoring.dashboard import DashboardApp
```

FastAPI dashboard with real-time SSE updates.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `start()` | `host`, `port` | — | Start dashboard server |
| `update_state()` | `state` | — | Update shared state |
| `get_app()` | — | `FastAPI` | Get ASGI app |

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/state` | GET | Current system state |
| `/api/robots` | GET | Robot fleet status |
| `/api/metrics/{name}` | GET | Metric time series |
| `/stream` | SSE | Real-time updates |

### Frontend

- Chart.js for live plots
- Robot status cards
- Alert feed
- Drift reports

---

*See `monitoring/dashboard.py` for full implementation.*
