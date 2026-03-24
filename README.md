# Necropsy
Automated post-mortem analysis for ML models in production. Detects data drift, diagnoses root cause, and triggers self-healing pipelines; no human intervention required.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![MLflow](https://img.shields.io/badge/MLflow-2.13-orange)
![Evidently](https://img.shields.io/badge/Evidently-0.4-purple)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

---

## The problem this solves

Most ML portfolios show how to **train** a model. This project solves what happens **after** deployment when real-world data changes and your model silently degrades without raising a single exception.

Necropsy monitors production inference, automatically diagnoses *why* performance dropped, and triggers a self-healing pipeline without human intervention. The same pattern used internally at Spotify, Netflix, and Stripe open, reproducible, and running locally in one command.

---

## How it works

```
Production data drifts
        ↓
Autopsy Engine detects drift (PSI + KL-divergence per feature)
        ↓
Decision Router evaluates severity (0.0 → 1.0)
        ↓
retrain | rollback | alert | no_op
        ↓
GitHub Actions triggers retraining pipeline automatically
        ↓
New model promoted to production if metrics improve
```

---

## Quick start

```bash
git clone https://github.com/jonuar/necropsy
cd necropsy
docker compose up --build
```

All four services start automatically:

| Service    | URL                         | Description                    |
|------------|-----------------------------|--------------------------------|
| API        | http://localhost:8000/docs  | Swagger UI — all endpoints     |
| MLflow     | http://localhost:5000       | Model registry and experiments |
| Prometheus | http://localhost:9090       | Raw metrics                    |
| Grafana    | http://localhost:3000       | Live dashboard (admin/necropsy)|

---

## Demo — full pipeline in 3 steps

**Step 1 — Run a prediction:**
```bash
curl.exe -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"amount\": 2.5, \"frequency\": -1.8, \"hour\": 0.3, \"seniority\": -0.5}"
```

**Step 2 — Inject production drift:**
```bash
python scripts/simulate_drift.py
```

**Step 3 — Run the autopsy:**
```bash
curl.exe -X POST http://localhost:8000/autopsy
```

---

## Autopsy report — real output

```
============================================================
NECROPSY AUTOPSY REPORT
============================================================
Timestamp : 2024-01-15T14:32:07+00:00
Model     : necropsy-classifier
Reference : 2000 rows
Production: 500 rows

DRIFT DETECTED — severity: 0.81

Feature analysis:
  CRITICAL   amount       PSI=0.4821  mean +198.3%
  CRITICAL   frequency    PSI=0.2140  mean -49.7%
  OK         hour         PSI=0.0031  mean +1.2%
  OK         seniority    PSI=0.0028  mean -0.8%

Diagnosis : Critical drift detected in 2/4 features. Primary
            driver: amount (PSI=0.4821). Production distribution
            has significantly shifted from training data.
Action    : RETRAIN
============================================================
```

---

## Architecture

```
necropsy/
├── api/
│   ├── main.py              # FastAPI — /predict, /autopsy, /health
│   └── logger.py            # Appends every prediction to production_log.csv
├── autopsy/
│   ├── engine.py            # AutopsyEngine — drift detection + diagnosis
│   ├── detectors.py         # PSI and KL-divergence per feature
│   ├── decision_router.py   # retrain / rollback / alert / no_op logic
│   └── report.py            # Terminal and Markdown report renderers
├── mlops/
│   ├── train.py             # Train + register model in MLflow
│   └── promote.py           # Promote model version after retraining
├── scripts/
│   └── simulate_drift.py    # Inject drift for demos and testing
├── grafana/
│   └── provisioning/        # Auto-provisioned datasource and dashboard
├── .github/workflows/
│   └── retrain.yml          # Triggered automatically on RETRAIN decision
├── prometheus.yml
├── docker-compose.yml
└── requirements.txt
```

---

## API endpoints

| Method | Endpoint          | Description                                      |
|--------|-------------------|--------------------------------------------------|
| GET    | `/health`         | Service health + model loaded status             |
| POST   | `/predict`        | Run inference and log input to production data   |
| POST   | `/autopsy`        | Full drift analysis + diagnosis + decision       |
| GET    | `/autopsy/report` | Latest autopsy as Markdown                       |
| GET    | `/metrics`        | Prometheus metrics                               |

---

## Drift detection — how it works

Necropsy computes **PSI (Population Stability Index)** per feature — a statistical measure of how much the current data distribution has shifted from the training distribution.

| PSI range    | Status   | Action         |
|--------------|----------|----------------|
| < 0.10       | Stable   | no_op          |
| 0.10 – 0.20  | Moderate | alert          |
| > 0.20       | Critical | retrain/rollback|

Individual PSI scores are combined into a **severity score (0.0–1.0)**. Features classified as critical are weighted 2× in the final score. When severity exceeds the configured threshold (default: 0.70), the Decision Router triggers the retraining pipeline.

---

## Stack

| Layer             | Technology              | Role                                        |
|-------------------|-------------------------|---------------------------------------------|
| Inference API     | FastAPI + Uvicorn       | Serves predictions and logs every input     |
| Drift detection   | Evidently AI            | PSI and KL-divergence per feature           |
| Model registry    | MLflow                  | Versioning, stages, artifact storage        |
| Retraining CI/CD  | GitHub Actions          | Auto-triggered on RETRAIN decision          |
| Metrics           | Prometheus              | Scrapes /metrics every 15s                  |
| Dashboard         | Grafana                 | Live model health visualization             |
| Orchestration     | Docker Compose          | One-command reproducible environment        |

---

## Configuration

All thresholds are configurable via environment variables:

```env
MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_NAME=necropsy-classifier
DRIFT_THRESHOLD_PSI=0.20       # PSI above this = drift detected
DRIFT_THRESHOLD_SEVERITY=0.70  # Severity above this = retrain/rollback
REFERENCE_DATA_PATH=data/reference.csv
PRODUCTION_DATA_PATH=data/production_log.csv
```

---

## Why this matters beyond the demo

The pattern here (baseline distribution → drift detection → severity scoring → automated remediation) is identical to what production ML teams implement in fraud detection, recommendation systems, and pricing models.

Necropsy is a concrete, testable implementation of that pattern with a clearly observable ground truth: **did the model degrade when the data shifted?** The answer is always yes and this system catches it before anyone notices.
