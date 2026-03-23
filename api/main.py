import os
import subprocess
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from prometheus_fastapi_instrumentator import Instrumentator

from api.logger import PredictionLogger
from autopsy.engine import AutopsyEngine
from autopsy.decision_router import DecisionRouter
from autopsy.report import render_terminal_report, render_markdown_report

load_dotenv()

app = FastAPI(
    title="Necropsy",
    description="Automated post-mortem analysis for ML models in production.",
    version="0.1.0",
)

# Expose /metrics endpoint for Prometheus to scrape
Instrumentator().instrument(app).expose(app)

# ── Startup: load model and initialize services ──────────────────────────────

logger = PredictionLogger()
engine = AutopsyEngine()
router = DecisionRouter()
model  = None


@app.on_event("startup")
def load_model():
    """
    Loads the registered model from MLflow on startup.
    If MLflow is unreachable the API starts anyway — predictions
    will return 503 until the model is available.

    Backend analogy: same pattern as loading a DB connection pool
    at startup — fail fast and loudly, don't fail silently mid-request.
    """
    global model
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        model_uri = f"models:/{os.getenv('MODEL_NAME')}/1"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded: {model_uri}")
    except Exception as e:
        print(f"WARNING: could not load model — {e}")
        print("Start MLflow server and re-run to load the model.")


# ── Request / Response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    amount:    float
    frequency: float
    hour:      float
    seniority: float

    class Config:
        json_schema_extra = {
            "example": {
                "amount":    0.5,
                "frequency": -1.2,
                "hour":       0.8,
                "seniority": -0.3,
            }
        }


class PredictResponse(BaseModel):
    prediction:  int    # 0 = legitimate, 1 = fraud
    confidence:  float  # probability of the predicted class
    label:       str    # "FRAUD" or "LEGITIMATE"
    logged:      bool   # whether the prediction was saved to disk


class AutopsyResponse(BaseModel):
    drift_detected:      bool
    severity_score:      float
    features_drifted:    int
    features_analyzed:   int
    diagnosis:           str
    recommended_action:  str
    decision:            str
    decision_reason:     str
    production_rows:     int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Standard health check — used by Docker and load balancers
    to verify the service is alive.
    """
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "logged_rows":  logger.row_count(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Runs inference on a single transaction and logs the result.

    Every call to this endpoint builds up the production_log.csv
    that the /autopsy endpoint will analyze for drift.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check MLflow connection.",
        )

    # Build input DataFrame — same shape the model was trained on
    input_df = pd.DataFrame([{
        "amount":    request.amount,
        "frequency": request.frequency,
        "hour":      request.hour,
        "seniority": request.seniority,
    }])

    prediction  = int(model.predict(input_df)[0])
    probability = model.predict_proba(input_df)[0]
    confidence  = float(round(probability.max(), 4))

    # Log every prediction — this is the production data stream
    logger.log(
        amount=request.amount,
        frequency=request.frequency,
        hour=request.hour,
        seniority=request.seniority,
        prediction=prediction,
        confidence=confidence,
    )

    return PredictResponse(
        prediction=prediction,
        confidence=confidence,
        label="FRAUD" if prediction == 1 else "LEGITIMATE",
        logged=True,
    )


@app.post("/autopsy", response_model=AutopsyResponse)
def run_autopsy():
    """
    The core endpoint. Runs the full autopsy pipeline:
      1. Loads reference data (training distribution)
      2. Loads production data (logged predictions)
      3. Runs the Autopsy Engine (drift detection + diagnosis)
      4. Runs the Decision Router (action recommendation)
      5. Saves the markdown report to disk
      6. Returns the structured result

    Call this endpoint periodically (e.g. via cron or GitHub Actions)
    to monitor your model in production.
    """
    # Load reference data
    ref_path = os.getenv("REFERENCE_DATA_PATH", "data/reference.csv")
    if not os.path.exists(ref_path):
        raise HTTPException(
            status_code=500,
            detail="Reference data not found. Run mlops/train.py first.",
        )
    reference = pd.read_csv(ref_path)

    # Load production data
    production = logger.load()
    if production.empty or len(production) < 50:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Not enough production data for analysis "
                f"(got {len(production)} rows, need at least 50). "
                f"Call /predict more times or run simulate_drift.py."
            ),
        )

    # Run the autopsy
    report   = engine.run(reference=reference, production=production)
    decision = router.decide(report)

    # Save markdown report to disk
    os.makedirs("reports", exist_ok=True)
    with open("reports/latest_autopsy.md", "w") as f:
        f.write(render_markdown_report(report))

    # Print to server logs — visible in Docker logs too
    print(render_terminal_report(report))

    if decision.action == "retrain":
        try:
            subprocess.run(["git", "add", "reports/latest_autopsy.md"], check=True)
            subprocess.run(
                ["git", "commit", "-m", f"chore: autopsy report — severity={report.severity_score:.2f} action=retrain"],
                check=True,
            )
            subprocess.run(["git", "push"], check=True)
            print("Autopsy report pushed — GitHub Actions will trigger retraining")
        except subprocess.CalledProcessError as e:
            print(f"Could not push report automatically: {e}")
            print("Push reports/latest_autopsy.md manually to trigger retraining")

    return AutopsyResponse(
        drift_detected=report.drift_detected,
        severity_score=report.severity_score,
        features_drifted=report.features_drifted,
        features_analyzed=report.features_analyzed,
        diagnosis=report.diagnosis,
        recommended_action=report.recommended_action,
        decision=decision.action,
        decision_reason=decision.reason,
        production_rows=report.production_rows,
    )




@app.get("/autopsy/report")
def get_latest_report():
    """
    Returns the raw markdown of the last autopsy run.
    Useful for displaying in a dashboard or GitHub Actions summary.
    """
    path = "reports/latest_autopsy.md"
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="No autopsy report found. Call POST /autopsy first.",
        )
    with open(path) as f:
        return {"report": f.read()}