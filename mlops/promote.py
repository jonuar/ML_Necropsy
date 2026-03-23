import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os

load_dotenv()


def promote_if_better() -> None:
    """
    Compares the latest trained model against the current production version.
    Promotes the new model to 'production' stage in MLflow only if its
    accuracy is equal or better than the current production model.

    This prevents a bad retraining run from silently replacing a good model —
    the same safety check a CD pipeline does before deploying a new release.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    model_name   = os.getenv("MODEL_NAME", "necropsy-classifier")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Get all registered versions of the model
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print("No model versions found — nothing to promote")
        return

    # Sort by version number — latest first
    versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)
    latest = versions_sorted[0]

    print(f"Latest version: {latest.version} (run_id={latest.run_id})")

    # Get accuracy of the latest run
    latest_run    = client.get_run(latest.run_id)
    latest_acc    = latest_run.data.metrics.get("accuracy", 0.0)

    # Find current production version (if any)
    production_versions = [v for v in versions_sorted if v.current_stage == "Production"]

    if not production_versions:
        # No production version yet — promote directly
        client.transition_model_version_stage(
            name=model_name,
            version=latest.version,
            stage="Production",
        )
        print(f"No previous production model — promoted v{latest.version} to Production")
        print(f"Accuracy: {latest_acc:.3f}")
        return

    current_prod     = production_versions[0]
    current_run      = client.get_run(current_prod.run_id)
    current_acc      = current_run.data.metrics.get("accuracy", 0.0)

    print(f"Current production: v{current_prod.version} (accuracy={current_acc:.3f})")
    print(f"New candidate:      v{latest.version}  (accuracy={latest_acc:.3f})")

    if latest_acc >= current_acc:
        # New model is at least as good — promote it
        client.transition_model_version_stage(
            name=model_name,
            version=latest.version,
            stage="Production",
        )
        # Archive the old production version
        client.transition_model_version_stage(
            name=model_name,
            version=current_prod.version,
            stage="Archived",
        )
        print(f"Promoted v{latest.version} to Production (+{latest_acc - current_acc:.3f} accuracy)")
    else:
        # New model is worse — keep current production
        print(
            f"New model (acc={latest_acc:.3f}) is worse than current "
            f"production (acc={current_acc:.3f}) — keeping v{current_prod.version}"
        )
        print("Retraining did not improve the model — no promotion")


if __name__ == "__main__":
    promote_if_better()