import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import os
import time
import urllib.request

load_dotenv()

def wait_for_mlflow(uri: str, timeout: int = 60) -> None:
    """
    Blocks until MLflow is reachable or timeout is exceeded.
    Prevents train.py from crashing when Docker starts the api
    container before MLflow is fully ready to accept connections.
    """
    health_url = f"{uri}/health"
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            urllib.request.urlopen(health_url, timeout=3)
            print(f"MLflow is ready at {uri}")
            return
        except Exception:
            print("Waiting for MLflow...")
            time.sleep(3)

    raise RuntimeError(f"MLflow not reachable at {uri} after {timeout}s")




def generate_reference_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset simulating financial transactions.
    In a real project this would be your historical data CSV.

    Each row = one transaction with 4 features:
      - amount:    transaction value
      - frequency: how many transactions that day
      - hour:      time of day
      - seniority: how many days the customer has been active
    """
    np.random.seed(seed)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=["amount", "frequency", "hour", "seniority"])
    df["fraud"] = y  # 0 = legitimate, 1 = fraud
    return df


def train_and_register():
    """
    Trains the model and registers it in MLflow.
    MLflow stores: parameters, metrics, and the serialized model artifact.
    Think of it as 'git commit' but for ML models.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    
    # Wait until MLflow is actually ready — not just "healthy"
    wait_for_mlflow(tracking_uri)
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("necropsy")

    # 1. Generate reference data
    df = generate_reference_data()

    # Save reference data — Evidently will use this later
    # to compare against production data and detect drift
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/reference.csv", index=False)
    print(f"Reference data saved: {len(df)} rows")

    # 2. Split into train and validation (80/20)
    features = ["amount", "frequency", "hour", "seniority"]
    X = df[features]
    y = df["fraud"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train the model
    # RandomForest = an ensemble of decision trees
    # Think of it as: 100 analysts voting in parallel on whether a transaction is fraud
    with mlflow.start_run(run_name="baseline-v1"):
        model = RandomForestClassifier(
            n_estimators=100,  # number of trees in the ensemble
            max_depth=5,       # maximum depth per tree
            random_state=42,
        )
        model.fit(X_train, y_train)

        # 4. Evaluate on validation set
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        # 5. Log everything to MLflow
        # Parameters = model configuration
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)

        # Metrics = how well the model performs
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Serialized model as a tracked artifact
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=os.getenv("MODEL_NAME"),
        )

        print(f"\nModel trained:")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        print(f"\nRegistered in MLflow as '{os.getenv('MODEL_NAME')}'")
        print("Open http://localhost:5000 to inspect it in the MLflow UI")


if __name__ == "__main__":
    train_and_register()