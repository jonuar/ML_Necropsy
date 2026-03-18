import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

# Tell MLflow where the server is — must match your .env
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

model = mlflow.sklearn.load_model("models:/necropsy-classifier/1")

transaction = pd.DataFrame([{
    "amount":    0.5,
    "frequency": -1.2,
    "hour":       0.8,
    "seniority": -0.3
}])

prediction  = model.predict(transaction)
probability = model.predict_proba(transaction)

print(f"Prediction:  {'FRAUD' if prediction[0] == 1 else 'LEGITIMATE'}")
print(f"Confidence:  {probability[0].max():.1%}")