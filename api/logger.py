import pandas as pd
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()


class PredictionLogger:
    """
    Appends every prediction to a CSV file on disk.

    This is the production data source that the Autopsy Engine
    reads to detect drift. In a real system this would write to
    a database or a message queue — CSV works fine for the portfolio.

    Backend analogy: this is your access log, but instead of
    logging HTTP requests, you're logging model inputs so you can
    analyze their distribution over time.
    """

    def __init__(self, log_path: str = None):
        self.log_path = log_path or os.getenv(
            "PRODUCTION_DATA_PATH", "data/production_log.csv"
        )
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log(
        self,
        amount: float,
        frequency: float,
        hour: float,
        seniority: float,
        prediction: int,
        confidence: float,
    ) -> None:
        """
        Appends one prediction record to the log CSV.
        Creates the file with headers if it doesn't exist yet.
        """
        row = pd.DataFrame([{
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "amount":     amount,
            "frequency":  frequency,
            "hour":       hour,
            "seniority":  seniority,
            "prediction": prediction,
            "confidence": confidence,
            "fraud":      prediction,  # alias for schema compatibility with Evidently
        }])

        # Append mode — adds to existing file without overwriting
        header = not os.path.exists(self.log_path)
        row.to_csv(self.log_path, mode="a", header=header, index=False)

    def load(self) -> pd.DataFrame:
        """
        Returns all logged predictions as a DataFrame.
        The Autopsy Engine calls this to get the production data.
        """
        if not os.path.exists(self.log_path):
            return pd.DataFrame()
        return pd.read_csv(self.log_path)

    def row_count(self) -> int:
        if not os.path.exists(self.log_path):
            return 0
        return sum(1 for _ in open(self.log_path)) - 1  # subtract header