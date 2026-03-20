import pandas as pd
import numpy as np
import sys
import os

# Add project root to path so imports work from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autopsy.engine import AutopsyEngine
from autopsy.report import render_terminal_report, render_markdown_report


def generate_drifted_data(
    n_samples: int = 500,
    drift_intensity: float = 2.0,
    seed: int = 99,
) -> pd.DataFrame:
    """
    Generates production data where 'amount' and 'frequency' have drifted.
    The other features stay normal — this mimics a real scenario where
    only some features shift (e.g. a promotional event inflates amounts).

    drift_intensity controls how far the distribution shifts:
    1.0 = subtle shift   → PSI ~0.10–0.15 (moderate)
    2.0 = clear shift    → PSI ~0.25–0.40 (critical)
    3.0 = extreme shift  → PSI ~0.50+
    """
    np.random.seed(seed)

    return pd.DataFrame({
        # Drifted: amount shifted by drift_intensity standard deviations
        "amount":    np.random.normal(loc=drift_intensity, scale=1.0, size=n_samples),
        # Drifted: frequency compressed (imagine users churning)
        "frequency": np.random.normal(loc=-drift_intensity * 0.5, scale=0.5, size=n_samples),
        # Stable: these features did not change
        "hour":      np.random.normal(loc=0.0, scale=1.0, size=n_samples),
        "seniority": np.random.normal(loc=0.0, scale=1.0, size=n_samples),
        # Label column — needed for schema compatibility
        "fraud":     np.random.randint(0, 2, size=n_samples),
    })


if __name__ == "__main__":
    # 1. Load reference data saved by train.py
    reference_path = os.getenv("REFERENCE_DATA_PATH", "data/reference.csv")
    if not os.path.exists(reference_path):
        print(f"ERROR: reference data not found at '{reference_path}'")
        print("Run 'python mlops/train.py' first.")
        sys.exit(1)

    reference = pd.read_csv(reference_path)

    # 2. Generate drifted production data
    print("Generating drifted production data...")
    production = generate_drifted_data(drift_intensity=2.0)

    # 3. Save production snapshot for the API to read
    os.makedirs("data", exist_ok=True)
    production.to_csv("data/production_log.csv", index=False)

    # 4. Run the Autopsy Engine
    print("Running Autopsy Engine...\n")
    engine = AutopsyEngine()
    report = engine.run(
        reference=reference,
        production=production,
    )

    # 5. Print terminal report
    print(render_terminal_report(report))

    # 6. Save markdown report to disk
    os.makedirs("reports", exist_ok=True)
    md_path = "reports/latest_autopsy.md"
    with open(md_path, "w") as f:
        f.write(render_markdown_report(report))
    print(f"\nMarkdown report saved to {md_path}")