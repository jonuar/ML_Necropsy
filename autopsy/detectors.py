import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class FeatureDriftResult:
    """
    Result of a drift analysis for a single feature.
    Think of this as the 'diff output' for one column of data.
    """
    feature_name: str
    psi_score: float           # Population Stability Index — how much the distribution shifted
    mean_reference: float      # average value when the model was trained
    mean_production: float     # average value right now in production
    mean_delta_pct: float      # percentage change in the mean
    drift_detected: bool       # True if PSI crossed the threshold
    severity: str              # "none" | "moderate" | "critical"


def compute_psi(
    reference: pd.Series,
    production: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-8,
) -> float:
    """
    Computes the Population Stability Index between two distributions.

    PSI answers: "how different is the production distribution
    compared to the reference distribution?"

    The formula compares each bucket of the distribution:
      PSI = sum((prod_pct - ref_pct) * ln(prod_pct / ref_pct))

    You don't need to memorize this — just remember the interpretation:
    PSI < 0.10  → stable, no action needed
    PSI 0.10–0.20 → moderate shift, monitor closely
    PSI > 0.20  → critical shift, retrain the model
    """
    # Build the bucket boundaries from the reference distribution
    breakpoints = np.linspace(
        min(reference.min(), production.min()),
        max(reference.max(), production.max()),
        bins + 1,
    )

    # Count how many values fall in each bucket
    ref_counts  = np.histogram(reference,  bins=breakpoints)[0]
    prod_counts = np.histogram(production, bins=breakpoints)[0]

    # Convert counts to percentages (add epsilon to avoid division by zero)
    ref_pct  = (ref_counts  + epsilon) / (len(reference)  + epsilon * bins)
    prod_pct = (prod_counts + epsilon) / (len(production) + epsilon * bins)

    # PSI formula — higher value = more drift
    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return float(round(psi, 6))


def classify_severity(psi: float) -> str:
    """
    Translates a PSI number into a human-readable severity label.
    These thresholds are industry standard in MLOps.
    """
    if psi < 0.10:
        return "none"
    elif psi < 0.20:
        return "moderate"
    else:
        return "critical"


def analyze_feature(
    feature_name: str,
    reference: pd.Series,
    production: pd.Series,
    psi_threshold: float = 0.10,
) -> FeatureDriftResult:
    """
    Runs the full drift analysis for a single feature column.
    Called once per feature by the AutopsyEngine.
    """
    psi = compute_psi(reference, production)

    mean_ref  = float(reference.mean())
    mean_prod = float(production.mean())

    # Percentage change: how much did the average shift?
    mean_delta_pct = (
        ((mean_prod - mean_ref) / abs(mean_ref)) * 100
        if mean_ref != 0 else 0.0
    )

    return FeatureDriftResult(
        feature_name=feature_name,
        psi_score=psi,
        mean_reference=round(mean_ref, 4),
        mean_production=round(mean_prod, 4),
        mean_delta_pct=round(mean_delta_pct, 2),
        drift_detected=psi >= psi_threshold,
        severity=classify_severity(psi),
    )