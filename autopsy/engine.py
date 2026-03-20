import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv
import os

from autopsy.detectors import FeatureDriftResult, analyze_feature

load_dotenv()


@dataclass
class AutopsyReport:
    """
    The structured output of the Autopsy Engine.
    Everything the Decision Router needs to decide what action to take.

    This is the equivalent of a structured error response in a REST API —
    instead of { "error": "something failed" }, you get a full diagnosis.
    """
    timestamp: str
    model_name: str
    severity_score: float          # 0.0 (healthy) to 1.0 (critical)
    drift_detected: bool
    features_analyzed: int
    features_drifted: int
    feature_results: list[FeatureDriftResult]
    diagnosis: str                 # human-readable root cause
    recommended_action: str        # "retrain" | "rollback" | "alert" | "no_op"
    reference_rows: int
    production_rows: int


class AutopsyEngine:
    """
    The core of Necropsy.

    Receives reference data (what the model was trained on) and
    production data (what it's seeing right now), compares their
    distributions feature by feature, and produces an AutopsyReport.

    Backend analogy: think of this as a middleware that intercepts
    every request/response cycle, compares the payload schema against
    the expected contract, and raises a structured alert when something
    deviates — except instead of JSON schemas, it compares statistical
    distributions.
    """

    FEATURES = ["amount", "frequency", "hour", "seniority"]

    def __init__(
        self,
        psi_threshold: float = None,
        severity_threshold: float = None,
    ):
        # Read thresholds from .env — keeps config out of code
        self.psi_threshold = psi_threshold or float(
            os.getenv("DRIFT_THRESHOLD_PSI", 0.10)
        )
        self.severity_threshold = severity_threshold or float(
            os.getenv("DRIFT_THRESHOLD_SEVERITY", 0.70)
        )

    def run(
        self,
        reference: pd.DataFrame,
        production: pd.DataFrame,
        model_name: str = None,
    ) -> AutopsyReport:
        """
        Main entry point. Call this with your two DataFrames
        and get back a full autopsy report.

        Usage:
            engine = AutopsyEngine()
            report = engine.run(reference_df, production_df)
            print(report.recommended_action)  # "retrain" | "no_op" | ...
        """
        model_name = model_name or os.getenv("MODEL_NAME", "unknown-model")

        # 1. Analyze each feature independently
        feature_results = [
            analyze_feature(
                feature_name=feat,
                reference=reference[feat],
                production=production[feat],
                psi_threshold=self.psi_threshold,
            )
            for feat in self.FEATURES
            if feat in reference.columns and feat in production.columns
        ]

        # 2. Compute overall severity score
        # Formula: weighted average of PSI scores, capped at 1.0
        # Features with critical drift count double — they matter more
        severity_score = self._compute_severity(feature_results)

        # 3. Count how many features drifted
        drifted = [r for r in feature_results if r.drift_detected]
        drift_detected = len(drifted) > 0

        # 4. Generate a human-readable diagnosis
        diagnosis = self._diagnose(feature_results, severity_score)

        # 5. Recommend an action based on severity
        action = self._recommend_action(severity_score, drift_detected)

        return AutopsyReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=model_name,
            severity_score=round(severity_score, 3),
            drift_detected=drift_detected,
            features_analyzed=len(feature_results),
            features_drifted=len(drifted),
            feature_results=feature_results,
            diagnosis=diagnosis,
            recommended_action=action,
            reference_rows=len(reference),
            production_rows=len(production),
        )

    def _compute_severity(self, results: list[FeatureDriftResult]) -> float:
        """
        Converts individual PSI scores into a single 0–1 severity number.

        Critical features (PSI > 0.20) are weighted 2x because they
        represent a fundamental distribution change, not just noise.
        """
        if not results:
            return 0.0

        weighted_sum = 0.0
        weight_total = 0.0

        for r in results:
            weight = 2.0 if r.severity == "critical" else 1.0
            weighted_sum += r.psi_score * weight
            weight_total += weight

        raw = weighted_sum / weight_total if weight_total > 0 else 0.0

        # Normalize: PSI of 0.5 maps roughly to severity 1.0
        normalized = min(raw / 0.5, 1.0)
        return normalized

    def _diagnose(
        self,
        results: list[FeatureDriftResult],
        severity_score: float,
    ) -> str:
        """
        Generates a plain-English diagnosis based on which features drifted
        and how much. This is what appears in the autopsy report summary.
        """
        drifted = [r for r in results if r.drift_detected]

        if not drifted:
            return "All features within expected distribution. Model stable."

        # Find the feature with the highest PSI — the main culprit
        worst = max(drifted, key=lambda r: r.psi_score)

        # Build a description of each drifted feature
        detail = ", ".join(
            f"{r.feature_name} (PSI={r.psi_score:.3f}, "
            f"mean {'+' if r.mean_delta_pct > 0 else ''}{r.mean_delta_pct:.1f}%)"
            for r in sorted(drifted, key=lambda r: r.psi_score, reverse=True)
        )

        if severity_score >= 0.70:
            return (
                f"Critical drift detected in {len(drifted)}/{len(results)} features. "
                f"Primary driver: {worst.feature_name} "
                f"(PSI={worst.psi_score:.3f}). "
                f"Affected features: {detail}. "
                f"Production distribution has significantly shifted from training data."
            )
        else:
            return (
                f"Moderate drift in {len(drifted)}/{len(results)} features. "
                f"Affected: {detail}. Monitor closely."
            )

    def _recommend_action(
        self,
        severity_score: float,
        drift_detected: bool,
    ) -> str:
        """
        Decision logic: maps severity to a concrete action.

        This is intentionally simple — the Decision Router in
        decision_router.py handles the full logic with rollback conditions.
        The engine only makes the first-pass recommendation.
        """
        if not drift_detected:
            return "no_op"
        elif severity_score >= self.severity_threshold:
            return "retrain"
        else:
            return "alert"