import os
from dataclasses import dataclass
from dotenv import load_dotenv

from autopsy.engine import AutopsyReport

load_dotenv()


@dataclass
class RouterDecision:
    """
    The final verdict after evaluating the autopsy report.

    The Decision Router takes the AutopsyEngine's recommendation
    and applies additional business rules on top — for example,
    never rolling back if the current model is already version 1,
    or escalating to a human if drift has been detected 3 days in a row.
    """
    action: str           # "retrain" | "rollback" | "alert" | "no_op"
    reason: str           # plain-English explanation of why
    severity_score: float
    should_notify: bool   # whether to send a Slack/email alert


class DecisionRouter:
    """
    Translates an AutopsyReport into a concrete action.

    Backend analogy: think of this as the circuit breaker pattern.
    When a downstream service degrades past a threshold, the circuit
    breaker trips and routes traffic elsewhere. The DecisionRouter
    does the same — when a model degrades past a threshold, it routes
    to retraining or rollback automatically.

    Action priority (highest to lowest):
    1. rollback  — model is actively harming predictions right now
    2. retrain   — distribution shifted, model needs new training data
    3. alert     — drift detected but not yet severe enough to act
    4. no_op     — everything within expected range, do nothing
    """

    def __init__(self, severity_threshold: float = None):
        self.severity_threshold = severity_threshold or float(
            os.getenv("DRIFT_THRESHOLD_SEVERITY", 0.70)
        )

    def decide(
        self,
        report: AutopsyReport,
        current_model_version: int = 1,
    ) -> RouterDecision:
        """
        Main entry point. Receives an AutopsyReport and returns
        the action to take with a plain-English reason.

        current_model_version is used to prevent rollback when
        there is no previous version to roll back to.
        """
        score = report.severity_score

        # No drift detected — system is healthy
        if not report.drift_detected:
            return RouterDecision(
                action="no_op",
                reason="All features within expected distribution. No action required.",
                severity_score=score,
                should_notify=False,
            )

        # Critical severity — trigger retraining pipeline
        if score >= self.severity_threshold:
            # If we're already on version 1, we can't roll back further
            # In a real system you'd check the MLflow registry here
            if current_model_version <= 1:
                return RouterDecision(
                    action="retrain",
                    reason=(
                        f"Severity {score:.2f} exceeds threshold "
                        f"{self.severity_threshold}. No previous version "
                        f"available for rollback — triggering retraining pipeline."
                    ),
                    severity_score=score,
                    should_notify=True,
                )
            else:
                return RouterDecision(
                    action="rollback",
                    reason=(
                        f"Severity {score:.2f} exceeds threshold "
                        f"{self.severity_threshold}. Rolling back to "
                        f"version {current_model_version - 1} while "
                        f"retraining pipeline runs."
                    ),
                    severity_score=score,
                    should_notify=True,
                )

        # Moderate drift — alert but don't act yet
        return RouterDecision(
            action="alert",
            reason=(
                f"Drift detected (severity={score:.2f}) but below action "
                f"threshold {self.severity_threshold}. Monitoring closely."
            ),
            severity_score=score,
            should_notify=True,
        )