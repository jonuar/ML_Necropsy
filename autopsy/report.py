from dataclasses import dataclass
from autopsy.engine import AutopsyReport


SEVERITY_ICONS = {
    "none":     "OK      ",
    "moderate": "WARNING ",
    "critical": "CRITICAL",
}


def render_terminal_report(report: AutopsyReport) -> str:
    """
    Renders the AutopsyReport as a formatted terminal string.
    This is what you'll see in the logs and in the README demo.

    Designed to be scannable at a glance — worst offenders first.
    """
    lines = []

    lines.append("=" * 60)
    lines.append("NECROPSY AUTOPSY REPORT")
    lines.append("=" * 60)
    lines.append(f"Timestamp : {report.timestamp}")
    lines.append(f"Model     : {report.model_name}")
    lines.append(f"Reference : {report.reference_rows} rows")
    lines.append(f"Production: {report.production_rows} rows")
    lines.append("")

    # Severity banner — the first thing an on-call engineer sees
    if report.drift_detected:
        lines.append(f"DRIFT DETECTED — severity: {report.severity_score:.2f}")
    else:
        lines.append(f"STATUS: STABLE — severity: {report.severity_score:.2f}")
    lines.append("")

    # Feature breakdown — sorted by PSI descending (worst first)
    lines.append("Feature analysis:")
    sorted_results = sorted(
        report.feature_results,
        key=lambda r: r.psi_score,
        reverse=True,
    )
    for r in sorted_results:
        icon = SEVERITY_ICONS.get(r.severity, "       ")
        delta_sign = "+" if r.mean_delta_pct > 0 else ""
        lines.append(
            f"  {icon}  {r.feature_name:<12} "
            f"PSI={r.psi_score:.4f}  "
            f"mean {delta_sign}{r.mean_delta_pct:.1f}%"
        )

    lines.append("")
    lines.append(f"Diagnosis : {report.diagnosis}")
    lines.append(f"Action    : {report.recommended_action.upper()}")
    lines.append("=" * 60)

    return "\n".join(lines)


def render_markdown_report(report: AutopsyReport) -> str:
    """
    Renders the AutopsyReport as a Markdown file.
    Saved to disk after each autopsy run — becomes the paper trail
    that GitHub Actions reads to decide whether to trigger retraining.
    """
    drifted = [r for r in report.feature_results if r.drift_detected]
    stable  = [r for r in report.feature_results if not r.drift_detected]

    md = f"""# Necropsy Autopsy Report

**Timestamp:** {report.timestamp}
**Model:** `{report.model_name}`
**Severity score:** `{report.severity_score:.3f}`
**Recommended action:** `{report.recommended_action}`

---

## Summary

{report.diagnosis}

| Metric | Value |
|--------|-------|
| Features analyzed | {report.features_analyzed} |
| Features with drift | {report.features_drifted} |
| Reference rows | {report.reference_rows} |
| Production rows | {report.production_rows} |

---

## Feature breakdown

### Drifted features
"""
    if drifted:
        md += "| Feature | PSI | Mean (ref) | Mean (prod) | Delta | Severity |\n"
        md += "|---------|-----|------------|-------------|-------|----------|\n"
        for r in sorted(drifted, key=lambda r: r.psi_score, reverse=True):
            sign = "+" if r.mean_delta_pct > 0 else ""
            md += (
                f"| `{r.feature_name}` | {r.psi_score:.4f} | "
                f"{r.mean_reference:.4f} | {r.mean_production:.4f} | "
                f"{sign}{r.mean_delta_pct:.1f}% | **{r.severity}** |\n"
            )
    else:
        md += "_No features drifted._\n"

    md += "\n### Stable features\n"
    if stable:
        for r in stable:
            md += f"- `{r.feature_name}` — PSI={r.psi_score:.4f}\n"
    else:
        md += "_All features showed drift._\n"

    return md