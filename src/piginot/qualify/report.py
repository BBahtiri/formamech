"""Audit-ready Markdown report renderer for qualification manifests."""

from __future__ import annotations

from piginot.qualify.manifest import QualificationManifest


def _fmt_float(value: object, digits: int = 4) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{digits}g}"
    return "N/A"


def render_qualification_report_markdown(manifest: QualificationManifest) -> str:
    """Render a deterministic, audit-ready Markdown qualification dossier."""
    lines: list[str] = [
        "# FormaMech Numerical Model Qualification Report",
        "",
        f"- **Verdict:** `{manifest.verdict.value}`",
        f"- **Granted Claim Level:** `{manifest.granted_claim_level.value}` "
        f"(Target: `{manifest.target_claim_level.value}`)",
        f"- **Bundle:** `{manifest.bundle_name}`",
        f"- **Bundle Manifest SHA-256:** `{manifest.bundle_manifest_sha256}`",
        f"- **Approved Envelope SHA-256:** `{manifest.approved_envelope_sha256}`",
        f"- **Policy:** `{manifest.policy_id}` (`{manifest.policy_sha256}`)",
        f"- **Qualification Manifest SHA-256:** `{manifest.manifest_sha256}`",
        f"- **FEM Solver Backend:** `{manifest.solver_backend_id}`",
        f"- **Created (UTC):** `{manifest.created_utc}`",
    ]
    if manifest.calibration_sha256 is not None:
        lines.append(f"- **CAD Calibration SHA-256:** `{manifest.calibration_sha256}`")
    lines.extend(
        [
            "",
            "## Epistemic Scope (ASME V&V 10)",
            "",
            f"- `numerical_verification_only`: `{manifest.epistemic_scope['numerical_verification_only']}`"
            if manifest.epistemic_scope
            else "- `numerical_verification_only`: `True`",
            f"- `experimental_validation`: `{manifest.epistemic_scope['experimental_validation']}`"
            if manifest.epistemic_scope
            else "- `experimental_validation`: `False`",
            f"- `regulatory_certification`: `{manifest.epistemic_scope['regulatory_certification']}`"
            if manifest.epistemic_scope
            else "- `regulatory_certification`: `False`",
            "",
            str(
                (manifest.epistemic_scope or {}).get(
                    "statement",
                    "Numerical verification against the declared discrete FEM reference.",
                )
            ),
            "",
            "## Verification Families",
            "",
            "| Family | Required | Status | Missing Checks |",
            "| :--- | :---: | :---: | :--- |",
        ]
    )
    for fam in manifest.families:
        missing = fam.get("missing_checks") or []
        missing_str = ", ".join(str(m) for m in missing) if missing else "None"
        lines.append(
            f"| `{fam['family_id']}` | `{fam['required']}` | "
            f"`{fam['status']}` | {missing_str} |"
        )

    lines.extend(
        [
            "",
            "## Detailed Check Outcomes",
            "",
            "| Check ID | Family | Status | Detail |",
            "| :--- | :--- | :---: | :--- |",
        ]
    )
    for chk in manifest.checks:
        lines.append(
            f"| `{chk.check_id}` | `{chk.family_id}` | `{chk.status.value}` | {chk.detail} |"
        )

    lines.extend(["", "## Summary Metrics", ""])
    for label, metrics in (
        ("DEV / Calibration Partition", manifest.dev_metrics),
        ("HOLDOUT Partition", manifest.holdout_metrics),
    ):
        if metrics is None:
            continue
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- Cases evaluated: `{metrics.get('case_count', 0)}`")
        lines.append(
            f"- Displacement rel-L2 (quantile): `{_fmt_float(metrics.get('disp_rel_l2_quantile'))}` "
            f"(max: `{_fmt_float(metrics.get('disp_rel_l2_max'))}`)"
        )
        lines.append(
            f"- Stress rel-L2 (quantile): `{_fmt_float(metrics.get('stress_rel_l2_quantile'))}` "
            f"(max: `{_fmt_float(metrics.get('stress_rel_l2_max'))}`)"
        )
        lines.append(f"- Minimum det(F_2D): `{_fmt_float(metrics.get('min_det_f'))}`")
        lines.append("")

    if manifest.amortization is not None:
        amort = manifest.amortization
        lines.extend(
            [
                "## Economics & Break-Even Amortization (`N*`)",
                "",
                f"- Online speedup: `{_fmt_float(amort.get('online_speedup'))}x` "
                f"(FEM `{_fmt_float(amort.get('fem_seconds'))}s` vs. "
                f"Inference `{_fmt_float(amort.get('inference_seconds'))}s`)",
                f"- Calibration-only break-even `N*_cal`: "
                f"`{_fmt_float(amort.get('calibration_only_break_even_queries'))}` queries",
                f"- Full break-even `N*`: `{_fmt_float(amort.get('break_even_queries'))}` queries",
                "",
            ]
        )

    lines.extend(["## Limitations", "", manifest.limitations, ""])
    return "\n".join(lines)


__all__ = ("render_qualification_report_markdown",)
