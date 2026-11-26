"""Population-Centric PLM, EU AI Act Annex IV & AS9100 Compliance Exporters (V5-4).

Replaces single-file CAD/CAE point exports with auditable **population-centric**
release packages indexing every evaluated variant, Pareto frontier, pruned OOD
and infeasible counts, watertight B-Rep STEP SHA-256, GD&T tolerance verdicts,
surrogate lineage, break-even ``N*``, and cryptographic HITL signoff links.

Exports:
* ``plm_population_package.jsonld`` (JSON-LD / EU AI Act Annex IV / AS9100),
* ``requirements_traceability.reqif`` (ReqIF 1.2 XML),
* ``eu_ai_act_annex_iv.md`` (human-auditable regulatory technical dossier),
* ``manifest.sha256.json`` (RFC 8785 hash-chain manifest).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from xml.etree import ElementTree as ET

from piginot.contracts.canonical import canonical_json_bytes
from piginot.governance.hitl_signoff import (
    HitlSignoffRecord,
    require_valid_hitl_signoff,
)

PLM_POPULATION_PACKAGE_SCHEMA: Final = "formamech.plm-population-package-v1"
EU_AI_ACT_ANNEX_IV_VERSION: Final = "EU-2024-1689-Annex-IV"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_doc(doc: object) -> str:
    return _sha256_bytes(canonical_json_bytes(doc))


@dataclass(frozen=True, slots=True)
class DesignVariantSummary:
    """One evaluated candidate variant within a generative design population."""

    variant_id: str
    parameters: Mapping[str, float]
    mass_kg: float
    compliance_mj: float
    peak_von_mises_mpa: float
    feasible: bool
    ood_refused: bool = False
    pareto_optimal: bool = False
    trust_receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.variant_id.strip():
            raise ValueError("variant_id must be non-empty")
        for label, val in (
            ("mass_kg", self.mass_kg),
            ("compliance_mj", self.compliance_mj),
            ("peak_von_mises_mpa", self.peak_von_mises_mpa),
        ):
            if not math.isfinite(float(val)) or float(val) < 0.0:
                raise ValueError(f"{label} must be finite and nonnegative")

    def document(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "parameters": {
                str(k): float(v) for k, v in sorted(self.parameters.items())
            },
            "mass_kg": float(self.mass_kg),
            "compliance_mj": float(self.compliance_mj),
            "peak_von_mises_mpa": float(self.peak_von_mises_mpa),
            "feasible": self.feasible,
            "ood_refused": self.ood_refused,
            "pareto_optimal": self.pareto_optimal,
            "trust_receipt_sha256": self.trust_receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class DesignPopulationPackage:
    """Population-centric PLM & EU AI Act Annex IV release package."""

    package_id: str
    intended_purpose: str
    operational_design_domain: str
    surrogate_model_id: str
    training_archive_sha256: str
    numerical_claim_level: str
    experimental_validation_tier: str
    mean_physical_error_summary: Mapping[str, float]
    worst_case_physical_error_summary: Mapping[str, float]
    variants: tuple[DesignVariantSummary, ...]
    promoted_variant_id: str
    promoted_brep_step_sha256: str | None = None
    gdt_robustness_passed: bool | None = None
    break_even_n_star: float | None = None
    regime_guards_enabled: tuple[str, ...] = (
        "bc_layout_ood",
        "yield_onset",
        "buckling_coercivity",
        "contact_gap_closure",
    )
    compliance_standards: tuple[str, ...] = (
        EU_AI_ACT_ANNEX_IV_VERSION,
        "AS9100D",
        "IATF-16949",
        "ASME-VV10",
    )
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.package_id.strip():
            raise ValueError("package_id must be non-empty")
        if not self.variants:
            raise ValueError("variants must be non-empty")
        ids = {v.variant_id for v in self.variants}
        if self.promoted_variant_id not in ids:
            raise ValueError(
                f"promoted_variant_id {self.promoted_variant_id!r} not found in variants"
            )

    @property
    def pareto_variant_ids(self) -> tuple[str, ...]:
        return tuple(v.variant_id for v in self.variants if v.pareto_optimal)

    @property
    def pruned_ood_count(self) -> int:
        return sum(1 for v in self.variants if v.ood_refused)

    @property
    def pruned_infeasible_count(self) -> int:
        return sum(1 for v in self.variants if not v.feasible)

    def document(self) -> dict[str, Any]:
        return {
            "@context": {
                "@vocab": "https://formamech.org/schema/plm/v1#",
                "eu_ai_act": "https://eur-lex.europa.eu/eli/reg/2024/1689/oj#",
            },
            "@type": "DesignPopulationPackage",
            "schema": PLM_POPULATION_PACKAGE_SCHEMA,
            "package_id": self.package_id,
            "compliance_standards": list(self.compliance_standards),
            "intended_purpose": self.intended_purpose,
            "operational_design_domain": self.operational_design_domain,
            "lineage": {
                "surrogate_model_id": self.surrogate_model_id,
                "training_archive_sha256": self.training_archive_sha256,
                "numerical_claim_level": self.numerical_claim_level,
                "experimental_validation_tier": self.experimental_validation_tier,
                "break_even_n_star": self.break_even_n_star,
            },
            "accuracy_and_safety": {
                "mean_physical_error": {
                    str(k): float(v)
                    for k, v in sorted(self.mean_physical_error_summary.items())
                },
                "worst_case_physical_error": {
                    str(k): float(v)
                    for k, v in sorted(self.worst_case_physical_error_summary.items())
                },
                "regime_guards_enabled": list(self.regime_guards_enabled),
                "gdt_robustness_passed": self.gdt_robustness_passed,
            },
            "population_summary": {
                "total_evaluated_variants": len(self.variants),
                "pareto_optimal_count": len(self.pareto_variant_ids),
                "pareto_variant_ids": list(self.pareto_variant_ids),
                "pruned_ood_count": self.pruned_ood_count,
                "pruned_infeasible_count": self.pruned_infeasible_count,
                "promoted_variant_id": self.promoted_variant_id,
                "promoted_brep_step_sha256": self.promoted_brep_step_sha256,
            },
            "variants": [v.document() for v in self.variants],
            "metadata": {str(k): str(v) for k, v in sorted(self.metadata.items())},
        }

    @property
    def package_sha256(self) -> str:
        return _sha256_doc(self.document())


def _build_reqif_xml(
    package: DesignPopulationPackage, signoff: HitlSignoffRecord | None
) -> bytes:
    """Emit a ReqIF 1.2 compatible XML requirements traceability file."""
    ns = "http://www.omg.org/spec/ReqIF/20110401/reqif.xsd"
    root = ET.Element("REQ-IF", {"xmlns": ns})
    header = ET.SubElement(root, "THE-HEADER")
    req_hdr = ET.SubElement(
        header, "REQ-IF-HEADER", {"IDENTIFIER": f"hdr-{package.package_id}"}
    )
    ET.SubElement(
        req_hdr, "TITLE"
    ).text = f"FormaMech PLM Traceability — {package.package_id}"
    ET.SubElement(req_hdr, "SOURCE-TOOL-ID").text = "FormaMech v0.5"

    core = ET.SubElement(root, "CORE-CONTENT")
    content = ET.SubElement(core, "REQ-IF-CONTENT")
    objs = ET.SubElement(content, "SPEC-OBJECTS")

    reqs = [
        ("REQ-ODD", "Operational Design Domain", package.operational_design_domain),
        (
            "REQ-EPISTEMIC",
            "ASME V&V 10 Epistemic Tier",
            f"Numerical={package.numerical_claim_level}; Experimental={package.experimental_validation_tier}",
        ),
        (
            "REQ-GDT",
            "GD&T Tolerance Robustness",
            f"passed={package.gdt_robustness_passed}; brep_sha256={package.promoted_brep_step_sha256}",
        ),
        (
            "REQ-HITL",
            "Senior Engineer HITL Signoff",
            (
                f"state={signoff.state.value}; reviewer={signoff.reviewer_id}; sig={signoff.signature_sha256}"
                if signoff is not None
                else "NONE"
            ),
        ),
    ]
    for rid, name, desc in reqs:
        obj = ET.SubElement(objs, "SPEC-OBJECT", {"IDENTIFIER": rid, "LONG-NAME": name})
        ET.SubElement(obj, "DESC").text = desc
    return bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))


def _build_annex_iv_markdown(
    package: DesignPopulationPackage, signoff: HitlSignoffRecord | None
) -> str:
    lines = [
        f"# EU AI Act Annex IV & AS9100 Technical Dossier — `{package.package_id}`",
        "",
        f"- **Package SHA-256:** `{package.package_sha256}`",
        f"- **Standards:** {', '.join(package.compliance_standards)}",
        f"- **Intended Purpose:** {package.intended_purpose}",
        f"- **Operational Design Domain (ODD):** {package.operational_design_domain}",
        "",
        "## 1. Epistemic Verification & Provenance",
        f"- **Surrogate Model ID:** `{package.surrogate_model_id}`",
        f"- **Training Archive SHA-256:** `{package.training_archive_sha256}`",
        f"- **Numerical Claim Level (FEM agreement only):** `{package.numerical_claim_level}`",
        f"- **Experimental Validation Tier (ASME V&V 10):** `{package.experimental_validation_tier}`",
        f"- **Regime Guards Active:** {', '.join(package.regime_guards_enabled)}",
        "",
        "## 2. Physical-Unit Accuracy (Mean & Worst-Case)",
        f"- **Mean Physical Errors:** `{dict(sorted(package.mean_physical_error_summary.items()))}`",
        f"- **Worst-Case Physical Errors:** `{dict(sorted(package.worst_case_physical_error_summary.items()))}`",
        "",
        "## 3. Design Population & Promoted Release",
        f"- **Total Evaluated Variants:** {len(package.variants)} "
        f"(Pareto: {len(package.pareto_variant_ids)}, "
        f"Pruned OOD: {package.pruned_ood_count}, "
        f"Pruned Infeasible: {package.pruned_infeasible_count})",
        f"- **Promoted Variant:** `{package.promoted_variant_id}`",
        f"- **Promoted Watertight B-Rep STEP SHA-256:** `{package.promoted_brep_step_sha256}`",
        f"- **GD&T Tolerance Robustness Passed:** `{package.gdt_robustness_passed}`",
        "",
        "## 4. Human Oversight (HITL Senior-Engineer Signoff)",
    ]
    if signoff is not None:
        lines.extend(
            [
                f"- **State:** `{signoff.state.value}` | **Reviewer:** `{signoff.reviewer_id}` (`{signoff.reviewer_role.value}`)",
                f"- **Decision Note:** {signoff.decision_note}",
                f"- **RFC 8785 Signature SHA-256:** `{signoff.signature_sha256}`",
            ]
        )
    else:
        lines.append("- **State:** `UNVERIFIED (HITL signoff not required by policy)`")
    lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class PlmExportResult:
    """Exported PLM compliance bundle paths and verified RFC 8785 manifest digest."""

    output_dir: Path
    jsonld_path: Path
    reqif_path: Path
    markdown_path: Path
    manifest_path: Path
    package_sha256: str
    manifest_sha256: str


def export_plm_compliance_package(
    package: DesignPopulationPackage,
    output_dir: Path | str,
    *,
    hitl_signoff: HitlSignoffRecord | None = None,
    require_hitl_signoff: bool = True,
) -> PlmExportResult:
    """Export population-centric PLM JSON-LD + ReqIF XML + EU AI Act Annex IV Markdown.

    Raises :class:`~piginot.contracts.state_machine.InvalidTransitionError` when
    ``require_hitl_signoff=True`` and the senior-engineer signoff is missing,
    not ``APPROVED``, bound to a different ``package_sha256``, or tampered.
    """
    pkg_sha = package.package_sha256
    if require_hitl_signoff:
        require_valid_hitl_signoff(hitl_signoff, expected_artifact_sha256=pkg_sha)

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    jsonld_doc = package.document()
    if hitl_signoff is not None:
        jsonld_doc["human_oversight"] = hitl_signoff.document()
    jsonld_bytes = canonical_json_bytes(jsonld_doc) + b"\n"
    jsonld_path = out / "plm_population_package.jsonld"
    jsonld_path.write_bytes(jsonld_bytes)

    reqif_bytes = _build_reqif_xml(package, hitl_signoff)
    reqif_path = out / "requirements_traceability.reqif"
    reqif_path.write_bytes(reqif_bytes)

    md_text = _build_annex_iv_markdown(package, hitl_signoff)
    md_bytes = md_text.encode("utf-8")
    md_path = out / "eu_ai_act_annex_iv.md"
    md_path.write_bytes(md_bytes)

    manifest_doc = {
        "schema": "formamech.plm-export-manifest-v1",
        "package_id": package.package_id,
        "package_sha256": pkg_sha,
        "hitl_signature_sha256": (
            None if hitl_signoff is None else hitl_signoff.signature_sha256
        ),
        "files": {
            jsonld_path.name: _sha256_bytes(jsonld_bytes),
            reqif_path.name: _sha256_bytes(reqif_bytes),
            md_path.name: _sha256_bytes(md_bytes),
        },
    }
    manifest_bytes = canonical_json_bytes(manifest_doc) + b"\n"
    manifest_path = out / "manifest.sha256.json"
    manifest_path.write_bytes(manifest_bytes)

    return PlmExportResult(
        output_dir=out,
        jsonld_path=jsonld_path,
        reqif_path=reqif_path,
        markdown_path=md_path,
        manifest_path=manifest_path,
        package_sha256=pkg_sha,
        manifest_sha256=_sha256_bytes(manifest_bytes),
    )


def verify_plm_compliance_package(output_dir: Path | str) -> dict[str, Any]:
    """Re-hash every file in an exported PLM compliance package and verify hash integrity."""
    import json

    out = Path(output_dir).resolve()
    manifest_path = out / "manifest.sha256.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.sha256.json not found in {out}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for fname, expected in manifest.get("files", {}).items():
        fpath = out / str(fname)
        if not fpath.is_file():
            raise ValueError(f"missing exported PLM file {fname!r}")
        actual = _sha256_bytes(fpath.read_bytes())
        if actual != expected:
            raise ValueError(f"PLM package file {fname!r} SHA-256 mismatch (tampered)")
    return {
        "ok": True,
        "package_id": manifest["package_id"],
        "package_sha256": manifest["package_sha256"],
        "file_count": len(manifest.get("files", {})),
    }


__all__ = (
    "EU_AI_ACT_ANNEX_IV_VERSION",
    "PLM_POPULATION_PACKAGE_SCHEMA",
    "DesignPopulationPackage",
    "DesignVariantSummary",
    "PlmExportResult",
    "export_plm_compliance_package",
    "verify_plm_compliance_package",
)
