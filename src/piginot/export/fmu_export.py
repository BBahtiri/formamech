"""FMI 3.0 Co-Simulation FMU exporter and evaluator for verified FormaMech bundles.

Packages a verified FormaMech bundle contract into a standards-compliant FMI 3.0
Co-Simulation ``.fmu`` archive containing:
- ``modelDescription.xml`` (FMI 3.0 schema with tunable envelope-bounded inputs,
  QoI outputs, ``in_envelope``, ``admissible``, ``min_det_f_2d``, and receipt digest)
- ``resources/fmu_manifest.json`` (RFC 8785 canonical bundle & envelope metadata)
- ``resources/runner.py`` (Python FMI 3.0 Co-Simulation wrapper class)
- optional ``resources/model.onnx`` when ``include_onnx=True`` and ``formamech[export]``
  is installed.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from piginot.contracts import canonical_json_bytes

FMU_MANIFEST_SCHEMA: Final = "formamech.fmi3-fmu-manifest-v1"
FMI_VERSION: Final = "3.0"


@dataclass(frozen=True, slots=True)
class FmuExportResult:
    """Auditable record of one packaged FMI 3.0 Co-Simulation FMU archive."""

    fmu_path: Path
    model_identifier: str
    instantiation_token: str
    bundle_manifest_sha256: str
    fmu_sha256: str
    input_variables: tuple[str, ...]
    output_variables: tuple[str, ...]
    includes_onnx: bool

    def document(self) -> dict[str, Any]:
        return {
            "schema": FMU_MANIFEST_SCHEMA,
            "fmi_version": FMI_VERSION,
            "fmu_path": str(self.fmu_path),
            "model_identifier": self.model_identifier,
            "instantiation_token": self.instantiation_token,
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "fmu_sha256": self.fmu_sha256,
            "input_variables": list(self.input_variables),
            "output_variables": list(self.output_variables),
            "includes_onnx": self.includes_onnx,
        }


@dataclass(frozen=True, slots=True)
class FmuStepResult:
    """Result of one FMI 3.0 Co-Simulation ``do_step`` evaluation."""

    current_time: float
    step_size: float
    in_envelope: bool
    admissible: bool
    min_det_f_2d: float
    max_u_mm: float
    qois: dict[str, float]
    bundle_manifest_sha256: str
    verification_tier: str
    receipt_sha256: str

    def document(self) -> dict[str, Any]:
        return {
            "current_time": self.current_time,
            "step_size": self.step_size,
            "in_envelope": self.in_envelope,
            "admissible": self.admissible,
            "min_det_f_2d": self.min_det_f_2d,
            "max_u_mm": self.max_u_mm,
            "qois": dict(self.qois),
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "verification_tier": self.verification_tier,
            "receipt_sha256": self.receipt_sha256,
        }


def _safe_identifier(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"fm_{cleaned}"
    return cleaned


def _build_fmi3_model_description(
    *,
    model_name: str,
    model_identifier: str,
    instantiation_token: str,
    input_bounds: Mapping[str, tuple[float, float, str]],
    qoi_names: tuple[str, ...],
    bundle_manifest_sha256: str,
) -> tuple[bytes, tuple[str, ...], tuple[str, ...]]:
    """Construct a standards-compliant FMI 3.0 ``modelDescription.xml``."""
    root = ET.Element(
        "fmiModelDescription",
        {
            "fmiVersion": FMI_VERSION,
            "modelName": model_name,
            "instantiationToken": instantiation_token,
            "description": (
                "FormaMech verified hyperelastic surrogate Co-Simulation FMU "
                "with executable envelope guard and Tier 0 admissibility ports."
            ),
            "generationTool": "FormaMech",
            "variableNamingConvention": "structured",
        },
    )
    ET.SubElement(
        root,
        "CoSimulation",
        {
            "modelIdentifier": model_identifier,
            "canHandleVariableCommunicationStepSize": "true",
            "providesIntermediateUpdate": "false",
        },
    )
    mv = ET.SubElement(root, "ModelVariables")
    vr = 1
    input_vars: list[str] = []
    output_vars: list[str] = []

    # Mandatory FMI 3.0 independent time variable
    ET.SubElement(
        mv,
        "Float64",
        {
            "name": "time",
            "valueReference": str(vr),
            "causality": "independent",
            "variability": "continuous",
            "description": "Simulation time (s)",
        },
    )
    vr += 1

    # Envelope-bounded input variables
    for name, (lo, hi, unit) in sorted(input_bounds.items()):
        mid = 0.5 * (lo + hi)
        ET.SubElement(
            mv,
            "Float64",
            {
                "name": name,
                "valueReference": str(vr),
                "causality": "input",
                "variability": "continuous",
                "start": f"{mid:.12g}",
                "min": f"{lo:.12g}",
                "max": f"{hi:.12g}",
                "description": f"Envelope-bounded input ({unit})",
            },
        )
        input_vars.append(name)
        vr += 1

    # Safety & admissibility output ports
    for bname, desc in (
        ("in_envelope", "True iff inputs satisfy executable envelope bounds."),
        ("admissible", "True iff Tier 0 zero-FEM admissibility checks passed."),
    ):
        ET.SubElement(
            mv,
            "Boolean",
            {
                "name": bname,
                "valueReference": str(vr),
                "causality": "output",
                "variability": "discrete",
                "description": desc,
            },
        )
        output_vars.append(bname)
        vr += 1

    for fname, desc in (
        ("min_det_f_2d", "Minimum in-plane deformation Jacobian det(F_2D)."),
        ("max_u_mm", "Maximum displacement magnitude ||u|| in mm."),
    ):
        ET.SubElement(
            mv,
            "Float64",
            {
                "name": fname,
                "valueReference": str(vr),
                "causality": "output",
                "variability": "continuous",
                "description": desc,
            },
        )
        output_vars.append(fname)
        vr += 1

    qoi_vrs: list[int] = []
    for qname in qoi_names:
        port = f"qoi_{_safe_identifier(qname)}"
        ET.SubElement(
            mv,
            "Float64",
            {
                "name": port,
                "valueReference": str(vr),
                "causality": "output",
                "variability": "continuous",
                "description": f"Declared bundle QoI: {qname}",
            },
        )
        output_vars.append(port)
        qoi_vrs.append(vr)
        vr += 1

    ET.SubElement(
        mv,
        "String",
        {
            "name": "bundle_manifest_sha256",
            "valueReference": str(vr),
            "causality": "output",
            "variability": "constant",
            "description": "SHA-256 of the verified FormaMech bundle manifest.",
        },
    ).append(ET.Element("Start", {"value": bundle_manifest_sha256}))
    output_vars.append("bundle_manifest_sha256")
    vr += 1

    ms = ET.SubElement(root, "ModelStructure")
    for out_vr in range(2 + len(input_vars), vr):
        ET.SubElement(ms, "Output", {"valueReference": str(out_vr)})

    ET.indent(root, space="  ")
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"
    return xml_bytes, tuple(input_vars), tuple(output_vars)


_RUNNER_SOURCE = '''"""Self-contained FMI 3.0 Co-Simulation Python runner for a FormaMech FMU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FormaMechFmuInstance:
    """Minimal FMI 3.0 Co-Simulation lifecycle adapter backed by FormaMech."""

    def __init__(self, bundle_path: str | Path, manifest_path: str | Path) -> None:
        self.bundle_path = Path(bundle_path)
        self.manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self._inputs: dict[str, float] = {
            k: 0.5 * (float(v["min"]) + float(v["max"]))
            for k, v in self.manifest["envelope_bounds"].items()
        }
        self._outputs: dict[str, Any] = {}
        self._initialized = False

    def enter_initialization_mode(self) -> None:
        self._initialized = False

    def exit_initialization_mode(self) -> None:
        self._initialized = True

    def set_float64(self, name: str, value: float) -> None:
        self._inputs[name] = float(value)

    def do_step(self, current_time: float, step_size: float) -> dict[str, Any]:
        from piginot.bundle import load_bundle
        from piginot.cli import parse_request
        from piginot.serving import ServingMode, serve

        bundle = load_bundle(self.bundle_path)
        req_inputs = {
            k: {
                "value": self._inputs[k],
                "unit": self.manifest["envelope_bounds"][k]["unit"],
            }
            for k in self.manifest["envelope_bounds"]
        }
        parsed, _ = parse_request({"inputs": req_inputs})
        served = serve(
            bundle,
            parsed,
            mode=ServingMode.FAST,
            tier0_check=True,
            emit_receipt=True,
        )
        pred = served.prediction
        in_env = pred.verdict.value == "IN_DOMAIN"
        tier0 = served.tier0_admissibility or {}
        admissible = bool(in_env and tier0.get("passed") is True)
        raw_min_det = tier0.get("min_det_f_2d")
    min_det = float(raw_min_det) if (in_env and raw_min_det is not None) else 0.0
        max_u = 0.0
        if pred.vertices is not None:
            max_u = float(pred.vertices.u.norm(dim=-1).max().item())
        qois = {q.name: float(q.value) for q in pred.qois}
        rc = served.trust_receipt or {}
        self._outputs = {
            "current_time": float(current_time),
            "step_size": float(step_size),
            "in_envelope": in_env,
            "admissible": admissible,
            "min_det_f_2d": min_det,
            "max_u_mm": max_u,
            "qois": qois,
            "bundle_manifest_sha256": bundle.manifest.manifest_sha256,
            "verification_tier": str(rc.get("verification_tier", "REFUSED")),
            "receipt_sha256": str(served.trust_receipt_sha256 or ""),
        }
        return dict(self._outputs)

    def get_outputs(self) -> dict[str, Any]:
        return dict(self._outputs)

    def terminate(self) -> None:
        self._initialized = False
'''


def export_bundle_to_fmu(
    bundle_path: Path | str,
    output_path: Path | str,
    *,
    model_name: str | None = None,
    include_onnx: bool = False,
) -> FmuExportResult:
    """Package a verified FormaMech bundle into an FMI 3.0 Co-Simulation ``.fmu`` archive."""
    from piginot.bundle import load_bundle

    bpath = Path(bundle_path)
    bundle = load_bundle(bpath)
    resolved_name = model_name or f"FormaMech_{bundle.manifest.name}"
    model_id = _safe_identifier(resolved_name)
    token = f"{{{uuid.uuid5(uuid.NAMESPACE_URL, bundle.manifest.manifest_sha256)}}}"

    input_bounds: dict[str, tuple[float, float, str]] = {
        b.name: (
            float(b.lower_bound.value),
            float(b.upper_bound.value),
            b.lower_bound.unit.value,
        )
        for b in bundle.envelope.primitive_bounds
    }
    qoi_names = tuple(q.name for q in bundle.model.qois)
    xml_bytes, input_vars, output_vars = _build_fmi3_model_description(
        model_name=resolved_name,
        model_identifier=model_id,
        instantiation_token=token,
        input_bounds=input_bounds,
        qoi_names=qoi_names,
        bundle_manifest_sha256=bundle.manifest.manifest_sha256,
    )

    manifest_doc = {
        "schema": FMU_MANIFEST_SCHEMA,
        "fmi_version": FMI_VERSION,
        "model_name": resolved_name,
        "model_identifier": model_id,
        "instantiation_token": token,
        "bundle_name": bundle.manifest.name,
        "bundle_manifest_sha256": bundle.manifest.manifest_sha256,
        "weights_sha256": bundle.identities["weights_sha256"],
        "claim_level": bundle.claim_level.value,
        "envelope_bounds": {
            k: {"min": lo, "max": hi, "unit": unit}
            for k, (lo, hi, unit) in sorted(input_bounds.items())
        },
        "qois": list(qoi_names),
        "limitations": str(bundle.limitations.get("paragraph", "")),
        "epistemic_scope": {
            "experimental_validation": False,
            "co_simulation_admissibility_guard": True,
        },
    }

    onnx_bytes: bytes | None = None
    if include_onnx:
        import tempfile

        from piginot.export.onnx_export import export_bundle_to_onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_onnx = Path(tmpdir) / "model.onnx"
            export_bundle_to_onnx(bpath, tmp_onnx, verify_parity=False)
            onnx_bytes = tmp_onnx.read_bytes()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("modelDescription.xml", xml_bytes)
        zf.writestr(
            "resources/fmu_manifest.json",
            canonical_json_bytes(manifest_doc) + b"\n",
        )
        zf.writestr("resources/runner.py", _RUNNER_SOURCE.encode("utf-8"))
        if onnx_bytes is not None:
            zf.writestr("resources/model.onnx", onnx_bytes)

    fmu_sha = hashlib.sha256(out.read_bytes()).hexdigest()
    return FmuExportResult(
        fmu_path=out,
        model_identifier=model_id,
        instantiation_token=token,
        bundle_manifest_sha256=bundle.manifest.manifest_sha256,
        fmu_sha256=fmu_sha,
        input_variables=input_vars,
        output_variables=output_vars,
        includes_onnx=onnx_bytes is not None,
    )


def evaluate_fmu_step(
    fmu_path: Path | str,
    *,
    bundle_path: Path | str,
    inputs: Mapping[str, float],
    current_time: float = 0.0,
    step_size: float = 0.01,
) -> FmuStepResult:
    """Validate an FMI 3.0 ``.fmu`` archive and execute one Co-Simulation step."""
    from piginot.bundle import load_bundle
    from piginot.cli import parse_request
    from piginot.serving import ServingMode, serve

    fpath = Path(fmu_path)
    with zipfile.ZipFile(fpath, "r") as zf:
        xml_root = ET.fromstring(zf.read("modelDescription.xml"))
        if xml_root.attrib.get("fmiVersion") != FMI_VERSION:
            raise ValueError(
                f"expected FMI {FMI_VERSION}, got {xml_root.attrib.get('fmiVersion')!r}"
            )
        manifest = json.loads(zf.read("resources/fmu_manifest.json").decode("utf-8"))

    bundle = load_bundle(Path(bundle_path))
    if bundle.manifest.manifest_sha256 != manifest["bundle_manifest_sha256"]:
        raise ValueError(
            "FMU manifest digest does not match supplied bundle_manifest_sha256"
        )

    bounds = manifest["envelope_bounds"]
    req_inputs = {
        k: {
            "value": float(inputs.get(k, 0.5 * (float(v["min"]) + float(v["max"])))),
            "unit": str(v["unit"]),
        }
        for k, v in bounds.items()
    }
    parsed, _ = parse_request({"inputs": req_inputs})
    served = serve(
        bundle,
        parsed,
        mode=ServingMode.FAST,
        tier0_check=True,
        emit_receipt=True,
    )
    pred = served.prediction
    in_env = pred.verdict.value == "IN_DOMAIN"
    tier0 = served.tier0_admissibility or {}
    admissible = bool(in_env and tier0.get("passed") is True)
    raw_min_det = tier0.get("min_det_f_2d")
    min_det = float(raw_min_det) if (in_env and raw_min_det is not None) else 0.0
    max_u = (
        float(pred.vertices.u.norm(dim=-1).max().item())
        if pred.vertices is not None
        else 0.0
    )
    qois = {q.name: float(q.value) for q in pred.qois}
    rc = served.trust_receipt or {}
    return FmuStepResult(
        current_time=float(current_time),
        step_size=float(step_size),
        in_envelope=in_env,
        admissible=admissible,
        min_det_f_2d=min_det,
        max_u_mm=max_u,
        qois=qois,
        bundle_manifest_sha256=bundle.manifest.manifest_sha256,
        verification_tier=str(rc.get("verification_tier", "REFUSED")),
        receipt_sha256=str(served.trust_receipt_sha256 or ""),
    )
