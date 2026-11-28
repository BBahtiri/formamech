"""Bidirectional MuJoCo / Newton / ROS 2 / FMI 3.0 robotics interop bridges (V6-R4).

Zero-heavy-dependency pure-Python adapters allowing:
1. Importing contact wrench and joint trajectories logged from Newton, MuJoCo,
   Genesis, or ROS 2 into FormaMech for certified FEM stress, fatigue, and
   hardware co-design verification.
2. Exporting verified FormaMech soft-link, tactile, or compliant-gripper
   surrogates as MuJoCo MJCF compliance fragments, ROS 2 safety/wrench node
   contracts, and FMI 3.0 robotics co-simulation manifests.
"""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from piginot.contracts.canonical import canonical_json_bytes

FloatArray = NDArray[np.float64]
WRENCH_TRACE_SCHEMA: Final = "formamech.simulator-wrench-trace-v1"
MUJOCO_COMPLIANCE_SCHEMA: Final = "formamech.mujoco-compliance-plugin-v1"
ROS2_CONTRACT_SCHEMA: Final = "formamech.ros2-safety-node-v1"


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


@dataclass(frozen=True, slots=True)
class SimulatorWrenchTrace:
    """Canonical imported time-series of 6D wrenches from MuJoCo/Newton/ROS 2."""

    source_simulator: str
    body_name: str
    times_s: FloatArray  # (T,)
    forces_n: FloatArray  # (T, 3)
    torques_n_mm: FloatArray  # (T, 3)
    peak_force_n: float
    peak_torque_n_mm: float
    trace_sha256: str

    def document(self) -> dict[str, Any]:
        return {
            "schema": WRENCH_TRACE_SCHEMA,
            "source_simulator": self.source_simulator,
            "body_name": self.body_name,
            "sample_count": int(self.times_s.size),
            "duration_s": float(self.times_s[-1] - self.times_s[0])
            if self.times_s.size > 1
            else 0.0,
            "peak_force_n": float(self.peak_force_n),
            "peak_torque_n_mm": float(self.peak_torque_n_mm),
            "trace_sha256": self.trace_sha256,
        }


def import_simulator_wrench_trace(
    path: str | Path,
    *,
    source_simulator: str = "mujoco",
    body_name: str = "end_effector",
    torque_unit_scale_to_n_mm: float = 1000.0,
) -> SimulatorWrenchTrace:
    """Import a CSV or JSON wrench log ``(t, fx, fy, fz, tx, ty, tz)`` cleanly."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"wrench trace file not found: {p}")

    times: list[float] = []
    forces: list[list[float]] = []
    torques: list[list[float]] = []

    if p.suffix.lower() == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
        records: Sequence[Mapping[str, Any]] = (
            raw["records"] if isinstance(raw, dict) and "records" in raw else raw
        )
        for row in records:
            times.append(float(row["t"]))
            forces.append([float(row["fx"]), float(row["fy"]), float(row["fz"])])
            torques.append(
                [
                    float(row.get("tx", 0.0)) * torque_unit_scale_to_n_mm,
                    float(row.get("ty", 0.0)) * torque_unit_scale_to_n_mm,
                    float(row.get("tz", 0.0)) * torque_unit_scale_to_n_mm,
                ]
            )
    else:
        with p.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                times.append(float(row["t"]))
                forces.append([float(row["fx"]), float(row["fy"]), float(row["fz"])])
                torques.append(
                    [
                        float(row.get("tx", "0.0")) * torque_unit_scale_to_n_mm,
                        float(row.get("ty", "0.0")) * torque_unit_scale_to_n_mm,
                        float(row.get("tz", "0.0")) * torque_unit_scale_to_n_mm,
                    ]
                )

    if not times:
        raise ValueError("wrench trace file contains zero rows")
    t_arr = np.asarray(times, dtype=np.float64)
    f_arr = np.asarray(forces, dtype=np.float64)
    tau_arr = np.asarray(torques, dtype=np.float64)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    return SimulatorWrenchTrace(
        source_simulator=source_simulator,
        body_name=body_name,
        times_s=t_arr,
        forces_n=f_arr,
        torques_n_mm=tau_arr,
        peak_force_n=float(np.max(np.linalg.norm(f_arr, axis=1))),
        peak_torque_n_mm=float(np.max(np.linalg.norm(tau_arr, axis=1))),
        trace_sha256=digest,
    )


def export_mujoco_compliance_plugin(
    output_dir: str | Path,
    *,
    component_name: str,
    stiffness_xyz_n_per_m: tuple[float, float, float],
    damping_xyz_n_s_per_m: tuple[float, float, float],
    mass_kg: float,
    max_safe_force_n: float,
    step_path: str | None = None,
    bundle_sha256: str = "",
) -> dict[str, Any]:
    """Export an MJCF XML compliance fragment + JSON manifest for MuJoCo / Newton."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    root = ET.Element("mujoco", {"model": f"formamech_{component_name}"})
    custom = ET.SubElement(root, "custom")
    ET.SubElement(
        custom,
        "numeric",
        {
            "name": f"{component_name}_stiffness_n_per_m",
            "data": " ".join(f"{float(v):.6g}" for v in stiffness_xyz_n_per_m),
        },
    )
    ET.SubElement(
        custom,
        "numeric",
        {
            "name": f"{component_name}_damping_n_s_per_m",
            "data": " ".join(f"{float(v):.6g}" for v in damping_xyz_n_s_per_m),
        },
    )
    ET.SubElement(
        custom,
        "numeric",
        {
            "name": f"{component_name}_as6983_safe_force_n",
            "data": f"{float(max_safe_force_n):.6g}",
        },
    )
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", {"name": component_name, "pos": "0 0 0"})
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": f"{float(mass_kg):.6g}",
            "diaginertia": "1e-4 1e-4 1e-4",
        },
    )
    for axis, k_val, c_val in zip(
        ("x", "y", "z"), stiffness_xyz_n_per_m, damping_xyz_n_s_per_m, strict=True
    ):
        axis_vec = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}[axis]
        ET.SubElement(
            body,
            "joint",
            {
                "name": f"{component_name}_flex_{axis}",
                "type": "slide",
                "axis": axis_vec,
                "stiffness": f"{float(k_val):.6g}",
                "damping": f"{float(c_val):.6g}",
            },
        )

    xml_path = out / f"{component_name}_compliance.xml"
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    xml_sha = hashlib.sha256(xml_path.read_bytes()).hexdigest()

    manifest: dict[str, Any] = {
        "schema": MUJOCO_COMPLIANCE_SCHEMA,
        "component_name": component_name,
        "mjcf_xml_file": xml_path.name,
        "mjcf_sha256": xml_sha,
        "stiffness_xyz_n_per_m": [float(v) for v in stiffness_xyz_n_per_m],
        "damping_xyz_n_s_per_m": [float(v) for v in damping_xyz_n_s_per_m],
        "mass_kg": float(mass_kg),
        "as6983_safe_force_n": float(max_safe_force_n),
        "step_geometry_ref": step_path,
        "bundle_sha256": bundle_sha256,
    }
    manifest["manifest_sha256"] = _sha256_doc(manifest)
    manifest_path = out / f"{component_name}_mujoco_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest


def export_ros2_node_manifest(
    output_dir: str | Path,
    *,
    node_name: str,
    control_rate_hz: float,
    wrench_topic: str = "/formamech/certified_wrench",
    safety_topic: str = "/formamech/conformal_cbf_margin",
    command_in_topic: str = "/controller/nominal_cmd",
    command_out_topic: str = "/controller/safe_cmd",
    bundle_sha256: str = "",
) -> dict[str, Any]:
    """Emit a canonical ROS 2 Humble/Jazzy safety-node interface manifest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": ROS2_CONTRACT_SCHEMA,
        "node_name": node_name,
        "control_rate_hz": float(control_rate_hz),
        "subscriptions": [
            {"topic": command_in_topic, "msg_type": "std_msgs/msg/Float64MultiArray"}
        ],
        "publications": [
            {"topic": command_out_topic, "msg_type": "std_msgs/msg/Float64MultiArray"},
            {"topic": wrench_topic, "msg_type": "geometry_msgs/msg/WrenchStamped"},
            {"topic": safety_topic, "msg_type": "std_msgs/msg/Float64"},
        ],
        "fail_closed_behavior": "engage_safe_hold_on_ood_or_deadline_miss",
        "bundle_sha256": bundle_sha256,
    }
    manifest["manifest_sha256"] = _sha256_doc(manifest)
    (out / f"{node_name}_ros2_manifest.json").write_bytes(
        canonical_json_bytes(manifest)
    )
    return manifest


__all__ = (
    "MUJOCO_COMPLIANCE_SCHEMA",
    "ROS2_CONTRACT_SCHEMA",
    "WRENCH_TRACE_SCHEMA",
    "SimulatorWrenchTrace",
    "export_mujoco_compliance_plugin",
    "export_ros2_node_manifest",
    "import_simulator_wrench_trace",
)
