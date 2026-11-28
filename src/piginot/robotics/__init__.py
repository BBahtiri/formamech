"""FormaMech v0.6 Robotics Physics AI package (pure-MIT).

Provides certified continuum tactile sensing, Port-Hamiltonian reduced dynamics,
sub-10 ms Condensed QP-MPC, AS6983 Conformal Control Barrier Function safety
filters, and bidirectional simulator/middleware bridges.
"""

from __future__ import annotations

from piginot.robotics.bridges import (
    MUJOCO_COMPLIANCE_SCHEMA,
    ROS2_CONTRACT_SCHEMA,
    WRENCH_TRACE_SCHEMA,
    SimulatorWrenchTrace,
    export_mujoco_compliance_plugin,
    export_ros2_node_manifest,
    import_simulator_wrench_trace,
)
from piginot.robotics.cbf import (
    CBF_DECISION_SCHEMA,
    CONTROL_SAFETY_MACHINE,
    CbfFilterResult,
    ConformalBarrierSpec,
    ConformalCbfSafetyFilter,
    ControlSafetyState,
)
from piginot.robotics.mpc import (
    MPC_STEP_SCHEMA,
    CondensedQpMpcConfig,
    CondensedQpMpcController,
    MpcStepResult,
)
from piginot.robotics.port_hamiltonian import (
    PHNN_ARCHITECTURE_ID,
    PortHamiltonianConfig,
    PortHamiltonianSurrogate,
)
from piginot.robotics.tactile import (
    TACTILE_PRED_SCHEMA,
    TACTILE_SPEC_SCHEMA,
    TactileCertifiedPrediction,
    TactileElastomerNet,
    TactileElastomerSurrogate,
    TactileProbeSample,
    TactileSensorSpec,
    project_contact_to_taxel_grid,
    simulate_tactile_probe_fem,
)

__all__ = (
    "CBF_DECISION_SCHEMA",
    "CONTROL_SAFETY_MACHINE",
    "MPC_STEP_SCHEMA",
    "MUJOCO_COMPLIANCE_SCHEMA",
    "PHNN_ARCHITECTURE_ID",
    "ROS2_CONTRACT_SCHEMA",
    "TACTILE_PRED_SCHEMA",
    "TACTILE_SPEC_SCHEMA",
    "WRENCH_TRACE_SCHEMA",
    "CbfFilterResult",
    "CondensedQpMpcConfig",
    "CondensedQpMpcController",
    "ConformalBarrierSpec",
    "ConformalCbfSafetyFilter",
    "ControlSafetyState",
    "MpcStepResult",
    "PortHamiltonianConfig",
    "PortHamiltonianSurrogate",
    "SimulatorWrenchTrace",
    "TactileCertifiedPrediction",
    "TactileElastomerNet",
    "TactileElastomerSurrogate",
    "TactileProbeSample",
    "TactileSensorSpec",
    "export_mujoco_compliance_plugin",
    "export_ros2_node_manifest",
    "import_simulator_wrench_trace",
    "project_contact_to_taxel_grid",
    "simulate_tactile_probe_fem",
)
