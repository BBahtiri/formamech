"""FormaMech v0.6-v0.7 Robotics Physics AI package (pure-MIT).

Provides certified continuum tactile sensing, Port-Hamiltonian reduced dynamics,
sub-10 ms Condensed QP-MPC, AS6983 Static & Online Adaptive ACP-CQR Conformal
Control Barrier Function safety filters, infinite-dimensional Operator CLIK,
energy-residual contact jamming/slip diagnostics, live two-way simulator
coupling callbacks, and bidirectional simulator/middleware bridges.
"""

from __future__ import annotations

from piginot.robotics.adaptive_cbf import (
    ADAPTIVE_CBF_DECISION_SCHEMA,
    AdaptiveCbfResult,
    AdaptiveCbfState,
    AdaptiveConformalBarrierSpec,
    AdaptiveConformalCbfFilter,
    ConformalQuantileRegressor,
)
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
from piginot.robotics.clik_3d import (
    OPERATOR_CLIK_SCHEMA,
    OperatorClikConfig,
    OperatorClikController,
    OperatorClikStepResult,
    ShapeAndJacobianFn,
    SphereObstacle3D,
    make_cosserat_backbone_shape_model,
    make_surface_operator_shape_model,
)
from piginot.robotics.contact_diagnostics import (
    CONTACT_DIAGNOSTICS_SCHEMA,
    ContactDiagnosticResult,
    ContactDiagnosticsConfig,
    EnergyResidualContactEstimator,
    calibrate_threshold_for_max_fpr,
    compute_roc_auroc,
)
from piginot.robotics.live_coupling import (
    LIVE_COUPLING_SCHEMA,
    ContactGeomState,
    ContactResidualNet,
    CoupledRolloutComparison,
    CoupledStepOutput,
    LiveSimulatorCoupler,
    compare_oneway_vs_twoway_rollout,
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
    "ADAPTIVE_CBF_DECISION_SCHEMA",
    "CBF_DECISION_SCHEMA",
    "CONTACT_DIAGNOSTICS_SCHEMA",
    "CONTROL_SAFETY_MACHINE",
    "LIVE_COUPLING_SCHEMA",
    "MPC_STEP_SCHEMA",
    "MUJOCO_COMPLIANCE_SCHEMA",
    "OPERATOR_CLIK_SCHEMA",
    "PHNN_ARCHITECTURE_ID",
    "ROS2_CONTRACT_SCHEMA",
    "TACTILE_PRED_SCHEMA",
    "TACTILE_SPEC_SCHEMA",
    "WRENCH_TRACE_SCHEMA",
    "AdaptiveCbfResult",
    "AdaptiveCbfState",
    "AdaptiveConformalBarrierSpec",
    "AdaptiveConformalCbfFilter",
    "CbfFilterResult",
    "CondensedQpMpcConfig",
    "CondensedQpMpcController",
    "ConformalBarrierSpec",
    "ConformalCbfSafetyFilter",
    "ConformalQuantileRegressor",
    "ContactDiagnosticResult",
    "ContactDiagnosticsConfig",
    "ContactGeomState",
    "ContactResidualNet",
    "ControlSafetyState",
    "CoupledRolloutComparison",
    "CoupledStepOutput",
    "EnergyResidualContactEstimator",
    "LiveSimulatorCoupler",
    "MpcStepResult",
    "OperatorClikConfig",
    "OperatorClikController",
    "OperatorClikStepResult",
    "PortHamiltonianConfig",
    "PortHamiltonianSurrogate",
    "ShapeAndJacobianFn",
    "SimulatorWrenchTrace",
    "SphereObstacle3D",
    "TactileCertifiedPrediction",
    "TactileElastomerNet",
    "TactileElastomerSurrogate",
    "TactileProbeSample",
    "TactileSensorSpec",
    "calibrate_threshold_for_max_fpr",
    "compare_oneway_vs_twoway_rollout",
    "compute_roc_auroc",
    "export_mujoco_compliance_plugin",
    "export_ros2_node_manifest",
    "import_simulator_wrench_trace",
    "make_cosserat_backbone_shape_model",
    "make_surface_operator_shape_model",
    "project_contact_to_taxel_grid",
    "simulate_tactile_probe_fem",
)
