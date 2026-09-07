from __future__ import annotations

from typing import Any, cast

from campaigns.dogbone_v1.family import compiled_family
from campaigns.dogbone_v1.s11_evidence import (
    DOGBONE_S11_PARAMETER_NAMES,
    build_dogbone_s11_operator,
    evidence_payload,
    run_dogbone_s11_evidence,
)
from piginot.operators import ARCHITECTURE_ID, BoundaryConditionMode, GateStatus


def test_dogbone_operator_retains_upstream_pi_ginot_shape_contract() -> None:
    model = build_dogbone_s11_operator()

    assert model.architecture_id == ARCHITECTURE_ID
    assert model.config.parameter_names == DOGBONE_S11_PARAMETER_NAMES
    assert model.config.boundary_tag_names == tuple(
        compiled_family().reference_mesh.semantic_tags
    )
    assert model.config.hidden_dim == 64
    assert model.config.latent_token_count == 32
    assert model.config.num_attention_heads == 4
    assert model.config.num_fourier_frequencies == 6
    assert model.config.encoder_self_attention_layers == 3
    assert model.config.decoder_cross_attention_layers == 6
    assert model.config.boundary_condition_mode is BoundaryConditionMode.SOFT


def test_actual_dogbone_reference_geometry_passes_g11_measurements() -> None:
    evidence = run_dogbone_s11_evidence(code_coverage_percent=90.0)
    measurements = evidence.measurements

    assert evidence.evaluation.verdict is GateStatus.PASS
    assert measurements.permutation_count == 100
    assert measurements.permutation_max_relative_change is not None
    assert measurements.permutation_max_relative_change < 1.0e-5
    assert (measurements.resampling_low_count, measurements.resampling_high_count) == (
        320,
        480,
    )
    assert measurements.resampling_relative_change is not None
    assert measurements.resampling_relative_change < 0.02
    assert measurements.node_order_relative_change is not None
    assert measurements.node_order_relative_change < 1.0e-6
    assert measurements.query_batch_relative_change is not None
    assert measurements.query_batch_relative_change < 1.0e-6
    assert measurements.reachable_parameter_count == len(DOGBONE_S11_PARAMETER_NAMES)
    assert measurements.mask_mutant_fired is True
    assert measurements.pickle_refused_before_deserialization is True
    assert measurements.u_shape_verified is True
    assert measurements.p_shape_verified is True
    assert measurements.served_stress_is_predicted is True


def test_dogbone_s11_evidence_is_deterministic_and_serializable() -> None:
    first = run_dogbone_s11_evidence(code_coverage_percent=90.0)
    second = run_dogbone_s11_evidence(code_coverage_percent=90.0)

    assert first == second
    payload = cast(dict[str, Any], evidence_payload(first))
    assert payload["architecture_id"] == ARCHITECTURE_ID
    assert payload["reference_mesh_id"] == compiled_family().reference_mesh.mesh_id
    assert payload["gate"]["verdict"] == "PASS"
    assert len(payload["gate"]["checks"]) == 9
    assert payload["diagnostics"]["mask_mutant_relative_change"] > 1.0e-3
    assert payload["diagnostics"]["mask_control_relative_change"] < 1.0e-10
