"""Tests for V6-R3/R5/R6: Surface3D AB-UPT/Transolver++/LRQ scaling, TB operator & streams."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from piginot.ingest.robotics_streams import (
    EpisodeLeakageError,
    RoboticsEpisode,
    align_multirate_streams,
    split_episodes_leak_free,
)
from piginot.operators.surface_operator_3d import (
    Surface3DConfig,
    Surface3DEnergyOperator,
)
from piginot.operators.temporal_bundle_3d import (
    TemporalBundleConfig,
    TemporallyBundledOperator3D,
)


def test_surface3d_abupt_adaptive_temp_lrq_and_chunked_parity() -> None:
    """V6-R5 scaling options run cleanly, stay autograd-safe, and match chunked decoding."""
    torch.manual_seed(5)
    cfg = Surface3DConfig(
        parameter_dim=3,
        surface_feature_dim=6,
        hidden_dim=24,
        slice_count=6,
        num_heads=4,
        num_blocks=2,
        adaptive_temperature=True,
        anchor_count=32,
        low_rank_query_dim=12,
        decode_chunk_size=None,
    )
    op = Surface3DEnergyOperator(cfg)
    enc = op.encode_surface(
        torch.randn(1, 128, 3),
        torch.randn(1, 128, 3),
        torch.ones(1, 128),
        torch.randn(1, 128, 6),
    )
    assert enc.slice_tokens.shape == (1, 6, 24)
    qxyz = torch.randn(1, 100, 3)
    params = torch.randn(1, 3)

    u_full, g_full = op.displacement_and_gradient(
        encoding=enc, query_xyz=qxyz, parameters=params, create_graph=True
    )
    assert u_full.shape == (1, 100, 3)
    assert g_full.shape == (1, 100, 3, 3)
    g_full.square().mean().backward()

    # Chunked decoding must reproduce unchunked output bit-for-bit within float32 tol
    cfg_chunk = Surface3DConfig(
        parameter_dim=3,
        surface_feature_dim=6,
        hidden_dim=24,
        slice_count=6,
        num_heads=4,
        num_blocks=2,
        adaptive_temperature=True,
        anchor_count=32,
        low_rank_query_dim=12,
        decode_chunk_size=17,
    )
    op_chunk = Surface3DEnergyOperator(cfg_chunk)
    op_chunk.load_state_dict(op.state_dict())
    with torch.no_grad():
        u_ref = op.displacement(encoding=enc, query_xyz=qxyz, parameters=params)
        u_chk = op_chunk.displacement(encoding=enc, query_xyz=qxyz, parameters=params)
    assert float(torch.max(torch.abs(u_ref - u_chk)).item()) < 1.0e-6


def test_temporally_bundled_operator_and_stop_grad_isolation() -> None:
    """TB-Surface3DOperator outputs K bundled frames and honors stop_grad_residual."""
    torch.manual_seed(9)
    tb = TemporallyBundledOperator3D(
        TemporalBundleConfig(
            surface_config=Surface3DConfig(
                parameter_dim=2, surface_feature_dim=6, hidden_dim=16, slice_count=4
            ),
            bundle_steps=4,
            control_dim=3,
            dt_s=0.01,
        )
    )
    enc = tb.encode_surface(
        torch.randn(1, 10, 3),
        torch.randn(1, 10, 3),
        torch.ones(1, 10),
        torch.randn(1, 10, 6),
    )
    u_cur = torch.zeros((1, 8, 3), requires_grad=True)
    v_cur = torch.zeros((1, 8, 3), requires_grad=True)
    ctrl = torch.randn((1, 3), requires_grad=True)
    u_seq, v_seq = tb.predict_velocity_bundle(
        encoding=enc,
        query_xyz=torch.randn(1, 8, 3),
        current_u=u_cur,
        current_v=v_cur,
        control_u=ctrl,
        parameters=torch.randn(1, 2),
        stop_grad_residual=True,
    )
    assert u_seq.shape == (1, 4, 8, 3)
    assert v_seq.shape == (1, 4, 8, 3)
    u_seq.sum().backward()
    # With stop_grad_residual=True, control gradient through residual head is cleanly blocked
    assert ctrl.grad is None or float(ctrl.grad.abs().sum().item()) == 0.0


def test_robotics_streams_alignment_and_leakage_guard() -> None:
    """Multi-rate alignment resamples streams cleanly and blocks duplicate episode leaks."""
    t_mocap = np.linspace(0.0, 1.0, 101)
    t_ft = np.linspace(0.0, 1.0, 501)
    ep0 = align_multirate_streams(
        "ep0",
        {
            "mocap": (t_mocap, np.sin(2.0 * np.pi * t_mocap)[:, None]),
            "ft": (t_ft, np.cos(2.0 * np.pi * t_ft)[:, None]),
        },
        target_dt_s=0.01,
    )
    assert ep0.times_s.size == 101
    assert abs(ep0.sample_rate_hz - 100.0) < 1.0e-6

    ep1 = RoboticsEpisode("ep1", t_mocap, {"ft": np.ones((101, 1))}, {})
    ep2 = RoboticsEpisode("ep2", t_mocap, {"ft": 2.0 * np.ones((101, 1))}, {})
    ep3 = RoboticsEpisode("ep3", t_mocap, {"ft": 3.0 * np.ones((101, 1))}, {})
    part = split_episodes_leak_free([ep1, ep2, ep3], seed=0)
    assert len(part.train) + len(part.dev) + len(part.holdout) == 3

    # Duplicate content hash under a new episode_id must fail closed
    ep_dup = RoboticsEpisode("ep_copy", t_mocap, {"ft": np.ones((101, 1))}, {})
    with pytest.raises(EpisodeLeakageError):
        split_episodes_leak_free([ep1, ep2, ep_dup])
