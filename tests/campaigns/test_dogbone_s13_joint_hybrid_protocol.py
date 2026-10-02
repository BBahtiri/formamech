from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from typing import Any, cast

import pytest
import rfc8785

from campaigns.dogbone_v1 import s13_joint_hybrid_protocol as protocol
from piginot.contracts import canonical_json_bytes, strict_json_loads
from piginot.doe import EvidenceRole

ROOT = Path(__file__).resolve().parents[2]


def test_proposal_is_fixed_unauthorized_train_only_and_canonical() -> None:
    proposal = protocol.build_joint_hybrid_proposal(ROOT)
    assert proposal.status == "DRAFT_REQUIRES_EXACT_APPROVAL"
    assert proposal.execution_authorized is False
    assert proposal.training_performed is False
    assert proposal.role is EvidenceRole.TRAIN
    assert proposal.non_gate is True
    assert proposal.g13_status == "NOT_PASSED"
    assert proposal.cp_freeze_status == "WITHHELD"
    assert proposal.s14_status == "BLOCKED"
    assert proposal.plan.role is EvidenceRole.TRAIN
    assert proposal.plan.non_gate is True
    assert proposal.plan.case_id == "candidate-002485"
    assert canonical_json_bytes(proposal) == canonical_json_bytes(
        protocol.build_joint_hybrid_proposal(ROOT)
    )
    assert len(proposal.content_hash) == 64
    assert len(proposal.plan.content_hash) == 64
    with pytest.raises(FrozenInstanceError):
        proposal.execution_authorized = True  # type: ignore[misc]
    for changes in (
        {"execution_authorized": True},
        {"training_performed": True},
        {"role": EvidenceRole.DEVELOPMENT},
        {"non_gate": False},
        {"status": "APPROVED"},
        {"g13_status": "PASS"},
        {"cp_freeze_status": "FROZEN"},
        {"s14_status": "READY"},
    ):
        with pytest.raises(ValueError, match="fixed draft"):
            replace(proposal, **changes)


def test_arms_share_budget_sampler_architecture_and_joint_scope() -> None:
    plan = protocol.JointHybridPlan()
    assert plan.architecture_policy == "exact-v5-baseline-both-arms"
    assert plan.trainable_policy == "all-shared-u-head-p-head-parameters"
    assert plan.warmstart_optimizer_steps == 4000
    assert plan.arm_optimizer_steps == 4000
    assert plan.total_optimizer_updates == 12000
    assert plan.vertex_batch_size == 57
    assert plan.expected_vertices == 969
    assert plan.learning_rate == 2e-5
    assert plan.seed == 13001
    assert plan.evaluation_steps == (0, 1000, 2000, 3000, 4000)
    assert plan.comparison_endpoint == 4000
    assert plan.physics_collocation_layout == (128, 64, 64)
    assert plan.gradient_clip_norm == 1.0
    assert plan.adaptive_weighting is False
    assert plan.scheduler is False
    assert plan.control_term_weights == (("data_u", 1.0), ("data_P", 1.0))
    assert plan.candidate_term_weights == plan.control_term_weights + (
        ("equilibrium_strong", 1e-3),
        ("constitutive", 1e-3),
        ("angular", 1e-3),
        ("dirichlet", 1e-3),
        ("traction_free", 1e-3),
    )
    assert plan.numeric_repair_threshold is None
    assert plan.checkpoint_policy == "none-no-promotion-no-best-step"
    assert plan.non_train_access is False
    assert plan.new_fem_solves == 0


@pytest.mark.parametrize(
    "changes",
    (
        {"role": EvidenceRole.DEVELOPMENT},
        {"non_gate": False},
        {"case_id": "other"},
        {"learning_rate": 1e-4},
        {"arm_optimizer_steps": 4001},
        {"cpu_num_threads": 1},
        {"gradient_clip_norm": True},
        {"physics_collocation_layout": (256, 0, 0)},
        {"evaluation_steps": (0, True, 2000, 3000, 4000)},
        {"candidate_term_weights": (("data_u", 1.0),)},
        {"control_term_weights": (("data_u", True), ("data_P", 1.0))},
        {"numeric_repair_threshold": 0.5},
        {"checkpoint_policy": "save-best"},
    ),
)
def test_plan_refuses_drift_including_nested_bool_numeric_aliases(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="fixed draft"):
        replace(protocol.JointHybridPlan(), **cast(Any, changes))


def test_all_contract_fields_have_fixed_defaults() -> None:
    for contract in (protocol.JointHybridPlan(), protocol.JointHybridProposal()):
        for item in fields(contract):
            assert getattr(contract, item.name) == item.default


def test_builder_reads_only_two_public_evidence_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.read_bytes
    reads: list[Path] = []

    def tracked(path: Path) -> bytes:
        reads.append(path)
        assert path.parent == ROOT / "artifacts/campaigns/dogbone-v1"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    protocol.build_joint_hybrid_proposal(ROOT)
    assert [path.name for path in reads] == [
        "S13-OVERFIT-BUDGET-V5.json",
        "S13-QUADRATURE-GRADIENT-COMPARISON.json",
    ]


def test_builder_fails_closed_on_missing_or_modified_public_evidence(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        protocol.build_joint_hybrid_proposal(tmp_path)
    public = tmp_path / "artifacts/campaigns/dogbone-v1"
    public.mkdir(parents=True)
    (public / "S13-OVERFIT-BUDGET-V5.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="SHA-256"):
        protocol.build_joint_hybrid_proposal(tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        protocol.build_joint_hybrid_proposal(Path("relative"))


def test_writer_is_exclusive_canonical_and_never_trains(tmp_path: Path) -> None:
    output = tmp_path / "proposal.json"
    assert protocol.main(["--output", str(output)]) == 0
    encoded = output.read_bytes()
    proposal = protocol.build_joint_hybrid_proposal(ROOT)
    assert encoded == canonical_json_bytes(proposal) + b"\n"
    parsed = strict_json_loads(encoded)
    assert type(parsed) is dict
    parsed_object = cast(dict[str, Any], parsed)
    assert parsed_object["execution_authorized"] is False
    assert parsed_object["plan"]["non_train_access"] is False
    assert len(hashlib.sha256(encoded).hexdigest()) == 64
    with pytest.raises(FileExistsError):
        protocol.main(["--output", str(output)])
    assert output.read_bytes() == encoded
    with pytest.raises(ValueError, match="absolute"):
        protocol.main(["--output", "relative.json"])
    with pytest.raises(FileNotFoundError):
        protocol.main(["--output", str(tmp_path / "missing/proposal.json")])


def test_saved_proposal_matches_fixed_contract() -> None:
    saved = ROOT / "artifacts/campaigns/dogbone-v1/S13-JOINT-HYBRID-PROPOSAL.json"
    assert (
        saved.read_bytes()
        == canonical_json_bytes(protocol.build_joint_hybrid_proposal(ROOT)) + b"\n"
    )


@pytest.mark.parametrize("encoded", (b"[]\n", b'{ "a": 1 }\n'))
def test_public_reader_rejects_non_object_or_noncanonical_bytes(
    tmp_path: Path,
    encoded: bytes,
) -> None:
    path = tmp_path / "public.json"
    path.write_bytes(encoded)
    with pytest.raises(ValueError, match="canonical JSON"):
        protocol._read_public_evidence(
            path, hashlib.sha256(encoded).hexdigest(), "expected"
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"$type": "wrong"},
        {"role": "DEVELOPMENT"},
        {"non_gate": False},
        {"non_gate": 1},
        {"case_id": "other"},
    ),
)
def test_public_reader_rejects_type_role_or_case_drift(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    record = {
        "$type": "expected",
        "role": "TRAIN",
        "non_gate": True,
        "case_id": "candidate-002485",
        **changes,
    }
    encoded = rfc8785.dumps(cast(Any, record)) + b"\n"
    path = tmp_path / "public.json"
    path.write_bytes(encoded)
    with pytest.raises(ValueError, match="binding"):
        protocol._read_public_evidence(
            path, hashlib.sha256(encoded).hexdigest(), "expected"
        )


def test_builder_rejects_inconsistent_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = protocol._read_public_evidence

    def inconsistent(path: Path, digest: str, expected_type: str) -> dict[str, object]:
        record = reader(path, digest, expected_type)
        return {**record, "source_bindings_hash": "wrong"}

    monkeypatch.setattr(protocol, "_read_public_evidence", inconsistent)
    with pytest.raises(ValueError, match="lineage"):
        protocol.build_joint_hybrid_proposal(ROOT)


def test_evidence_and_output_refuse_symlink_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == tmp_path)
    with pytest.raises(ValueError, match="symlinks"):
        protocol.build_joint_hybrid_proposal(tmp_path)
    with pytest.raises(ValueError, match="symlinks"):
        protocol.main(["--output", str(tmp_path / "draft.json")])


@pytest.mark.parametrize("modified", (False, True))
def test_builder_requires_the_second_pinned_public_report(
    tmp_path: Path,
    modified: bool,
) -> None:
    public = tmp_path / "artifacts/campaigns/dogbone-v1"
    public.mkdir(parents=True)
    filename = "S13-OVERFIT-BUDGET-V5.json"
    (public / filename).write_bytes(
        (ROOT / "artifacts/campaigns/dogbone-v1" / filename).read_bytes()
    )
    if modified:
        (public / "S13-QUADRATURE-GRADIENT-COMPARISON.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError if modified else FileNotFoundError):
        protocol.build_joint_hybrid_proposal(tmp_path)


@pytest.mark.parametrize("flag", ("--execute", "--approve"))
def test_cli_has_no_execution_or_approval_path(tmp_path: Path, flag: str) -> None:
    output = tmp_path / "draft.json"
    with pytest.raises(SystemExit) as failure:
        protocol.main(["--output", str(output), flag])
    assert failure.value.code == 2
    assert not output.exists()
