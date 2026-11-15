"""Installed, FIT-only two-arm pilot; never solves FEM or starts a full run."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save

from piginot.compiler.boundary_objective import BoundaryPilotObjective
from piginot.compiler.boundary_pilot_inputs import (
    document,
    load_inputs,
    prepare_objective,
    sha,
)
from piginot.compiler.boundary_pilot_inputs import (
    validate_pilot_terms as validate_pilot_terms,
)
from piginot.compiler.energy_training import EnergyTraining
from piginot.compiler.execution_records import digest
from piginot.compiler.records import atomic_write
from piginot.contracts import canonical_json_bytes
from piginot.fem import ScikitFemBackend
from piginot.training._checkpoint_codec import _canonical_safetensors
from piginot.training.checkpoint import model_state_hash
from piginot.training.lbfgs_stage import (
    GradientEvaluation,
    LBFGSConfig,
    StageDeadline,
    train_lbfgs_stage,
)


def _write(path: Path, value: dict[str, Any]) -> None:
    atomic_write(path, canonical_json_bytes(value) + b"\n")


def assess_pilot(arms: dict[str, Any], baseline: dict[str, Any]) -> str:
    """Apply only the prospective FIT-only engineering selection rule."""
    if set(arms) != {"A", "B"}:
        return "NO_SCALE_UP"
    if any(
        arm["stage"]["status"] not in ("COMPLETE", "CONVERGED") for arm in arms.values()
    ):
        return "NO_SCALE_UP"
    rows = [baseline, *(arm["validation"] for arm in arms.values())]
    if any(
        row["admissible"] is not True
        or not all(
            math.isfinite(row[key]) and row[key] >= 0
            for key in ("bulk_energy", "traction_squared")
        )
        for row in rows
    ):
        return "NO_SCALE_UP"
    a, b = arms["A"]["validation"], arms["B"]["validation"]
    if (
        b["traction_squared"] <= 0.25 * a["traction_squared"]
        and b["bulk_energy"] <= 1.005 * a["bulk_energy"]
    ):
        return "B_SUPPORTED"
    if (
        a["traction_squared"] <= 0.25 * baseline["traction_squared"]
        and a["bulk_energy"] <= 1.005 * baseline["bulk_energy"]
    ):
        return "A_SUPPORTED"
    return "NO_SCALE_UP"


def execute_arms(
    data: BoundaryPilotObjective,
    root: Path,
    *,
    config: LBFGSConfig,
    binding: str,
    deadline: float | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    max_validation_evaluations: int = 20,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    if any((root / name).exists() for name in ("A", "B")):
        raise FileExistsError(
            "pilot arms require fresh output; automatic resume is disabled"
        )
    arms: dict[str, Any] = {}
    validations = 0

    def publish(event: dict[str, Any]) -> None:
        payload = {
            "new_fem_solves": 0,
            "validation_evaluations": validations,
            "completed_arms": list(arms),
            **event,
        }
        if progress is not None:
            progress(payload)

    def validate(model: torch.nn.Module) -> dict[str, Any]:
        nonlocal validations
        if validations >= max_validation_evaluations:
            raise ValueError("independent validation evaluation limit reached")
        validations += 1
        publish({"event": "VALIDATION_RESERVED"})
        metrics = data.evaluate(
            model,
            traction_weight=0.0,
            gradients=False,
            validation=True,
            deadline=deadline,
        )
        return asdict(metrics)

    baseline = validate(data.new_model())
    if baseline["admissible"] is not True:
        raise ValueError("the imported initializer violates the admissibility guard")
    _write(root / "baseline.json", baseline)
    frozen_document = data.document()
    for name, weight in (("A", 0.0), ("B", 1.0)):
        if deadline is not None and time.monotonic() >= deadline:
            raise StageDeadline("pilot wall-clock allowance ended before the next arm")
        if data.document() != frozen_document:
            raise ValueError("shared pilot quadrature or normalization changed")
        model = data.new_model()
        initial_hash = model_state_hash(model)

        def objective(
            model: torch.nn.Module, weight: float = weight
        ) -> GradientEvaluation:
            observed = data.evaluate(
                model, traction_weight=weight, gradients=True, deadline=deadline
            )
            return GradientEvaluation(observed.value, observed.admissible)

        def on_progress(event: dict[str, Any], name: str = name) -> None:
            _write(root / f"progress-{name}.json", event)
            publish({"arm": name, **event})

        result = train_lbfgs_stage(
            root / name,
            model=model,
            objective=objective,
            binding=digest(
                {"pilot_binding": binding, "arm": name, "traction_weight": weight}
            ),
            config=config,
            deadline=deadline,
            progress=on_progress,
        )
        validation = validate(model)
        weights = _canonical_safetensors(
            save(
                {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
            )
        )
        atomic_write(root / name / "model.safetensors", weights)
        entry = {
            "initial_model_hash": initial_hash,
            "stage": {**asdict(result), "checkpoint": str(result.checkpoint)},
            "validation": validation,
            "weights_sha256": sha(root / name / "model.safetensors"),
            "checkpoint_manifest_sha256": sha(result.checkpoint / "checkpoint.json"),
            "binding": binding,
            "claim_level": "NOT_VERIFIED",
        }
        arms[name] = entry
        _write(root / f"outcome-{name}.json", entry)
        publish({"event": "ARM_FINISHED", "arm": name, "status": result.status})
    if (
        arms["A"]["initial_model_hash"] != arms["B"]["initial_model_hash"]
        or data.document() != frozen_document
    ):
        raise ValueError(
            "pilot arms did not share their frozen initialization and samples"
        )
    assessment = {
        "arms": arms,
        "baseline": baseline,
        "engineering_selection": assess_pilot(arms, baseline),
        "new_fem_solves": 0,
        "optimizer_iterations": sum(v["stage"]["iterations"] for v in arms.values()),
        "gradient_evaluations": sum(v["stage"]["evaluations"] for v in arms.values()),
        "validation_evaluations": validations,
        "claim_level": "NOT_VERIFIED",
        "overall_g29_status": "NOT_RUN",
        "follow_on_authorized": False,
    }
    _write(root / "assessment.json", assessment)
    return assessment


def _forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("the boundary pilot cannot run FEM or Adam training")


def _output(path: Path, expected: Path, initializer: dict[str, Any]) -> None:
    protected = Path(initializer["training"]["checkpoint_path"]).parents[1]
    target = path.resolve()
    if (
        target != expected.resolve()
        or path.is_symlink()
        or path.exists()
        or target.is_relative_to(protected)
        or protected.is_relative_to(target)
    ):
        raise ValueError(
            "pilot output must be the frozen fresh path, outside the original run"
        )


def execution_deadline(
    output: Path, protocol_sha: str, deadline_utc: float, wall_seconds: int
) -> float:
    """Bind the worker to its coordinator's original clock; never renew a run."""
    try:
        lease = document(output.parent / "job.json")
        if (
            lease.get("schema") != "piginot.boundary-pilot-job-v1"
            or lease.get("status") != "RUNNING"
            or lease.get("protocol_sha256") != protocol_sha
        ):
            raise ValueError("worker requires its running coordinator lease")
        if (
            not math.isfinite(deadline_utc)
            or deadline_utc != lease.get("deadline_epoch")
            or abs(deadline_utc - lease["started_epoch"] - wall_seconds) > 1e-6
        ):
            raise ValueError("worker deadline differs from its original allowance")
        remaining = deadline_utc - time.time()
        if remaining > wall_seconds + 2 or remaining <= 0:
            raise ValueError("pilot allowance has expired or was extended")
        return time.monotonic() + min(remaining, float(wall_seconds))
    except (OSError, KeyError, TypeError) as error:
        raise ValueError(
            "worker requires a valid coordinator wall-clock lease"
        ) from error


def run_pilot(
    protocol_path: Path, output: Path, *, deadline_utc: float
) -> dict[str, Any]:
    protocol, proposal, initializer = load_inputs(protocol_path)
    _output(output, Path(protocol["execution_output"]) / "pilot", initializer)
    protocol_sha = sha(protocol_path)
    deadline = execution_deadline(
        output,
        protocol_sha,
        deadline_utc,
        proposal["budgets"]["maximum_wall_seconds_total"],
    )
    output.mkdir(parents=True, exist_ok=False)
    _write(
        output / "job.json",
        {
            "status": "RUNNING",
            "phase": "PREPARING",
            "protocol_sha256": protocol_sha,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "new_fem_solves": 0,
        },
    )
    guarded = (
        (ScikitFemBackend, "solve_many", ScikitFemBackend.solve_many),
        (ScikitFemBackend, "solve_from", ScikitFemBackend.solve_from),
        (torch.optim.Adam, "step", torch.optim.Adam.step),
        (EnergyTraining, "train", EnergyTraining.train),
    )
    previous_profile = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
    )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for owner, name, _ in guarded:
        setattr(owner, name, _forbidden)

    def progress(event: dict[str, Any]) -> None:
        _write(output / "progress.json", {"protocol_sha256": protocol_sha, **event})
        if event["event"] in ("ACCEPTED_ITERATION", "ARM_FINISHED"):
            print(
                f"arm={event['arm']} event={event['event']} iterations={event.get('iterations')} evaluations={event.get('evaluations')}",
                flush=True,
            )

    try:
        data, prepared = prepare_objective(protocol, proposal, initializer, output)
        _write(output / "preparation.json", prepared)
        tensors = {
            f"{case.case_id}/{name}": tensor.clone()
            for case in data.cases
            for name, tensor in data.case_tensors(case).items()
        }
        atomic_write(
            output / "quadrature.safetensors", _canonical_safetensors(save(tensors))
        )
        binding = digest(
            {
                "protocol_sha256": protocol_sha,
                "preparation_sha256": sha(output / "preparation.json"),
                "quadrature_sha256": sha(output / "quadrature.safetensors"),
            }
        )
        result = execute_arms(
            data,
            output,
            config=LBFGSConfig(),
            binding=binding,
            deadline=deadline,
            progress=progress,
        )
        if (
            model_state_hash(data.context.model)
            != initializer["training"]["model_state_hash"]
        ):
            raise ValueError("pilot modified its imported initializer")
        for name, expected in initializer["consumed_files_sha256"].items():
            if sha(Path(name)) != expected:
                raise ValueError("an original frozen source changed during the pilot")
        complete = all(
            arm["stage"]["status"] in ("COMPLETE", "CONVERGED")
            for arm in result["arms"].values()
        )
        _write(
            output / "job.json",
            {
                "status": "COMPLETE" if complete else "INCOMPLETE",
                "protocol_sha256": protocol_sha,
                "completed_at_utc": datetime.now(UTC).isoformat(),
                **{
                    key: result[key]
                    for key in (
                        "engineering_selection",
                        "optimizer_iterations",
                        "gradient_evaluations",
                        "validation_evaluations",
                        "new_fem_solves",
                        "claim_level",
                        "overall_g29_status",
                        "follow_on_authorized",
                    )
                },
            },
        )
        return result
    except BaseException as error:
        status = (
            "INTERRUPTED"
            if isinstance(error, KeyboardInterrupt)
            else "TIME_LIMIT"
            if isinstance(error, StageDeadline)
            else "ERROR"
        )
        _write(
            output / "job.json",
            {
                "status": status,
                "protocol_sha256": protocol_sha,
                "new_fem_solves": 0,
                "error": f"{type(error).__name__}: {error}",
                "claim_level": "NOT_VERIFIED",
                "overall_g29_status": "NOT_RUN",
            },
        )
        raise
    finally:
        for owner, name, previous in guarded:
            setattr(owner, name, previous)
        torch.set_num_threads(previous_profile[0])
        torch.use_deterministic_algorithms(previous_profile[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--deadline-utc", required=True, type=float)
    arguments = parser.parse_args()
    result = run_pilot(
        arguments.protocol, arguments.output, deadline_utc=arguments.deadline_utc
    )
    return (
        0
        if all(
            arm["stage"]["status"] in ("COMPLETE", "CONVERGED")
            for arm in result["arms"].values()
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
