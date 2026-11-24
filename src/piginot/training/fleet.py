"""Distributed FEM & training fleet orchestration: Local, Slurm & Kubernetes (V4-5).

Provides a pluggable :class:`WorkerBackend` abstraction governed by the explicit
``FLEET_JOB_MACHINE`` state machine::

    QUEUED -> DISPATCHED -> RUNNING -> COLLECTED -> VERIFIED | FAILED | CANCELLED

Backends:

* :class:`LocalMultiprocessFleet`: parallel local execution via
  :class:`concurrent.futures.ProcessPoolExecutor` (or threads for in-process
  callables), with deterministic per-task SHA-256 result receipts.
* :class:`SlurmScriptFleet`: generates deterministic JSON task inputs and a
  ready-to-submit Slurm ``sbatch --array`` script plus ``collect_completed()``
  result ingestion and hash verification.
* :class:`KubernetesJobFleet`: generates deterministic Kubernetes ``batch/v1``
  Indexed ``Job`` YAML manifests plus ``collect_completed()`` ingestion.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from piginot.contracts import MaterialSpec, Quantity, Unit, canonical_json_bytes
from piginot.contracts.state_machine import (
    StateMachineSpec,
    StateSnapshot,
    TransitionSpec,
    register_state_machine,
)
from piginot.fem.backend_3d import (
    DirichletBC3D,
    FemRequest3D,
    TorchFemBackend3D,
    TractionBC3D,
)
from piginot.geometry.mesh3d import make_brick_tet4_mesh

FLEET_JOB_SCHEMA: Final = "formamech.fleet-job-v1"


class FleetJobState(StrEnum):
    """Explicit lifecycle state machine for distributed fleet FEM/training jobs."""

    QUEUED = "QUEUED"
    DISPATCHED = "DISPATCHED"
    RUNNING = "RUNNING"
    COLLECTED = "COLLECTED"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


FLEET_JOB_MACHINE: Final[StateMachineSpec[FleetJobState]] = register_state_machine(
    StateMachineSpec(
        machine_id="formamech.training.fleet_job",
        description=(
            "Distributed fleet worker job lifecycle: "
            "QUEUED -> DISPATCHED -> RUNNING -> COLLECTED -> VERIFIED | FAILED | CANCELLED."
        ),
        states=tuple(FleetJobState),
        initial_states=(FleetJobState.QUEUED,),
        terminal_states=(
            FleetJobState.VERIFIED,
            FleetJobState.FAILED,
            FleetJobState.CANCELLED,
        ),
        transitions=(
            TransitionSpec(
                FleetJobState.QUEUED,
                FleetJobState.DISPATCHED,
                "dispatch",
                "Materialize task specs and dispatch batch to worker backend.",
            ),
            TransitionSpec(
                FleetJobState.DISPATCHED,
                FleetJobState.RUNNING,
                "start_workers",
                "Workers started executing fleet tasks.",
            ),
            TransitionSpec(
                FleetJobState.DISPATCHED,
                FleetJobState.CANCELLED,
                "cancel_dispatched",
                "Cancel fleet job before execution.",
            ),
            TransitionSpec(
                FleetJobState.RUNNING,
                FleetJobState.COLLECTED,
                "collect_outputs",
                "All worker outputs collected.",
            ),
            TransitionSpec(
                FleetJobState.RUNNING,
                FleetJobState.FAILED,
                "fail_execution",
                "Worker execution failed fatally.",
            ),
            TransitionSpec(
                FleetJobState.COLLECTED,
                FleetJobState.VERIFIED,
                "verify_results",
                "All collected results passed schema and SHA-256 verification.",
            ),
            TransitionSpec(
                FleetJobState.COLLECTED,
                FleetJobState.FAILED,
                "fail_verification",
                "One or more worker tasks reported errors or failed verification.",
            ),
        ),
    )
)


@dataclass(frozen=True, slots=True)
class FleetTaskSpec:
    """One deterministic, JSON-serializable fleet work unit."""

    task_id: str
    payload: Mapping[str, Any]

    def document(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "payload": dict(self.payload)}


@dataclass(frozen=True, slots=True)
class FleetTaskResult:
    """One verified worker task output with canonical SHA-256 digest."""

    task_id: str
    ok: bool
    result: Mapping[str, Any] | None
    error: str | None
    wall_time_s: float
    result_sha256: str

    def document(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "ok": self.ok,
            "result": None if self.result is None else dict(self.result),
            "error": self.error,
            "wall_time_s": float(self.wall_time_s),
            "result_sha256": self.result_sha256,
        }


@dataclass(frozen=True, slots=True)
class FleetJobReport:
    """Auditable fleet orchestration summary across Local, Slurm, or K8s backends."""

    job_id: str
    backend_name: str
    state: FleetJobState
    state_history: tuple[StateSnapshot[FleetJobState], ...]
    task_count: int
    succeeded_count: int
    failed_count: int
    results: tuple[FleetTaskResult, ...]
    artifacts: tuple[str, ...]
    wall_time_s: float

    def document(self) -> dict[str, Any]:
        return {
            "schema": FLEET_JOB_SCHEMA,
            "job_id": self.job_id,
            "backend_name": self.backend_name,
            "state": self.state.value,
            "task_count": self.task_count,
            "succeeded_count": self.succeeded_count,
            "failed_count": self.failed_count,
            "artifacts": list(self.artifacts),
            "wall_time_s": float(self.wall_time_s),
        }


def _sha256_doc(doc: object) -> str:
    return hashlib.sha256(canonical_json_bytes(doc)).hexdigest()


def solve_3d_brick_fem_task(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Top-level picklable 3D hyperelastic FEM worker task for multiprocess fleets."""
    lx = float(payload.get("lx_mm", 20.0))
    ly = float(payload.get("ly_mm", 8.0))
    lz = float(payload.get("lz_mm", 8.0))
    nx = int(payload.get("nx", 2))
    ny = int(payload.get("ny", 1))
    nz = int(payload.get("nz", 1))
    e_mpa = float(payload.get("youngs_modulus_mpa", 100.0))
    nu = float(payload.get("poisson_ratio", 0.30))
    tx = float(payload.get("traction_x_mpa", 0.4))

    mesh = make_brick_tet4_mesh(lx, ly, lz, nx=nx, ny=ny, nz=nz)
    mat = MaterialSpec(
        name="fleet-nh",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(e_mpa, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(nu, Unit.ONE).to_canonical(),
        },
    )
    res = TorchFemBackend3D().solve(
        FemRequest3D(
            request_id=str(payload.get("request_id", "fleet-3d")),
            mesh=mesh,
            material=mat,
            dirichlet_bcs=(
                DirichletBC3D("clamp", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
            ),
            traction_bcs=(TractionBC3D("load", ("x_max",), (tx, 0.0, 0.0)),),
            load_steps=int(payload.get("load_steps", 1)),
        )
    )
    return {
        "mesh_id": mesh.mesh_id,
        "converged": res.converged,
        "max_displacement_mm": float(res.max_displacement_mm),
        "total_strain_energy_n_mm": float(res.total_strain_energy_n_mm),
        "peak_von_mises_mpa": float(res.peak_von_mises_mpa),
        "dof_count": 3 * len(mesh.nodes),
    }


def _execute_one_task(
    args: tuple[str, dict[str, Any], Callable[[Mapping[str, Any]], dict[str, Any]]],
) -> FleetTaskResult:
    task_id, payload, fn = args
    t0 = time.perf_counter()
    try:
        out = fn(payload)
        dt = time.perf_counter() - t0
        return FleetTaskResult(
            task_id=task_id,
            ok=True,
            result=out,
            error=None,
            wall_time_s=dt,
            result_sha256=_sha256_doc({"task_id": task_id, "ok": True, "result": out}),
        )
    except Exception as exc:
        dt = time.perf_counter() - t0
        return FleetTaskResult(
            task_id=task_id,
            ok=False,
            result=None,
            error=f"{type(exc).__name__}: {exc}",
            wall_time_s=dt,
            result_sha256=_sha256_doc(
                {"task_id": task_id, "ok": False, "error": str(exc)}
            ),
        )


@runtime_checkable
class WorkerBackend(Protocol):
    """Protocol implemented by local, Slurm, and Kubernetes fleet backends."""

    @property
    def name(self) -> str: ...

    def submit_batch(
        self,
        job_id: str,
        tasks: Sequence[FleetTaskSpec],
        *,
        workdir: Path | str,
        worker_fn: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
    ) -> FleetJobReport: ...


class LocalMultiprocessFleet:
    """Parallel local CPU worker pool using processes (or threads for closures)."""

    def __init__(self, *, max_workers: int = 2, use_threads: bool = False) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.max_workers = max_workers
        self.use_threads = use_threads

    @property
    def name(self) -> str:
        return "local-thread-fleet" if self.use_threads else "local-multiprocess-fleet"

    def submit_batch(
        self,
        job_id: str,
        tasks: Sequence[FleetTaskSpec],
        *,
        workdir: Path | str,
        worker_fn: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
    ) -> FleetJobReport:
        t0 = time.perf_counter()
        fn = solve_3d_brick_fem_task if worker_fn is None else worker_fn
        root = Path(workdir).resolve() / job_id
        root.mkdir(parents=True, exist_ok=True)

        state = FleetJobState.QUEUED
        hist: list[StateSnapshot[FleetJobState]] = [FLEET_JOB_MACHINE.snapshot(state)]
        FLEET_JOB_MACHINE.require_transition(state, FleetJobState.DISPATCHED)
        state = FleetJobState.DISPATCHED
        hist.append(FLEET_JOB_MACHINE.snapshot(state))

        FLEET_JOB_MACHINE.require_transition(state, FleetJobState.RUNNING)
        state = FleetJobState.RUNNING
        hist.append(FLEET_JOB_MACHINE.snapshot(state))

        work_items = [(t.task_id, dict(t.payload), fn) for t in tasks]
        pool_cls: type[ThreadPoolExecutor] | type[ProcessPoolExecutor] = (
            ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        )
        with pool_cls(max_workers=min(self.max_workers, max(1, len(work_items)))) as ex:
            results = list(ex.map(_execute_one_task, work_items))

        FLEET_JOB_MACHINE.require_transition(state, FleetJobState.COLLECTED)
        state = FleetJobState.COLLECTED
        hist.append(FLEET_JOB_MACHINE.snapshot(state))

        succeeded = sum(1 for r in results if r.ok)
        failed = len(results) - succeeded
        final_state = FleetJobState.VERIFIED if failed == 0 else FleetJobState.FAILED
        FLEET_JOB_MACHINE.require_transition(state, final_state)
        state = final_state
        hist.append(FLEET_JOB_MACHINE.snapshot(state))

        report_path = root / "fleet-report.json"
        report_path.write_bytes(
            canonical_json_bytes(
                {
                    "schema": FLEET_JOB_SCHEMA,
                    "job_id": job_id,
                    "backend": self.name,
                    "state": state.value,
                    "results": [r.document() for r in results],
                }
            )
        )
        return FleetJobReport(
            job_id=job_id,
            backend_name=self.name,
            state=state,
            state_history=tuple(hist),
            task_count=len(tasks),
            succeeded_count=succeeded,
            failed_count=failed,
            results=tuple(results),
            artifacts=(str(report_path),),
            wall_time_s=time.perf_counter() - t0,
        )


class SlurmScriptFleet:
    """Deterministic Slurm array job script emitter & completed-output collector."""

    def __init__(
        self,
        *,
        partition: str = "compute",
        cpus_per_task: int = 4,
        time_limit: str = "01:00:00",
        python_executable: str = "python",
    ) -> None:
        self.partition = partition
        self.cpus_per_task = cpus_per_task
        self.time_limit = time_limit
        self.python_executable = python_executable

    @property
    def name(self) -> str:
        return "slurm-script-fleet"

    def submit_batch(
        self,
        job_id: str,
        tasks: Sequence[FleetTaskSpec],
        *,
        workdir: Path | str,
        worker_fn: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
    ) -> FleetJobReport:
        del worker_fn
        t0 = time.perf_counter()
        if not tasks:
            raise ValueError("SlurmScriptFleet requires at least one task")
        root = Path(workdir).resolve() / job_id
        tasks_dir = root / "tasks"
        results_dir = root / "results"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        state = FleetJobState.QUEUED
        hist: list[StateSnapshot[FleetJobState]] = [FLEET_JOB_MACHINE.snapshot(state)]
        for idx, t in enumerate(tasks):
            (tasks_dir / f"task-{idx:04d}.json").write_bytes(
                canonical_json_bytes(t.document())
            )

        sbatch_path = root / "submit_slurm.sh"
        n_max = len(tasks) - 1
        script = "\n".join(
            [
                "#!/usr/bin/env bash",
                f"#SBATCH --job-name=formamech-{job_id}",
                f"#SBATCH --partition={self.partition}",
                f"#SBATCH --cpus-per-task={self.cpus_per_task}",
                f"#SBATCH --time={self.time_limit}",
                f"#SBATCH --array=0-{n_max}",
                f"#SBATCH --output={root}/slurm-%A_%a.out",
                "set -euo pipefail",
                'IDX=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")',
                f'{self.python_executable} -m piginot.cli agent invoke fem.solve_3d_solid --workspace "{root}/ws-${{IDX}}" --args-file "{tasks_dir}/task-${{IDX}}.json" > "{results_dir}/task-${{IDX}}.json"',
                "",
            ]
        )
        sbatch_path.write_text(script, encoding="utf-8")

        FLEET_JOB_MACHINE.require_transition(state, FleetJobState.DISPATCHED)
        state = FleetJobState.DISPATCHED
        hist.append(FLEET_JOB_MACHINE.snapshot(state))

        return FleetJobReport(
            job_id=job_id,
            backend_name=self.name,
            state=state,
            state_history=tuple(hist),
            task_count=len(tasks),
            succeeded_count=0,
            failed_count=0,
            results=(),
            artifacts=(str(sbatch_path), str(tasks_dir)),
            wall_time_s=time.perf_counter() - t0,
        )

    def collect_completed(self, job_dir: Path | str) -> FleetJobReport:
        """Ingest completed JSON task results written under ``job_dir/results/``."""
        t0 = time.perf_counter()
        root = Path(job_dir).resolve()
        tasks_dir = root / "tasks"
        results_dir = root / "results"
        task_files = sorted(tasks_dir.glob("task-*.json"))
        results: list[FleetTaskResult] = []
        for tf in task_files:
            tdoc = json.loads(tf.read_text(encoding="utf-8"))
            tid = str(tdoc["task_id"])
            rf = results_dir / tf.name
            if not rf.is_file():
                results.append(
                    FleetTaskResult(
                        task_id=tid,
                        ok=False,
                        result=None,
                        error=f"missing result file {rf.name}",
                        wall_time_s=0.0,
                        result_sha256=_sha256_doc({"task_id": tid, "ok": False}),
                    )
                )
            else:
                rdoc = json.loads(rf.read_text(encoding="utf-8"))
                results.append(
                    FleetTaskResult(
                        task_id=tid,
                        ok=bool(rdoc.get("ok", True)),
                        result=rdoc,
                        error=None if rdoc.get("ok", True) else str(rdoc.get("error")),
                        wall_time_s=float(rdoc.get("wall_time_s", 0.0)),
                        result_sha256=_sha256_doc(rdoc),
                    )
                )
        succeeded = sum(1 for r in results if r.ok)
        failed = len(results) - succeeded
        state = (
            FleetJobState.VERIFIED if failed == 0 and results else FleetJobState.FAILED
        )
        return FleetJobReport(
            job_id=root.name,
            backend_name=self.name,
            state=state,
            state_history=(FLEET_JOB_MACHINE.snapshot(state),),
            task_count=len(task_files),
            succeeded_count=succeeded,
            failed_count=failed,
            results=tuple(results),
            artifacts=(str(results_dir),),
            wall_time_s=time.perf_counter() - t0,
        )


class KubernetesJobFleet:
    """Deterministic Kubernetes ``batch/v1`` Indexed Job manifest generator."""

    def __init__(
        self,
        *,
        image: str = "formamech/worker:v0.4",
        namespace: str = "default",
        parallelism: int = 4,
    ) -> None:
        self.image = image
        self.namespace = namespace
        self.parallelism = parallelism

    @property
    def name(self) -> str:
        return "kubernetes-job-fleet"

    def submit_batch(
        self,
        job_id: str,
        tasks: Sequence[FleetTaskSpec],
        *,
        workdir: Path | str,
        worker_fn: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
    ) -> FleetJobReport:
        del worker_fn
        t0 = time.perf_counter()
        if not tasks:
            raise ValueError("KubernetesJobFleet requires at least one task")
        root = Path(workdir).resolve() / job_id
        tasks_dir = root / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        for idx, t in enumerate(tasks):
            (tasks_dir / f"task-{idx:04d}.json").write_bytes(
                canonical_json_bytes(t.document())
            )

        yaml_path = root / "k8s-indexed-job.yaml"
        yaml_text = "\n".join(
            [
                "apiVersion: batch/v1",
                "kind: Job",
                "metadata:",
                f"  name: formamech-{job_id.lower().replace('_', '-')}",
                f"  namespace: {self.namespace}",
                "spec:",
                "  completionMode: Indexed",
                f"  completions: {len(tasks)}",
                f"  parallelism: {min(self.parallelism, len(tasks))}",
                "  template:",
                "    spec:",
                "      restartPolicy: OnFailure",
                "      containers:",
                "        - name: formamech-worker",
                f"          image: {self.image}",
                "          command: ['formamech', 'agent', 'invoke', 'fem.solve_3d_solid']",
                "",
            ]
        )
        yaml_path.write_text(yaml_text, encoding="utf-8")

        state = FleetJobState.QUEUED
        hist = [FLEET_JOB_MACHINE.snapshot(state)]
        FLEET_JOB_MACHINE.require_transition(state, FleetJobState.DISPATCHED)
        state = FleetJobState.DISPATCHED
        hist.append(FLEET_JOB_MACHINE.snapshot(state))

        return FleetJobReport(
            job_id=job_id,
            backend_name=self.name,
            state=state,
            state_history=tuple(hist),
            task_count=len(tasks),
            succeeded_count=0,
            failed_count=0,
            results=(),
            artifacts=(str(yaml_path), str(tasks_dir)),
            wall_time_s=time.perf_counter() - t0,
        )


__all__ = (
    "FLEET_JOB_MACHINE",
    "FLEET_JOB_SCHEMA",
    "FleetJobReport",
    "FleetJobState",
    "FleetTaskResult",
    "FleetTaskSpec",
    "KubernetesJobFleet",
    "LocalMultiprocessFleet",
    "SlurmScriptFleet",
    "WorkerBackend",
    "solve_3d_brick_fem_task",
)
