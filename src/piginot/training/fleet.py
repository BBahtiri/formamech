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
import math
import os
import re
import shlex
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from piginot.contracts import MaterialSpec, Quantity, Unit, canonical_json_bytes
from piginot.contracts.canonical import strict_json_loads
from piginot.contracts.errors import (
    ContractValidationError,
    DuplicateKeyError,
    NonFiniteNumberError,
)
from piginot.contracts.state_machine import (
    StateMachineSpec,
    StateSnapshot,
    TransitionSpec,
    register_state_machine,
)

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


_SAFE_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")
_SLURM_TIME_RE: Final = re.compile(r"^(\d+-)?\d{1,2}:\d{2}(:\d{2})?$")
_CONTAINER_IMAGE_RE: Final = re.compile(r"^[^\s\x00-\x1f]+$")


def _validate_safe_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID_RE.match(value):
        raise ValueError(
            f"invalid {label} {value!r}; must match {_SAFE_ID_RE.pattern!r}"
        )
    return value


def _resolve_job_root(workdir: Path | str, job_id: str) -> Path:
    _validate_safe_id(job_id, "job_id")
    base = Path(workdir).resolve()
    root = (base / job_id).resolve()
    if not root.is_relative_to(base):
        raise ValueError(f"job_id {job_id!r} escapes workdir {base}")
    return root


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _sha256_doc(doc: object) -> str:
    return hashlib.sha256(canonical_json_bytes(doc)).hexdigest()


def solve_3d_brick_fem_task(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Top-level picklable 3D hyperelastic FEM worker task for multiprocess fleets."""
    from piginot.fem.backend_3d import (
        DirichletBC3D,
        FemRequest3D,
        TorchFemBackend3D,
        TractionBC3D,
    )
    from piginot.geometry.mesh3d import make_brick_tet4_mesh

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
    if not res.converged:
        raise RuntimeError(f"3D FEM task did not converge: {res.reason}")
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
        err_msg = f"{type(exc).__name__}: {exc}"
        return FleetTaskResult(
            task_id=task_id,
            ok=False,
            result=None,
            error=err_msg,
            wall_time_s=dt,
            result_sha256=_sha256_doc(
                {"task_id": task_id, "ok": False, "error": err_msg}
            ),
        )


def _collect_disk_results(job_dir: Path | str, *, backend_name: str) -> FleetJobReport:
    """Ingest & verify completed JSON task results written under ``job_dir/results/``."""
    t0 = time.perf_counter()
    root = Path(job_dir).resolve()
    tasks_dir = root / "tasks"
    results_dir = root / "results"
    task_files = sorted(tasks_dir.glob("task-*.json"))

    state = FleetJobState.QUEUED
    hist: list[StateSnapshot[FleetJobState]] = [FLEET_JOB_MACHINE.snapshot(state)]
    for nxt in (
        FleetJobState.DISPATCHED,
        FleetJobState.RUNNING,
        FleetJobState.COLLECTED,
    ):
        FLEET_JOB_MACHINE.require_transition(state, nxt)
        state = nxt
        hist.append(FLEET_JOB_MACHINE.snapshot(state))

    results: list[FleetTaskResult] = []
    for tf in task_files:
        tdoc = strict_json_loads(tf.read_text(encoding="utf-8"))
        if not isinstance(tdoc, dict):
            raise ValueError(f"corrupt task spec file {tf.name}")
        raw_tid = tdoc.get("task_id")
        if not isinstance(raw_tid, str):
            raise ValueError(f"corrupt task spec file {tf.name}")
        tid: str = raw_tid
        rf = results_dir / tf.name
        if not rf.is_file():
            err_msg = f"missing result file {rf.name}"
            results.append(
                FleetTaskResult(
                    task_id=tid,
                    ok=False,
                    result=None,
                    error=err_msg,
                    wall_time_s=0.0,
                    result_sha256=_sha256_doc(
                        {"task_id": tid, "ok": False, "error": err_msg}
                    ),
                )
            )
            continue
        try:
            rdoc = strict_json_loads(rf.read_text(encoding="utf-8"))
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ContractValidationError,
            DuplicateKeyError,
            NonFiniteNumberError,
        ) as exc:
            err_msg = f"corrupt JSON in {rf.name}: {exc}"
            results.append(
                FleetTaskResult(
                    task_id=tid,
                    ok=False,
                    result=None,
                    error=err_msg,
                    wall_time_s=0.0,
                    result_sha256=_sha256_doc(
                        {"task_id": tid, "ok": False, "error": err_msg}
                    ),
                )
            )
            continue
        if not isinstance(rdoc, dict):
            err_msg = f"result file {rf.name} must be a JSON object"
            results.append(
                FleetTaskResult(
                    task_id=tid,
                    ok=False,
                    result=None,
                    error=err_msg,
                    wall_time_s=0.0,
                    result_sha256=_sha256_doc(
                        {"task_id": tid, "ok": False, "error": err_msg}
                    ),
                )
            )
            continue
        if "task_id" in rdoc and str(rdoc["task_id"]) != tid:
            err_msg = f"task_id mismatch in {rf.name}: expected {tid!r}, got {rdoc['task_id']!r}"
            results.append(
                FleetTaskResult(
                    task_id=tid,
                    ok=False,
                    result=None,
                    error=err_msg,
                    wall_time_s=0.0,
                    result_sha256=_sha256_doc(
                        {"task_id": tid, "ok": False, "error": err_msg}
                    ),
                )
            )
            continue
        ok_flag = rdoc.get("ok") is True
        raw_wt = rdoc.get("wall_time_s", 0.0)
        wt = (
            float(raw_wt)
            if isinstance(raw_wt, (int, float))
            and not isinstance(raw_wt, bool)
            and math.isfinite(float(raw_wt))
            and float(raw_wt) >= 0.0
            else 0.0
        )
        err_val = None if ok_flag else str(rdoc.get("error", "result ok flag not true"))
        results.append(
            FleetTaskResult(
                task_id=tid,
                ok=ok_flag,
                result=rdoc if ok_flag else None,
                error=err_val,
                wall_time_s=wt,
                result_sha256=_sha256_doc(rdoc),
            )
        )

    succeeded = sum(1 for r in results if r.ok)
    failed = len(results) - succeeded
    final_state = (
        FleetJobState.VERIFIED
        if failed == 0 and bool(results)
        else FleetJobState.FAILED
    )
    FLEET_JOB_MACHINE.require_transition(state, final_state)
    state = final_state
    hist.append(FLEET_JOB_MACHINE.snapshot(state))

    return FleetJobReport(
        job_id=root.name,
        backend_name=backend_name,
        state=state,
        state_history=tuple(hist),
        task_count=len(task_files),
        succeeded_count=succeeded,
        failed_count=failed,
        results=tuple(results),
        artifacts=(str(results_dir),),
        wall_time_s=time.perf_counter() - t0,
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
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("max_workers must be an integer >= 1")
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
        if not tasks:
            raise ValueError("LocalMultiprocessFleet requires at least one task")
        fn = solve_3d_brick_fem_task if worker_fn is None else worker_fn
        root = _resolve_job_root(workdir, job_id)
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
        results: list[FleetTaskResult] = []
        with pool_cls(max_workers=min(self.max_workers, len(work_items))) as ex:
            futs = [
                (item[0], ex.submit(_execute_one_task, item)) for item in work_items
            ]
            for tid, fut in futs:
                try:
                    results.append(fut.result())
                except Exception as exc:
                    err_msg = f"{type(exc).__name__}: {exc}"
                    results.append(
                        FleetTaskResult(
                            task_id=tid,
                            ok=False,
                            result=None,
                            error=err_msg,
                            wall_time_s=0.0,
                            result_sha256=_sha256_doc(
                                {"task_id": tid, "ok": False, "error": err_msg}
                            ),
                        )
                    )

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
        _atomic_write_bytes(
            report_path,
            canonical_json_bytes(
                {
                    "schema": FLEET_JOB_SCHEMA,
                    "job_id": job_id,
                    "backend": self.name,
                    "state": state.value,
                    "results": [r.document() for r in results],
                }
            ),
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
        _validate_safe_id(partition, "partition")
        if type(cpus_per_task) is not int or cpus_per_task < 1:
            raise ValueError("cpus_per_task must be an integer >= 1")
        if not isinstance(time_limit, str) or not _SLURM_TIME_RE.match(time_limit):
            raise ValueError(f"invalid Slurm time_limit {time_limit!r}")
        if (
            not isinstance(python_executable, str)
            or not python_executable.strip()
            or any(c in python_executable for c in ("\n", "\r", "\x00"))
        ):
            raise ValueError("python_executable must be a single-line non-empty path")
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
        root = _resolve_job_root(workdir, job_id)
        tasks_dir = root / "tasks"
        results_dir = root / "results"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        state = FleetJobState.QUEUED
        hist: list[StateSnapshot[FleetJobState]] = [FLEET_JOB_MACHINE.snapshot(state)]
        for idx, t in enumerate(tasks):
            _atomic_write_bytes(
                tasks_dir / f"task-{idx:04d}.json",
                canonical_json_bytes(t.document()),
            )

        sbatch_path = root / "submit_slurm.sh"
        n_max = len(tasks) - 1
        q_py = shlex.quote(self.python_executable)
        q_root = shlex.quote(str(root))
        q_tasks = shlex.quote(str(tasks_dir))
        q_results = shlex.quote(str(results_dir))
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
                (
                    f"{q_py} -m piginot.cli agent invoke fem.solve_3d_solid "
                    f'--workspace {q_root}/ws-"${{IDX}}" '
                    f'--args-file {q_tasks}/task-"${{IDX}}".json '
                    f'> {q_results}/task-"${{IDX}}".json'
                ),
                "",
            ]
        )
        _atomic_write_bytes(sbatch_path, script.encode("utf-8"))

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
        return _collect_disk_results(job_dir, backend_name=self.name)


class KubernetesJobFleet:
    """Deterministic Kubernetes ``batch/v1`` Indexed Job manifest generator."""

    def __init__(
        self,
        *,
        image: str = "formamech/worker:v0.4",
        namespace: str = "default",
        parallelism: int = 4,
    ) -> None:
        if not isinstance(image, str) or not _CONTAINER_IMAGE_RE.match(image):
            raise ValueError(f"invalid container image {image!r}")
        _validate_safe_id(namespace, "namespace")
        if type(parallelism) is not int or parallelism < 1:
            raise ValueError("parallelism must be an integer >= 1")
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
        root = _resolve_job_root(workdir, job_id)
        tasks_dir = root / "tasks"
        results_dir = root / "results"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        for idx, t in enumerate(tasks):
            _atomic_write_bytes(
                tasks_dir / f"task-{idx:04d}.json",
                canonical_json_bytes(t.document()),
            )

        dns_name = re.sub(r"[^a-z0-9-]+", "-", f"formamech-{job_id.lower()}").strip(
            "-"
        )[:63]
        yaml_path = root / "k8s-indexed-job.yaml"
        yaml_text = "\n".join(
            [
                "apiVersion: batch/v1",
                "kind: Job",
                "metadata:",
                f"  name: {json.dumps(dns_name)}",
                f"  namespace: {json.dumps(self.namespace)}",
                "spec:",
                "  completionMode: Indexed",
                f"  completions: {len(tasks)}",
                f"  parallelism: {min(self.parallelism, len(tasks))}",
                "  template:",
                "    spec:",
                "      restartPolicy: OnFailure",
                "      containers:",
                "        - name: formamech-worker",
                f"          image: {json.dumps(self.image)}",
                '          command: ["formamech", "agent", "invoke", "fem.solve_3d_solid"]',
                "",
            ]
        )
        _atomic_write_bytes(yaml_path, yaml_text.encode("utf-8"))

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

    def collect_completed(self, job_dir: Path | str) -> FleetJobReport:
        """Ingest completed JSON task results written under ``job_dir/results/``."""
        return _collect_disk_results(job_dir, backend_name=self.name)


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
