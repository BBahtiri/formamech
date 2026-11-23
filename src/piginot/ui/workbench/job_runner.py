"""Detached background job runner for Workbench compile and qualification runs.

Runs ``formamech compile`` and ``formamech qualify run`` in child subprocesses
with file-persisted status manifests and streaming log files so Streamlit
reruns never interrupt long-running training or FEM solves.
"""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal

from piginot.contracts.state_machine import (
    StateMachineSpec,
    StateSnapshot,
    TransitionSpec,
    register_state_machine,
)

JOB_STATUS_SCHEMA: Final = "formamech.workbench-job-v1"
JobKind = Literal["compile", "qualify"]
JobState = Literal[
    "PENDING", "RUNNING", "SUCCEEDED", "REFUSED_SCIENTIFIC", "FAILED", "CANCELLED"
]


class JobLifecycleState(StrEnum):
    """Explicit finite-state-machine states for a detached workbench job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    REFUSED_SCIENTIFIC = "REFUSED_SCIENTIFIC"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


JOB_LIFECYCLE_MACHINE: Final[StateMachineSpec[JobLifecycleState]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.workbench.job",
            description=(
                "Detached subprocess lifecycle for Workbench compile and "
                "qualification jobs. Terminal states are strictly immutable."
            ),
            states=tuple(JobLifecycleState),
            initial_states=(JobLifecycleState.PENDING, JobLifecycleState.RUNNING),
            terminal_states=(
                JobLifecycleState.SUCCEEDED,
                JobLifecycleState.REFUSED_SCIENTIFIC,
                JobLifecycleState.FAILED,
                JobLifecycleState.CANCELLED,
            ),
            transitions=(
                TransitionSpec(
                    JobLifecycleState.PENDING,
                    JobLifecycleState.RUNNING,
                    "start",
                    "Subprocess spawned with a live OS PID.",
                ),
                TransitionSpec(
                    JobLifecycleState.PENDING,
                    JobLifecycleState.CANCELLED,
                    "cancel",
                    "Job cancelled before subprocess launch.",
                ),
                TransitionSpec(
                    JobLifecycleState.PENDING,
                    JobLifecycleState.FAILED,
                    "launch_failed",
                    "Subprocess failed to start.",
                ),
                TransitionSpec(
                    JobLifecycleState.RUNNING,
                    JobLifecycleState.RUNNING,
                    "poll_alive",
                    "Process is still alive; refresh checkpoint progress.",
                ),
                TransitionSpec(
                    JobLifecycleState.RUNNING,
                    JobLifecycleState.SUCCEEDED,
                    "exit_ok",
                    "Subprocess exited cleanly with code 0.",
                ),
                TransitionSpec(
                    JobLifecycleState.RUNNING,
                    JobLifecycleState.REFUSED_SCIENTIFIC,
                    "exit_scientific_refusal",
                    "Subprocess exited with code 3 (scientific refusal or NOT_QUALIFIED).",
                ),
                TransitionSpec(
                    JobLifecycleState.RUNNING,
                    JobLifecycleState.FAILED,
                    "exit_error",
                    "Subprocess exited with a non-zero operational/refusal error code.",
                ),
                TransitionSpec(
                    JobLifecycleState.RUNNING,
                    JobLifecycleState.CANCELLED,
                    "cancel",
                    "User or caller terminated the running subprocess.",
                ),
            ),
        )
    )
)

# In-process registry mapping (jobs_dir, job_id) -> live Popen handle for current UI server
_LIVE_PROCS: dict[tuple[str, str], subprocess.Popen[Any]] = {}


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _state_from_exit_code(code: int) -> JobState:
    if code == 0:
        return "SUCCEEDED"
    if code == 3:
        return "REFUSED_SCIENTIFIC"
    return "FAILED"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class JobRecord:
    """Persisted metadata and progress snapshot for one workbench background job."""

    job_id: str
    kind: JobKind
    command: tuple[str, ...]
    pid: int | None
    state: JobState
    exit_code: int | None
    started_utc: str
    finished_utc: str | None
    log_path: str
    output_dir: str
    latest_checkpoint_step: int | None = None

    def __post_init__(self) -> None:
        lifecycle = JobLifecycleState(str(self.state))
        object.__setattr__(self, "state", lifecycle.value)

    @property
    def lifecycle_state(self) -> JobLifecycleState:
        return JobLifecycleState(self.state)

    @property
    def is_terminal(self) -> bool:
        return JOB_LIFECYCLE_MACHINE.is_terminal(self.lifecycle_state)

    def snapshot(self) -> StateSnapshot[JobLifecycleState]:
        """Self-describing state snapshot for UI badges and agent polling."""
        state = self.lifecycle_state
        hint = (
            "Job is running; poll for checkpoint progress or cancel."
            if state is JobLifecycleState.RUNNING
            else (
                "Job is queued to launch."
                if state is JobLifecycleState.PENDING
                else f"Job finished in terminal state {state.value}."
            )
        )
        return JOB_LIFECYCLE_MACHINE.snapshot(
            state,
            recommended_trigger="poll_alive"
            if state is JobLifecycleState.RUNNING
            else None,
            recommended_hint=hint,
            metadata={
                "job_id": self.job_id,
                "kind": self.kind,
                "exit_code": self.exit_code,
                "latest_checkpoint_step": self.latest_checkpoint_step,
            },
        )

    def document(self) -> dict[str, Any]:
        return {
            "schema": JOB_STATUS_SCHEMA,
            "job_id": self.job_id,
            "kind": self.kind,
            "command": list(self.command),
            "pid": self.pid,
            "state": self.state,
            "exit_code": self.exit_code,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "log_path": self.log_path,
            "output_dir": self.output_dir,
            "latest_checkpoint_step": self.latest_checkpoint_step,
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> JobRecord:
        if doc.get("schema") != JOB_STATUS_SCHEMA:
            raise ValueError(f"not a {JOB_STATUS_SCHEMA} job record")
        kind_raw = str(doc["kind"])
        if kind_raw not in ("compile", "qualify"):
            raise ValueError(f"unsupported job kind {kind_raw!r}")
        kind: JobKind = "compile" if kind_raw == "compile" else "qualify"
        state_raw = str(doc["state"])
        try:
            state_enum = JobLifecycleState(state_raw)
        except ValueError as exc:
            raise ValueError(f"unsupported job state {state_raw!r}") from exc
        state: JobState = state_enum.value
        raw_pid = doc.get("pid")
        raw_code = doc.get("exit_code")
        raw_step = doc.get("latest_checkpoint_step")
        return cls(
            job_id=str(doc["job_id"]),
            kind=kind,
            command=tuple(str(a) for a in doc.get("command", ())),
            pid=None if raw_pid is None else int(raw_pid),
            state=state,
            exit_code=None if raw_code is None else int(raw_code),
            started_utc=str(doc["started_utc"]),
            finished_utc=(
                None if doc.get("finished_utc") is None else str(doc["finished_utc"])
            ),
            log_path=str(doc["log_path"]),
            output_dir=str(doc["output_dir"]),
            latest_checkpoint_step=None if raw_step is None else int(raw_step),
        )


def _latest_checkpoint_step(output_dir: Path) -> int | None:
    ckpt_dir = output_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    steps: list[int] = []
    for entry in ckpt_dir.iterdir():
        if entry.name.startswith("step-"):
            tail = entry.name[len("step-") :]
            if tail.isdigit():
                steps.append(int(tail))
    return max(steps) if steps else None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            proc = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            pid_str = str(pid)
            for row in csv.reader(proc.stdout.splitlines()):
                if len(row) >= 2 and row[1].strip() == pid_str:
                    return True
            return False
        except OSError:
            return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _terminate_external_pid(pid: int) -> None:
    """Best-effort termination of a running subprocess from a prior server session."""
    if pid <= 0 or not _pid_alive(pid):
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except OSError:
            return
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return


def _infer_external_terminal(
    kind: JobKind, out_path: Path
) -> tuple[JobLifecycleState, int]:
    """Infer terminal state & exit code when an externally-owned PID has exited."""
    if kind == "compile":
        report_path = out_path / "report.json"
        if report_path.is_file():
            try:
                report_doc = json.loads(report_path.read_text(encoding="utf-8"))
                if isinstance(report_doc, dict) and isinstance(
                    report_doc.get("exit_code"), int
                ):
                    code = int(report_doc["exit_code"])
                    return JobLifecycleState(_state_from_exit_code(code)), code
            except (OSError, json.JSONDecodeError, ValueError):
                pass
        if (out_path / "candidate-bundle" / "bundle.json").is_file():
            return JobLifecycleState.SUCCEEDED, 0
        return JobLifecycleState.FAILED, 1

    manifest_path = out_path / "qualification-manifest.json"
    if not manifest_path.is_file():
        return JobLifecycleState.FAILED, 1
    try:
        manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return JobLifecycleState.FAILED, 1
    if (
        isinstance(manifest_doc, dict)
        and manifest_doc.get("verdict") == "NOT_QUALIFIED"
    ):
        return JobLifecycleState.REFUSED_SCIENTIFIC, 3
    return JobLifecycleState.SUCCEEDED, 0


class WorkbenchJobRunner:
    """Manages detached subprocess jobs under ``<workspace>/jobs/``."""

    def __init__(self, jobs_dir: Path | str, *, cwd: Path | str | None = None) -> None:
        self.jobs_dir = Path(jobs_dir).resolve()
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.cwd = None if cwd is None else Path(cwd).resolve()

    def _proc_key(self, job_id: str) -> tuple[str, str]:
        return (str(self.jobs_dir), job_id)

    def _record_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _save(self, record: JobRecord) -> None:
        _atomic_write_json(self._record_path(record.job_id), record.document())

    def load(self, job_id: str) -> JobRecord:
        path = self._record_path(job_id)
        if not path.is_file():
            raise FileNotFoundError(f"unknown workbench job {job_id!r}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return JobRecord.from_document(raw)

    def list_jobs(self) -> tuple[JobRecord, ...]:
        records: list[JobRecord] = []
        for path in sorted(self.jobs_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                records.append(JobRecord.from_document(raw))
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                continue
        records.sort(key=lambda r: r.started_utc, reverse=True)
        return tuple(records)

    def active_jobs(self, *, kind: JobKind | None = None) -> tuple[JobRecord, ...]:
        """Poll and return all currently non-terminal jobs (optionally filtered by kind)."""
        active: list[JobRecord] = []
        for rec in self.list_jobs():
            if kind is not None and rec.kind != kind:
                continue
            if not rec.is_terminal:
                refreshed = self.poll(rec.job_id)
                if not refreshed.is_terminal:
                    active.append(refreshed)
        return tuple(active)

    def launch(
        self,
        *,
        kind: JobKind,
        cli_args: Sequence[str],
        output_dir: Path | str,
        job_id: str | None = None,
    ) -> JobRecord:
        """Spawn ``python -m formamech <cli_args>`` asynchronously with log capture."""
        jid = job_id or f"{kind}-{uuid.uuid4().hex[:10]}"
        log_file = self.jobs_dir / f"{jid}.log"
        cmd = (sys.executable, "-m", "formamech", *(str(a) for a in cli_args))
        out_path = Path(output_dir).resolve()

        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")

        handle = log_file.open("w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                list(cmd),
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=str(self.cwd) if self.cwd is not None else None,
                env=env,
            )
        finally:
            handle.close()

        JOB_LIFECYCLE_MACHINE.require_transition(
            JobLifecycleState.PENDING,
            JobLifecycleState.RUNNING,
            trigger="start",
        )
        _LIVE_PROCS[self._proc_key(jid)] = proc
        record = JobRecord(
            job_id=jid,
            kind=kind,
            command=cmd,
            pid=int(proc.pid),
            state="RUNNING",
            exit_code=None,
            started_utc=_utc_now(),
            finished_utc=None,
            log_path=str(log_file),
            output_dir=str(out_path),
            latest_checkpoint_step=_latest_checkpoint_step(out_path),
        )
        self._save(record)
        return record

    def start_compile(
        self,
        *,
        spec_path: Path | str,
        config_path: Path | str,
        output_dir: Path | str,
        resume: bool = False,
    ) -> JobRecord:
        args = [
            "compile",
            "--spec",
            str(Path(spec_path).resolve()),
            "--config",
            str(Path(config_path).resolve()),
            "--output",
            str(Path(output_dir).resolve()),
        ]
        if resume:
            args.append("--resume")
        root = self.jobs_dir.parent
        output = Path(output_dir).resolve()
        inputs = (Path(spec_path).resolve(), Path(config_path).resolve())
        expected = (root / "inputs" / "problem.json", root / "inputs" / "config.json")
        if self.jobs_dir.name == "jobs" and (
            output == root / "run" or inputs == expected
        ):
            from piginot.ui.workbench.state import WorkbenchWorkspace
            from piginot.ui.workbench.study_plan import (
                has_compatible_checkpoint,
                plan_blocker,
            )

            workspace = WorkbenchWorkspace(root)
            with workspace.locked():
                if inputs != expected:
                    raise ValueError(
                        "Compile requires this study's prepared problem and configuration."
                    )
                if blocker := plan_blocker(workspace):
                    raise ValueError(blocker)
                if self.active_jobs(kind="compile"):
                    raise ValueError("A compile job is already active for this study.")
                if resume and not has_compatible_checkpoint(workspace, output):
                    raise ValueError(
                        "Resume requires a compatible checkpoint from this plan."
                    )
                return self.launch(kind="compile", cli_args=args, output_dir=output_dir)
        return self.launch(kind="compile", cli_args=args, output_dir=output_dir)

    def start_qualify(
        self,
        *,
        bundle_path: Path | str,
        output_dir: Path | str,
        preset: str = "dev",
        policy_path: Path | str | None = None,
        sample_dev: int = 9,
        sample_holdout: int = 0,
        backend: str = "torch-fem",
        seed: int = 0,
        emit_qualified_bundle: Path | str | None = None,
    ) -> JobRecord:
        args = [
            "qualify",
            "run",
            str(Path(bundle_path).resolve()),
            "--sample-dev",
            str(int(sample_dev)),
            "--sample-holdout",
            str(int(sample_holdout)),
            "--seed",
            str(int(seed)),
            "--backend",
            str(backend),
            "--output",
            str(Path(output_dir).resolve()),
        ]
        if policy_path is not None:
            args.extend(["--policy", str(Path(policy_path).resolve())])
        else:
            args.extend(["--preset", str(preset)])
        if emit_qualified_bundle is not None:
            args.extend(
                ["--emit-qualified-bundle", str(Path(emit_qualified_bundle).resolve())]
            )
        root = self.jobs_dir.parent
        if (
            self.jobs_dir.name == "jobs"
            and Path(output_dir).resolve() == root / "qualification"
        ):
            from piginot.ui.workbench.state import WorkbenchWorkspace

            with WorkbenchWorkspace(root).locked():
                if self.active_jobs(kind="qualify"):
                    raise ValueError(
                        "A qualification job is already active for this study."
                    )
                return self.launch(kind="qualify", cli_args=args, output_dir=output_dir)
        return self.launch(kind="qualify", cli_args=args, output_dir=output_dir)

    def poll(self, job_id: str) -> JobRecord:
        """Refresh state, exit code, and checkpoint progress for ``job_id``."""
        record = self.load(job_id)
        current = record.lifecycle_state
        # Terminal states are strictly immutable: never contaminate finished records
        # with checkpoints written by later runs in the same output_dir.
        if JOB_LIFECYCLE_MACHINE.is_terminal(current):
            return record
        if current is JobLifecycleState.PENDING:
            return record

        out_path = Path(record.output_dir)
        step = _latest_checkpoint_step(out_path)

        proc_key = self._proc_key(job_id)
        proc = _LIVE_PROCS.get(proc_key)
        if proc is not None:
            ret = proc.poll()
            if ret is None:
                JOB_LIFECYCLE_MACHINE.require_transition(
                    current, JobLifecycleState.RUNNING, trigger="poll_alive"
                )
                updated = replace(record, latest_checkpoint_step=step)
                self._save(updated)
                return updated
            _LIVE_PROCS.pop(proc_key, None)
            next_state_str = _state_from_exit_code(int(ret))
            next_state = JobLifecycleState(next_state_str)
            JOB_LIFECYCLE_MACHINE.require_transition(current, next_state)
            finished = replace(
                record,
                state=next_state_str,
                exit_code=int(ret),
                finished_utc=_utc_now(),
                latest_checkpoint_step=step,
            )
            self._save(finished)
            return finished

        # Process started in a prior server session: probe OS PID
        if record.pid is not None and _pid_alive(record.pid):
            JOB_LIFECYCLE_MACHINE.require_transition(
                current, JobLifecycleState.RUNNING, trigger="poll_alive"
            )
            updated = replace(record, latest_checkpoint_step=step)
            self._save(updated)
            return updated

        # Infer terminal state accurately from persisted report/manifest artifacts
        inferred_state, inferred_code = _infer_external_terminal(record.kind, out_path)
        JOB_LIFECYCLE_MACHINE.require_transition(current, inferred_state)
        finished = replace(
            record,
            state=inferred_state.value,
            exit_code=inferred_code,
            finished_utc=_utc_now(),
            latest_checkpoint_step=step,
        )
        self._save(finished)
        return finished

    def cancel(self, job_id: str) -> JobRecord:
        """Terminate a non-terminal workbench job cleanly (refuses terminal jobs)."""
        record = self.load(job_id)
        JOB_LIFECYCLE_MACHINE.require_transition(
            record.lifecycle_state,
            JobLifecycleState.CANCELLED,
            trigger="cancel",
            reason="cannot cancel a job already in a terminal state",
        )
        key = self._proc_key(job_id)
        proc = _LIVE_PROCS.get(key)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        elif proc is None and record.pid is not None:
            _terminate_external_pid(record.pid)
            deadline = time.monotonic() + 5
            while _pid_alive(record.pid):
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Job {job_id} is still running after cancellation."
                    )
                time.sleep(0.1)
        _LIVE_PROCS.pop(key, None)

        cancelled = replace(
            record,
            state="CANCELLED",
            finished_utc=_utc_now(),
        )
        self._save(cancelled)
        return cancelled

    def read_log_tail(self, job_id: str, *, max_lines: int = 80) -> str:
        record = self.load(job_id)
        log_path = Path(record.log_path)
        if not log_path.is_file():
            return ""
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])


__all__ = (
    "JOB_LIFECYCLE_MACHINE",
    "JOB_STATUS_SCHEMA",
    "JobKind",
    "JobLifecycleState",
    "JobRecord",
    "JobState",
    "WorkbenchJobRunner",
)
