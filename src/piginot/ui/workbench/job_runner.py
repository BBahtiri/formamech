"""Detached background job runner for Workbench compile and qualification runs.

Runs ``formamech compile`` and ``formamech qualify run`` in child subprocesses
with file-persisted status manifests and streaming log files so Streamlit
reruns never interrupt long-running training or FEM solves.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

JOB_STATUS_SCHEMA: Final = "formamech.workbench-job-v1"
JobKind = Literal["compile", "qualify"]
JobState = Literal[
    "PENDING", "RUNNING", "SUCCEEDED", "REFUSED_SCIENTIFIC", "FAILED", "CANCELLED"
]

# In-process registry mapping job_id -> live Popen handle for the current UI server
_LIVE_PROCS: dict[str, subprocess.Popen[Any]] = {}


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _state_from_exit_code(code: int) -> JobState:
    if code == 0:
        return "SUCCEEDED"
    if code == 3:
        return "REFUSED_SCIENTIFIC"
    return "FAILED"


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
        valid_states = (
            "PENDING",
            "RUNNING",
            "SUCCEEDED",
            "REFUSED_SCIENTIFIC",
            "FAILED",
            "CANCELLED",
        )
        if state_raw not in valid_states:
            raise ValueError(f"unsupported job state {state_raw!r}")
        state: JobState = state_raw  # type: ignore[assignment]
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
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            return str(pid) in proc.stdout
        except OSError:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class WorkbenchJobRunner:
    """Manages detached subprocess jobs under ``<workspace>/jobs/``."""

    def __init__(self, jobs_dir: Path | str, *, cwd: Path | str | None = None) -> None:
        self.jobs_dir = Path(jobs_dir).resolve()
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.cwd = None if cwd is None else Path(cwd).resolve()

    def _record_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _save(self, record: JobRecord) -> None:
        self._record_path(record.job_id).write_text(
            json.dumps(record.document(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

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

        _LIVE_PROCS[jid] = proc
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
        return self.launch(kind="qualify", cli_args=args, output_dir=output_dir)

    def poll(self, job_id: str) -> JobRecord:
        """Refresh state, exit code, and checkpoint progress for ``job_id``."""
        record = self.load(job_id)
        out_path = Path(record.output_dir)
        step = _latest_checkpoint_step(out_path)
        if record.state != "RUNNING":
            if step != record.latest_checkpoint_step:
                updated = replace(record, latest_checkpoint_step=step)
                self._save(updated)
                return updated
            return record

        proc = _LIVE_PROCS.get(job_id)
        if proc is not None:
            ret = proc.poll()
            if ret is None:
                updated = replace(record, latest_checkpoint_step=step)
                self._save(updated)
                return updated
            _LIVE_PROCS.pop(job_id, None)
            finished = replace(
                record,
                state=_state_from_exit_code(int(ret)),
                exit_code=int(ret),
                finished_utc=_utc_now(),
                latest_checkpoint_step=step,
            )
            self._save(finished)
            return finished

        # Process started in a prior server session: probe OS PID
        if record.pid is not None and _pid_alive(record.pid):
            updated = replace(record, latest_checkpoint_step=step)
            self._save(updated)
            return updated

        # Infer terminal state from expected output artifacts when PID exited externally
        inferred_ok = (
            (out_path / "candidate-bundle" / "bundle.json").is_file()
            if record.kind == "compile"
            else (out_path / "qualification-manifest.json").is_file()
        )
        finished = replace(
            record,
            state="SUCCEEDED" if inferred_ok else "FAILED",
            exit_code=0 if inferred_ok else 1,
            finished_utc=_utc_now(),
            latest_checkpoint_step=step,
        )
        self._save(finished)
        return finished

    def cancel(self, job_id: str) -> JobRecord:
        """Terminate a running workbench job cleanly."""
        record = self.load(job_id)
        proc = _LIVE_PROCS.pop(job_id, None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
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
    "JOB_STATUS_SCHEMA",
    "JobKind",
    "JobRecord",
    "JobState",
    "WorkbenchJobRunner",
)
