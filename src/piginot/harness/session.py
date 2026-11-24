"""Persistent, sandboxed, hash-chained AI-agent harness session runtime (V4-0)."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from piginot.contracts.canonical import canonical_json_bytes
from piginot.contracts.state_machine import (
    InvalidTransitionError,
    StateSnapshot,
)
from piginot.harness.contracts import (
    HARNESS_LEDGER_ENTRY_SCHEMA,
    HARNESS_SESSION_MACHINE,
    HARNESS_SESSION_SCHEMA,
    CapabilitySpec,
    CostClass,
    HarnessBudget,
    HarnessEnvelope,
    HarnessSessionState,
)
from piginot.harness.registry import CapabilityRegistry, validate_json_schema_args
from piginot.mcp_server import ResourceLimitError, SandboxViolationError


def _utc_now() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_doc(doc: object) -> str:
    return hashlib.sha256(canonical_json_bytes(doc)).hexdigest()


def _error_recovery(exc: Exception) -> tuple[str, str, tuple[str, ...]]:
    if isinstance(exc, SandboxViolationError):
        return (
            "SANDBOX_VIOLATION",
            "Use a relative path inside the configured workspace root.",
            ("discover.workspace_tree",),
        )
    if isinstance(exc, ResourceLimitError):
        return (
            "BUDGET_OR_RESOURCE_LIMIT",
            "Reduce mesh DOF count, lower solve/epoch budget, or inspect session status.",
            ("discover.capabilities", "governance.verify_session_ledger"),
        )
    if isinstance(exc, InvalidTransitionError):
        return (
            "INVALID_STATE_TRANSITION",
            "Inspect allowed transitions via discover.state_machines before retrying.",
            ("discover.state_machines",),
        )
    if isinstance(exc, KeyError):
        return (
            "UNKNOWN_CAPABILITY",
            "Call discover.capabilities to list valid capability names.",
            ("discover.capabilities",),
        )
    if isinstance(exc, (FileNotFoundError, LookupError)):
        return (
            "RESOURCE_NOT_FOUND",
            "Check workspace files with discover.workspace_tree or generate/import required inputs first.",
            ("discover.workspace_tree", "cad.generate_sample_3d"),
        )
    if isinstance(exc, (TypeError, ValueError)):
        return (
            "INVALID_ARGUMENTS",
            "Inspect input_schema via discover.describe and fix malformed parameter types or bounds.",
            ("discover.describe",),
        )
    return (
        type(exc).__name__.upper(),
        "Inspect the error message and workspace artifacts before retrying.",
        ("discover.workspace_tree",),
    )


class HarnessContext:
    """Sandboxed execution context supplied to every registered capability handler."""

    def __init__(self, harness: FormaMechAgentHarness) -> None:
        self._harness = harness
        self._emitted_artifacts: list[str] = []
        self._receipt_sha256: str | None = None

    @property
    def workspace_root(self) -> Path:
        return self._harness.workspace_root

    @property
    def registry(self) -> CapabilityRegistry:
        return self._harness.registry

    @property
    def state_kv(self) -> dict[str, Any]:
        return self._harness._state_kv

    @property
    def harness(self) -> FormaMechAgentHarness:
        return self._harness

    def resolve_path(self, raw_path: str | Path) -> Path:
        return self._harness.resolve_path(raw_path)

    def relative_path(self, path: Path) -> str:
        return path.resolve().relative_to(self.workspace_root).as_posix()

    def check_dofs(self, n_dofs: int) -> None:
        if n_dofs > self._harness.budget.max_dofs:
            raise ResourceLimitError(
                f"mesh requires {n_dofs} DOFs, exceeding max_dofs={self._harness.budget.max_dofs}"
            )

    def consume_fem_solves(self, count: int) -> None:
        if count < 0:
            raise ValueError("FEM solve count must be nonnegative")
        used = self._harness.fem_solves_used
        cap = self._harness.budget.max_fem_solves
        if used + count > cap:
            self._harness._mark_budget_exhausted(
                f"FEM solve budget exhausted ({used}+{count} > {cap})"
            )
            raise ResourceLimitError(
                f"FEM solve budget exhausted: requested {count}, used {used}/{cap}"
            )
        self._harness.fem_solves_used += count

    def consume_training_epochs(self, epochs: int) -> None:
        if epochs < 0:
            raise ValueError("training epochs must be nonnegative")
        used = self._harness.training_epochs_used
        cap = self._harness.budget.max_training_epochs
        if used + epochs > cap:
            self._harness._mark_budget_exhausted(
                f"training epoch budget exhausted ({used}+{epochs} > {cap})"
            )
            raise ResourceLimitError(
                f"training epoch budget exhausted: requested {epochs}, used {used}/{cap}"
            )
        self._harness.training_epochs_used += epochs

    def record_artifact(self, path: Path | str) -> str:
        resolved = self.resolve_path(path)
        rel = self.relative_path(resolved)
        if rel not in self._emitted_artifacts:
            self._emitted_artifacts.append(rel)
        if rel not in self._harness._artifacts:
            self._harness._artifacts.append(rel)
        return rel

    def set_receipt(self, sha256_hex: str) -> None:
        self._receipt_sha256 = str(sha256_hex)
        self._harness._last_receipt_sha256 = self._receipt_sha256

    @property
    def emitted_artifacts(self) -> tuple[str, ...]:
        return tuple(self._emitted_artifacts)

    @property
    def receipt_sha256(self) -> str | None:
        return self._receipt_sha256


class FormaMechAgentHarness:
    """Full-capability AI-agent harness with persistent resumable hash-chained ledger."""

    def __init__(
        self,
        workspace_root: Path | str,
        *,
        registry: CapabilityRegistry | None = None,
        budget: HarnessBudget | None = None,
        session_dir: Path | str | None = None,
        session_id: str | None = None,
    ) -> None:
        root = Path(workspace_root).resolve()
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise ValueError(f"workspace_root must be a directory: {root}")
        self.workspace_root = root
        self.budget = HarnessBudget() if budget is None else budget
        if registry is None:
            from piginot.harness.catalog import build_default_registry

            self.registry = build_default_registry()
        else:
            self.registry = registry

        sdir = (
            root / ".formamech-agent"
            if session_dir is None
            else self.resolve_path(session_dir)
        )
        sdir.mkdir(parents=True, exist_ok=True)
        self.session_dir = sdir
        self.session_file = sdir / "session.json"
        self.ledger_file = sdir / "ledger.jsonl"

        self.session_id = session_id or f"fm-agent-{uuid.uuid4().hex[:12]}"
        self.state: HarnessSessionState = HarnessSessionState.INITIALIZED
        self.invocation_count: int = 0
        self.fem_solves_used: int = 0
        self.training_epochs_used: int = 0
        self.total_wall_time_s: float = 0.0
        self._prev_ledger_sha: str = "0" * 64
        self._artifacts: list[str] = []
        self._state_kv: dict[str, Any] = {}
        self._last_receipt_sha256: str | None = None
        self._persist_session()

    @classmethod
    def resume(
        cls,
        workspace_root: Path | str,
        *,
        session_dir: Path | str | None = None,
        registry: CapabilityRegistry | None = None,
    ) -> FormaMechAgentHarness:
        """Reload a persisted session after verifying its SHA-256 hash-chained ledger."""
        root = Path(workspace_root).resolve()
        sdir = (
            root / ".formamech-agent"
            if session_dir is None
            else (
                Path(session_dir).resolve()
                if Path(session_dir).is_absolute()
                else (root / session_dir).resolve()
            )
        )
        sfile = sdir / "session.json"
        if not sfile.is_file():
            raise FileNotFoundError(f"no saved harness session found at {sfile}")
        raw = json.loads(sfile.read_text(encoding="utf-8"))
        if raw.get("schema") != HARNESS_SESSION_SCHEMA:
            raise ValueError(f"unexpected session schema {raw.get('schema')!r}")

        harness = cls(
            root,
            registry=registry,
            budget=HarnessBudget.from_document(raw["budget"]),
            session_dir=sdir,
            session_id=str(raw["session_id"]),
        )
        harness.state = HarnessSessionState(str(raw["state"]))
        harness.invocation_count = int(raw["invocation_count"])
        harness.fem_solves_used = int(raw["fem_solves_used"])
        harness.training_epochs_used = int(raw["training_epochs_used"])
        harness.total_wall_time_s = float(raw["total_wall_time_s"])
        harness._prev_ledger_sha = str(raw["last_ledger_sha256"])
        harness._artifacts = [str(p) for p in raw.get("artifacts", ())]
        harness._state_kv = dict(raw.get("state_kv", {}))
        harness._last_receipt_sha256 = raw.get("last_receipt_sha256")

        ok, errs = harness.verify_ledger()
        if not ok:
            raise ValueError(
                "cannot resume session with tampered ledger: " + "; ".join(errs)
            )
        if harness.state is HarnessSessionState.SUSPENDED:
            harness.state = HARNESS_SESSION_MACHINE.require_transition(
                harness.state, HarnessSessionState.ACTIVE, trigger="resume"
            ).to_state
        harness._persist_session()
        return harness

    def resolve_path(self, raw_path: str | Path) -> Path:
        if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
            raise SandboxViolationError("path argument must be non-empty")
        cand = Path(raw_path)
        resolved = (
            cand.resolve()
            if cand.is_absolute()
            else (self.workspace_root / cand).resolve()
        )
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise SandboxViolationError(
                f"path {str(raw_path)!r} escapes workspace root {str(self.workspace_root)!r}"
            ) from exc
        return resolved

    def budget_status(self) -> dict[str, Any]:
        return {
            "fem_solves_used": self.fem_solves_used,
            "fem_solves_remaining": max(
                0, self.budget.max_fem_solves - self.fem_solves_used
            ),
            "max_fem_solves": self.budget.max_fem_solves,
            "training_epochs_used": self.training_epochs_used,
            "training_epochs_remaining": max(
                0, self.budget.max_training_epochs - self.training_epochs_used
            ),
            "max_training_epochs": self.budget.max_training_epochs,
            "max_dofs": self.budget.max_dofs,
            "total_wall_time_s": round(self.total_wall_time_s, 4),
            "max_wall_time_s": float(self.budget.max_wall_time_s),
        }

    def snapshot(self) -> StateSnapshot[HarnessSessionState]:
        hints = {
            HarnessSessionState.INITIALIZED: "Call discover.capabilities or invoke any capability to begin.",
            HarnessSessionState.ACTIVE: "Session active; invoke capabilities or call suspend/complete.",
            HarnessSessionState.BUDGET_EXHAUSTED: "Compute budget exhausted; only ZERO_SOLVE inspection/governance calls remain.",
            HarnessSessionState.SUSPENDED: "Session suspended; call FormaMechAgentHarness.resume() to continue.",
            HarnessSessionState.COMPLETED: "Session completed cleanly.",
            HarnessSessionState.ABORTED: "Session aborted.",
        }
        return HARNESS_SESSION_MACHINE.snapshot(
            self.state,
            recommended_hint=hints[self.state],
            metadata={
                "session_id": self.session_id,
                "invocation_count": self.invocation_count,
                "capability_count": len(self.registry),
                "artifact_count": len(self._artifacts),
                "last_receipt_sha256": self._last_receipt_sha256,
                **self.budget_status(),
            },
        )

    def _mark_budget_exhausted(self, reason: str) -> None:
        if self.state is HarnessSessionState.ACTIVE:
            self.state = HARNESS_SESSION_MACHINE.require_transition(
                self.state,
                HarnessSessionState.BUDGET_EXHAUSTED,
                trigger="exhaust_budget",
                reason=reason,
            ).to_state

    def suspend(self) -> StateSnapshot[HarnessSessionState]:
        self.state = HARNESS_SESSION_MACHINE.require_transition(
            self.state, HarnessSessionState.SUSPENDED, trigger="suspend"
        ).to_state
        self._persist_session()
        return self.snapshot()

    def complete(self) -> StateSnapshot[HarnessSessionState]:
        self.state = HARNESS_SESSION_MACHINE.require_transition(
            self.state, HarnessSessionState.COMPLETED, trigger="complete"
        ).to_state
        self._persist_session()
        return self.snapshot()

    def abort(self, reason: str = "aborted") -> StateSnapshot[HarnessSessionState]:
        self.state = HARNESS_SESSION_MACHINE.require_transition(
            self.state, HarnessSessionState.ABORTED, trigger="abort", reason=reason
        ).to_state
        self._persist_session()
        return self.snapshot()

    def _persist_session(self) -> None:
        doc = {
            "schema": HARNESS_SESSION_SCHEMA,
            "session_id": self.session_id,
            "workspace_root": str(self.workspace_root),
            "state": self.state.value,
            "invocation_count": self.invocation_count,
            "fem_solves_used": self.fem_solves_used,
            "training_epochs_used": self.training_epochs_used,
            "total_wall_time_s": float(self.total_wall_time_s),
            "budget": self.budget.document(),
            "last_ledger_sha256": self._prev_ledger_sha,
            "artifacts": list(self._artifacts),
            "state_kv": dict(self._state_kv),
            "last_receipt_sha256": self._last_receipt_sha256,
        }
        self.session_file.write_bytes(canonical_json_bytes(doc) + b"\n")

    def _append_ledger(
        self,
        *,
        capability: str,
        arguments: Mapping[str, Any],
        dry_run: bool,
        ok: bool,
        error_code: str | None,
        receipt_sha256: str | None,
        artifact_paths: tuple[str, ...],
        wall_time_s: float,
    ) -> str:
        body = {
            "schema": HARNESS_LEDGER_ENTRY_SCHEMA,
            "session_id": self.session_id,
            "index": self.invocation_count,
            "timestamp_utc": _utc_now(),
            "capability": capability,
            "arguments": dict(arguments),
            "dry_run": dry_run,
            "ok": ok,
            "error_code": error_code,
            "receipt_sha256": receipt_sha256,
            "artifact_paths": list(artifact_paths),
            "wall_time_s": round(float(wall_time_s), 6),
            "prev_sha256": self._prev_ledger_sha,
        }
        entry_sha = _sha256_doc(body)
        record = {**body, "entry_sha256": entry_sha}
        with self.ledger_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._prev_ledger_sha = entry_sha
        return entry_sha

    def verify_ledger(self) -> tuple[bool, tuple[str, ...]]:
        """Independently verify the SHA-256 hash chain of ``ledger.jsonl``."""
        if not self.ledger_file.is_file():
            return True, ()
        errors: list[str] = []
        prev = "0" * 64
        lines = [
            ln
            for ln in self.ledger_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        for idx, line in enumerate(lines, start=1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {idx}: invalid JSON ({exc})")
                continue
            recorded_sha = rec.get("entry_sha256")
            if rec.get("prev_sha256") != prev:
                errors.append(
                    f"line {idx}: prev_sha256 mismatch (expected {prev[:12]}...)"
                )
            body = {k: v for k, v in rec.items() if k != "entry_sha256"}
            expected = _sha256_doc(body)
            if recorded_sha != expected:
                errors.append(
                    f"line {idx}: entry_sha256 tampered (expected {expected[:12]}...)"
                )
            prev = str(recorded_sha or "")
        return len(errors) == 0, tuple(errors)

    def read_ledger(self) -> list[dict[str, Any]]:
        if not self.ledger_file.is_file():
            return []
        return [
            json.loads(ln)
            for ln in self.ledger_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    def invoke(
        self,
        capability: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        dry_run: bool = False,
    ) -> HarnessEnvelope:
        """Execute (or dry-run) any registered FormaMech capability safely."""
        t0 = time.perf_counter()
        args = {} if arguments is None else dict(arguments)
        self.invocation_count += 1
        ctx = HarnessContext(self)

        spec: CapabilitySpec | None = None
        try:
            spec, handler = self.registry.get(capability)
            if self.state in (
                HarnessSessionState.COMPLETED,
                HarnessSessionState.ABORTED,
                HarnessSessionState.SUSPENDED,
            ):
                HARNESS_SESSION_MACHINE.require_transition(
                    self.state, HarnessSessionState.ACTIVE, trigger="invoke"
                )
            if (
                self.state is HarnessSessionState.BUDGET_EXHAUSTED
                and spec.cost_class is not CostClass.ZERO_SOLVE
            ):
                raise ResourceLimitError(
                    f"session compute budget is exhausted; {capability!r} requires "
                    f"{spec.cost_class.value}"
                )
            if self.total_wall_time_s > self.budget.max_wall_time_s and (
                spec.cost_class is not CostClass.ZERO_SOLVE
            ):
                self._mark_budget_exhausted("wall-time ceiling exceeded")
                raise ResourceLimitError(
                    f"session wall time {self.total_wall_time_s:.1f}s exceeded "
                    f"max_wall_time_s={self.budget.max_wall_time_s:.1f}s"
                )

            validate_json_schema_args(spec.name, spec.input_schema, args)

            if self.state is HarnessSessionState.INITIALIZED:
                self.state = HARNESS_SESSION_MACHINE.require_transition(
                    self.state, HarnessSessionState.ACTIVE, trigger="invoke"
                ).to_state
            elif self.state is HarnessSessionState.ACTIVE:
                self.state = HARNESS_SESSION_MACHINE.require_transition(
                    self.state, HarnessSessionState.ACTIVE, trigger="invoke"
                ).to_state
            elif self.state is HarnessSessionState.BUDGET_EXHAUSTED:
                self.state = HARNESS_SESSION_MACHINE.require_transition(
                    self.state,
                    HarnessSessionState.BUDGET_EXHAUSTED,
                    trigger="invoke_free",
                ).to_state

            if dry_run:
                plan_doc: dict[str, Any] = {
                    "planned": True,
                    "capability": spec.full_descriptor(),
                    "validated_arguments": args,
                    "estimated_fem_solves": spec.default_fem_solves,
                    "budget_ok": (
                        self.fem_solves_used + spec.default_fem_solves
                        <= self.budget.max_fem_solves
                    ),
                }
                dt = time.perf_counter() - t0
                entry_sha = self._append_ledger(
                    capability=capability,
                    arguments=args,
                    dry_run=True,
                    ok=True,
                    error_code=None,
                    receipt_sha256=None,
                    artifact_paths=(),
                    wall_time_s=dt,
                )
                self._persist_session()
                return HarnessEnvelope(
                    ok=True,
                    capability=capability,
                    session_id=self.session_id,
                    invocation_index=self.invocation_count,
                    dry_run=True,
                    result=plan_doc,
                    error_code=None,
                    error_message=None,
                    recovery_hint="Dry-run validated; re-invoke with dry_run=False to execute.",
                    allowed_next_actions=(capability,),
                    state_snapshot=self.snapshot(),
                    budget=self.budget_status(),
                    wall_time_s=dt,
                    ledger_entry_sha256=entry_sha,
                )

            result = dict(handler(ctx, args))
            dt = time.perf_counter() - t0
            self.total_wall_time_s += dt
            receipt = ctx.receipt_sha256 or (
                str(result["result_hash"])
                if isinstance(result.get("result_hash"), str)
                else (
                    str(result["trust_receipt_sha256"])
                    if isinstance(result.get("trust_receipt_sha256"), str)
                    else None
                )
            )
            entry_sha = self._append_ledger(
                capability=capability,
                arguments=args,
                dry_run=False,
                ok=True,
                error_code=None,
                receipt_sha256=receipt,
                artifact_paths=ctx.emitted_artifacts,
                wall_time_s=dt,
            )
            self._persist_session()
            return HarnessEnvelope(
                ok=True,
                capability=capability,
                session_id=self.session_id,
                invocation_index=self.invocation_count,
                dry_run=False,
                result=result,
                error_code=None,
                error_message=None,
                recovery_hint="Capability succeeded.",
                allowed_next_actions=self._suggest_next(spec),
                state_snapshot=self.snapshot(),
                budget=self.budget_status(),
                receipt_sha256=receipt,
                artifact_paths=ctx.emitted_artifacts,
                wall_time_s=dt,
                ledger_entry_sha256=entry_sha,
            )

        except Exception as exc:
            dt = time.perf_counter() - t0
            self.total_wall_time_s += dt
            code, hint, next_actions = _error_recovery(exc)
            entry_sha = self._append_ledger(
                capability=capability,
                arguments=args,
                dry_run=dry_run,
                ok=False,
                error_code=code,
                receipt_sha256=None,
                artifact_paths=ctx.emitted_artifacts,
                wall_time_s=dt,
            )
            self._persist_session()
            return HarnessEnvelope(
                ok=False,
                capability=capability,
                session_id=self.session_id,
                invocation_index=self.invocation_count,
                dry_run=dry_run,
                result=None,
                error_code=code,
                error_message=str(exc),
                recovery_hint=hint,
                allowed_next_actions=next_actions,
                state_snapshot=self.snapshot(),
                budget=self.budget_status(),
                artifact_paths=ctx.emitted_artifacts,
                wall_time_s=dt,
                ledger_entry_sha256=entry_sha,
            )

    @staticmethod
    def _suggest_next(spec: CapabilitySpec) -> tuple[str, ...]:
        suggestions = {
            "cad.generate_sample_3d": (
                "fem.solve_3d_solid",
                "multiphysics.solve_modal_3d",
                "multiphysics.solve_thermo_mechanical_3d",
            ),
            "cad.import_3d": (
                "fem.solve_3d_solid",
                "multiphysics.solve_modal_3d",
            ),
            "fem.solve_3d_solid": (
                "export.vtu_3d",
                "export.glb_3d",
                "governance.verify_session_ledger",
            ),
            "cad.parametric_3d_sample": ("surrogate.train_parametric_3d",),
            "surrogate.train_parametric_3d": (
                "surrogate.qualify_parametric_3d",
                "surrogate.predict_parametric_3d",
                "optimize.design_parametric_3d",
            ),
        }
        return suggestions.get(spec.name, ("discover.capabilities",))

    def replay_into(
        self,
        target_workspace: Path | str,
        *,
        skip_failed: bool = True,
    ) -> list[HarnessEnvelope]:
        """Deterministically replay this session's ledger into a fresh workspace."""
        ok, errs = self.verify_ledger()
        if not ok:
            raise ValueError("cannot replay tampered ledger: " + "; ".join(errs))
        replayer = FormaMechAgentHarness(
            target_workspace,
            registry=self.registry,
            budget=self.budget,
        )
        envelopes: list[HarnessEnvelope] = []
        for entry in self.read_ledger():
            if skip_failed and not entry.get("ok", False):
                continue
            envelopes.append(
                replayer.invoke(
                    str(entry["capability"]),
                    entry.get("arguments", {}),
                    dry_run=bool(entry.get("dry_run", False)),
                )
            )
        return envelopes


__all__ = (
    "FormaMechAgentHarness",
    "HarnessContext",
)
