"""Append-only SHA-256 hash-chained production serving audit ledger (V4-1).

Records every served prediction, Trust Receipt digest, OOD refusal, and FEM
fallback event so operators can audit compliance and feed ``V4-5`` continual
qualification loops without leakage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from piginot.contracts.canonical import canonical_json_bytes, strict_json_loads
from piginot.contracts.errors import (
    ContractValidationError,
    DuplicateKeyError,
    NonFiniteNumberError,
)

AUDIT_LEDGER_SCHEMA: Final = "formamech.audit-ledger-entry-v1"


class AuditEventKind(StrEnum):
    PREDICT = "PREDICT"
    OOD_REFUSED = "OOD_REFUSED"
    FEM_FALLBACK = "FEM_FALLBACK"
    VERIFY_RECEIPT = "VERIFY_RECEIPT"
    HARNESS_INVOKE = "HARNESS_INVOKE"


def _sha256_doc(doc: object) -> str:
    return hashlib.sha256(canonical_json_bytes(doc)).hexdigest()


class SignedAuditLedger:
    """Append-only hash-chained JSONL ledger for verified production serving."""

    def __init__(self, ledger_path: Path | str) -> None:
        self.path = Path(ledger_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._last_size = -1
        self._prev_sha = self._recover_tail_sha()

    def _recover_tail_sha(self) -> str:
        if not self.path.is_file():
            self._last_size = 0
            return "0" * 64
        try:
            raw_bytes = self.path.read_bytes()
            self._last_size = len(raw_bytes)
            text = raw_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return "0" * 64
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return "0" * 64
        try:
            last = strict_json_loads(lines[-1])
        except (
            json.JSONDecodeError,
            ContractValidationError,
            DuplicateKeyError,
            NonFiniteNumberError,
        ):
            return "0" * 64
        if not isinstance(last, dict) or not isinstance(last.get("entry_sha256"), str):
            return "0" * 64
        return str(last["entry_sha256"])

    def _sync_tail_locked(self) -> None:
        if not self.path.is_file():
            self._last_size = 0
            self._prev_sha = "0" * 64
            return
        cur_size = self.path.stat().st_size
        if cur_size != self._last_size:
            self._prev_sha = self._recover_tail_sha()

    def append(
        self,
        *,
        event_kind: AuditEventKind,
        bundle_name: str,
        mode: str,
        verdict: str,
        inputs: Mapping[str, Any],
        trust_receipt_sha256: str | None = None,
        latency_s: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        lat = float(latency_s)
        if not math.isfinite(lat) or lat < 0.0:
            raise ValueError(
                f"latency_s must be finite and nonnegative, got {latency_s!r}"
            )
        with self._lock:
            self._sync_tail_locked()
            body = {
                "schema": AUDIT_LEDGER_SCHEMA,
                "timestamp_utc": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "event_kind": event_kind.value,
                "bundle_name": bundle_name,
                "mode": mode,
                "verdict": verdict,
                "inputs": dict(inputs),
                "trust_receipt_sha256": trust_receipt_sha256,
                "latency_s": round(lat, 6),
                "metadata": {} if metadata is None else dict(metadata),
                "prev_sha256": self._prev_sha,
            }
            entry_sha = _sha256_doc(body)
            record = {**body, "entry_sha256": entry_sha}
            line_bytes = canonical_json_bytes(record) + b"\n"
            with self.path.open("ab") as fh:
                fh.write(line_bytes)
                fh.flush()
                os.fsync(fh.fileno())
            self._prev_sha = entry_sha
            self._last_size = self.path.stat().st_size
            return record

    def read_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.path.is_file():
                return []
            out: list[dict[str, Any]] = []
            for ln in self.path.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                rec = strict_json_loads(ln)
                if not isinstance(rec, dict):
                    raise ContractValidationError(
                        "audit ledger line must be a JSON object"
                    )
                out.append(rec)
            return out

    def verify_chain(self) -> tuple[bool, tuple[str, ...]]:
        """Verify complete cryptographic hash-chain integrity of the audit ledger."""
        with self._lock:
            if not self.path.is_file():
                return True, ()
            try:
                text = self.path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                return False, (f"ledger unreadable or invalid UTF-8 ({exc})",)

            errors: list[str] = []
            prev = "0" * 64
            for idx, line in enumerate(
                (ln for ln in text.splitlines() if ln.strip()),
                start=1,
            ):
                try:
                    rec = strict_json_loads(line)
                except (
                    json.JSONDecodeError,
                    ContractValidationError,
                    DuplicateKeyError,
                    NonFiniteNumberError,
                ) as exc:
                    errors.append(f"line {idx}: invalid JSON ({exc})")
                    prev = ""
                    continue
                if not isinstance(rec, dict):
                    errors.append(f"line {idx}: record must be a JSON object")
                    prev = ""
                    continue
                if rec.get("prev_sha256") != prev:
                    errors.append(f"line {idx}: prev_sha256 mismatch")
                body = {k: v for k, v in rec.items() if k != "entry_sha256"}
                try:
                    expected = _sha256_doc(body)
                except (ContractValidationError, NonFiniteNumberError) as exc:
                    errors.append(f"line {idx}: non-canonical record ({exc})")
                    prev = ""
                    continue
                observed_sha = rec.get("entry_sha256")
                if not isinstance(observed_sha, str) or observed_sha != expected:
                    errors.append(f"line {idx}: entry_sha256 tampered")
                prev = observed_sha if isinstance(observed_sha, str) else ""
            return len(errors) == 0, tuple(errors)

    def query_fallbacks(self) -> list[dict[str, Any]]:
        """Return all OOD refusal and FEM fallback entries for continual learning."""
        with self._lock:
            if not self.path.is_file():
                return []
            fallbacks: list[dict[str, Any]] = []
            try:
                text = self.path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return []
            for ln in text.splitlines():
                if not ln.strip():
                    continue
                try:
                    entry = strict_json_loads(ln)
                except (
                    json.JSONDecodeError,
                    ContractValidationError,
                    DuplicateKeyError,
                    NonFiniteNumberError,
                ):
                    continue
                if isinstance(entry, dict) and entry.get("event_kind") in (
                    AuditEventKind.OOD_REFUSED.value,
                    AuditEventKind.FEM_FALLBACK.value,
                ):
                    fallbacks.append(entry)
            return fallbacks

    def metrics_summary(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.is_file():
                return {
                    "total_events": 0,
                    "ood_refusals": 0,
                    "fem_fallbacks": 0,
                    "latency_seconds_sum": 0.0,
                    "corrupt_lines": 0,
                }
            total = 0
            ood = 0
            fb = 0
            lat_sum = 0.0
            corrupt = 0
            try:
                text = self.path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return {
                    "total_events": 0,
                    "ood_refusals": 0,
                    "fem_fallbacks": 0,
                    "latency_seconds_sum": 0.0,
                    "corrupt_lines": 1,
                }
            for ln in text.splitlines():
                if not ln.strip():
                    continue
                try:
                    e = strict_json_loads(ln)
                except (
                    json.JSONDecodeError,
                    ContractValidationError,
                    DuplicateKeyError,
                    NonFiniteNumberError,
                ):
                    corrupt += 1
                    continue
                if not isinstance(e, dict):
                    corrupt += 1
                    continue
                total += 1
                kind = e.get("event_kind")
                if kind == AuditEventKind.OOD_REFUSED.value:
                    ood += 1
                elif kind == AuditEventKind.FEM_FALLBACK.value:
                    fb += 1
                raw_lat = e.get("latency_s", 0.0)
                if isinstance(raw_lat, (int, float)) and not isinstance(raw_lat, bool):
                    fv = float(raw_lat)
                    if math.isfinite(fv) and fv >= 0.0:
                        lat_sum += fv
            return {
                "total_events": total,
                "ood_refusals": ood,
                "fem_fallbacks": fb,
                "latency_seconds_sum": round(lat_sum, 6),
                "corrupt_lines": corrupt,
            }


__all__ = (
    "AUDIT_LEDGER_SCHEMA",
    "AuditEventKind",
    "SignedAuditLedger",
)
