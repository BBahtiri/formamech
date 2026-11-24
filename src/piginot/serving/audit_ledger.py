"""Append-only SHA-256 hash-chained production serving audit ledger (V4-1).

Records every served prediction, Trust Receipt digest, OOD refusal, and FEM
fallback event so operators can audit compliance and feed ``V4-5`` continual
qualification loops without leakage.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from piginot.contracts.canonical import canonical_json_bytes

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
        self._prev_sha = self._recover_tail_sha()

    def _recover_tail_sha(self) -> str:
        if not self.path.is_file():
            return "0" * 64
        lines = [
            ln
            for ln in self.path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        if not lines:
            return "0" * 64
        last = json.loads(lines[-1])
        return str(last.get("entry_sha256", "0" * 64))

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
        body = {
            "schema": AUDIT_LEDGER_SCHEMA,
            "timestamp_utc": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event_kind": event_kind.value,
            "bundle_name": bundle_name,
            "mode": mode,
            "verdict": verdict,
            "inputs": dict(inputs),
            "trust_receipt_sha256": trust_receipt_sha256,
            "latency_s": round(float(latency_s), 6),
            "metadata": {} if metadata is None else dict(metadata),
            "prev_sha256": self._prev_sha,
        }
        entry_sha = _sha256_doc(body)
        record = {**body, "entry_sha256": entry_sha}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._prev_sha = entry_sha
        return record

    def read_entries(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [
            json.loads(ln)
            for ln in self.path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    def verify_chain(self) -> tuple[bool, tuple[str, ...]]:
        """Verify complete cryptographic hash-chain integrity of the audit ledger."""
        if not self.path.is_file():
            return True, ()
        errors: list[str] = []
        prev = "0" * 64
        for idx, line in enumerate(
            (
                ln
                for ln in self.path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ),
            start=1,
        ):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {idx}: invalid JSON ({exc})")
                continue
            if rec.get("prev_sha256") != prev:
                errors.append(f"line {idx}: prev_sha256 mismatch")
            body = {k: v for k, v in rec.items() if k != "entry_sha256"}
            expected = _sha256_doc(body)
            if rec.get("entry_sha256") != expected:
                errors.append(f"line {idx}: entry_sha256 tampered")
            prev = str(rec.get("entry_sha256", ""))
        return len(errors) == 0, tuple(errors)

    def query_fallbacks(self) -> list[dict[str, Any]]:
        """Return all OOD refusal and FEM fallback entries for continual learning."""
        return [
            entry
            for entry in self.read_entries()
            if entry.get("event_kind")
            in (AuditEventKind.OOD_REFUSED.value, AuditEventKind.FEM_FALLBACK.value)
        ]

    def metrics_summary(self) -> dict[str, Any]:
        entries = self.read_entries()
        total = len(entries)
        ood = sum(
            1
            for e in entries
            if e.get("event_kind") == AuditEventKind.OOD_REFUSED.value
        )
        fb = sum(
            1
            for e in entries
            if e.get("event_kind") == AuditEventKind.FEM_FALLBACK.value
        )
        lat_sum = sum(float(e.get("latency_s", 0.0)) for e in entries)
        return {
            "total_events": total,
            "ood_refusals": ood,
            "fem_fallbacks": fb,
            "latency_seconds_sum": round(lat_sum, 6),
        }


__all__ = (
    "AUDIT_LEDGER_SCHEMA",
    "AuditEventKind",
    "SignedAuditLedger",
)
