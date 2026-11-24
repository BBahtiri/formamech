"""Zero-extra-dependency production verified HTTP/JSON & Prometheus serving API (V4-1)."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

from piginot import __version__
from piginot.certify import verify_receipt_document
from piginot.harness import FormaMechAgentHarness, HarnessBudget
from piginot.serving.audit_ledger import AuditEventKind, SignedAuditLedger


class FormaMechApiApplication:
    """Framework-agnostic HTTP router backing ``formamech serve-api``."""

    def __init__(
        self,
        workspace_root: Path | str,
        *,
        bundle_path: Path | str | None = None,
        ledger_path: Path | str | None = None,
        budget: HarnessBudget | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.bundle_path = None if bundle_path is None else Path(bundle_path).resolve()
        lpath = (
            self.workspace_root / "audit-ledger.jsonl"
            if ledger_path is None
            else Path(ledger_path).resolve()
        )
        self.ledger = SignedAuditLedger(lpath)
        self.harness = FormaMechAgentHarness(
            self.workspace_root,
            budget=budget,
        )
        self._bundle_cache: Any = None
        self._recipe_cache: Any = None

    def _load_configured_bundle(self) -> tuple[Any, Any]:
        if self.bundle_path is None:
            raise ValueError("no default bundle configured on this serving instance")
        if self._bundle_cache is None:
            from piginot.bundle import load_bundle
            from piginot.bundle.problem import PortableProblem
            from piginot.serving import load_recipe

            bundle = load_bundle(self.bundle_path)
            is_portable = isinstance(
                getattr(bundle.model, "problem", None), PortableProblem
            )
            recipe = None if is_portable else load_recipe(bundle, self.bundle_path)
            self._bundle_cache = bundle
            self._recipe_cache = recipe
        return self._bundle_cache, self._recipe_cache

    def prometheus_metrics_text(self) -> str:
        m = self.ledger.metrics_summary()
        return "\n".join(
            [
                "# HELP formamech_requests_total Total audited FormaMech API events.",
                "# TYPE formamech_requests_total counter",
                f"formamech_requests_total {m['total_events']}",
                "# HELP formamech_ood_refusals_total Total OOD/envelope refusals.",
                "# TYPE formamech_ood_refusals_total counter",
                f"formamech_ood_refusals_total {m['ood_refusals']}",
                "# HELP formamech_fem_fallbacks_total Total FEM fallback escalations.",
                "# TYPE formamech_fem_fallbacks_total counter",
                f"formamech_fem_fallbacks_total {m['fem_fallbacks']}",
                "# HELP formamech_latency_seconds_sum Cumulative serving latency in seconds.",
                "# TYPE formamech_latency_seconds_sum counter",
                f"formamech_latency_seconds_sum {m['latency_seconds_sum']}",
                "",
            ]
        )

    def handle_request(
        self, method: str, path: str, body: bytes = b""
    ) -> tuple[int, str, bytes]:
        """Dispatch one HTTP request and return ``(status_code, content_type, body_bytes)``."""
        m = method.upper()
        clean_path = path.split("?", 1)[0]

        def _json(status: int, payload: Mapping[str, Any]) -> tuple[int, str, bytes]:
            return (
                status,
                "application/json",
                (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode(
                    "utf-8"
                ),
            )

        if m == "GET" and clean_path == "/health":
            return _json(
                200,
                {
                    "status": "ok",
                    "version": __version__,
                    "bundle": None
                    if self.bundle_path is None
                    else self.bundle_path.name,
                    "session_id": self.harness.session_id,
                },
            )

        if m == "GET" and clean_path == "/metrics":
            return (
                200,
                "text/plain; version=0.0.4; charset=utf-8",
                self.prometheus_metrics_text().encode("utf-8"),
            )

        if m == "GET" and clean_path == "/v1/capabilities":
            return _json(
                200,
                {
                    "count": len(self.harness.registry),
                    "domains": self.harness.registry.domains_summary(),
                    "capabilities": self.harness.registry.list_capabilities(),
                },
            )

        if m == "GET" and clean_path == "/v1/ledger/verify":
            ok, errs = self.ledger.verify_chain()
            return _json(
                200 if ok else 409,
                {
                    "verified": ok,
                    "errors": list(errs),
                    "metrics": self.ledger.metrics_summary(),
                },
            )

        if m == "POST" and clean_path == "/v1/verify-receipt":
            try:
                req = json.loads(body.decode("utf-8"))
                receipt = req.get("receipt", req)
                ok, errs = verify_receipt_document(receipt)
                self.ledger.append(
                    event_kind=AuditEventKind.VERIFY_RECEIPT,
                    bundle_name=str(receipt.get("bundle_name", "unknown")),
                    mode="verify",
                    verdict="VERIFIED" if ok else "INVALID",
                    inputs={},
                    trust_receipt_sha256=receipt.get("receipt_sha256"),
                )
                return _json(200, {"verified": ok, "errors": list(errs)})
            except Exception as exc:
                return _json(400, {"error": str(exc)})

        if m == "POST" and clean_path == "/v1/predict":
            t0 = time.perf_counter()
            try:
                from piginot.cli import parse_request
                from piginot.serving import ServingMode, serve

                req = json.loads(body.decode("utf-8"))
                raw_inputs = req["inputs"]
                mode = ServingMode(str(req.get("mode", "certified")))
                bundle, recipe = self._load_configured_bundle()
                parsed_inputs, parsed_queries = parse_request(
                    {"inputs": dict(raw_inputs)}
                )
                served = serve(
                    bundle,
                    parsed_inputs,
                    cast("Sequence[Sequence[float]] | None", parsed_queries),
                    mode=mode,
                    recipe=recipe if mode is not ServingMode.FAST else None,
                    corrected=mode is not ServingMode.FAST,
                    escalate=bool(req.get("escalate", False)),
                    emit_receipt=True,
                )
                doc = served.document()
                verdict = str(doc["prediction"]["verdict"])
                serving_path = str(doc["path"])
                if verdict == "REFUSED":
                    ev_kind = AuditEventKind.OOD_REFUSED
                elif "fem" in serving_path.lower() or mode is ServingMode.EXACT:
                    ev_kind = AuditEventKind.FEM_FALLBACK
                else:
                    ev_kind = AuditEventKind.PREDICT
                dt = time.perf_counter() - t0
                entry = self.ledger.append(
                    event_kind=ev_kind,
                    bundle_name=bundle.manifest.name,
                    mode=mode.value,
                    verdict=verdict,
                    inputs=raw_inputs,
                    trust_receipt_sha256=doc.get("trust_receipt_sha256"),
                    latency_s=dt,
                    metadata={"serving_path": serving_path},
                )
                return _json(
                    200,
                    {
                        "mode": doc["mode"],
                        "serving_path": serving_path,
                        "verdict": verdict,
                        "qois": doc["prediction"].get("qois", []),
                        "trust_receipt": doc.get("trust_receipt"),
                        "trust_receipt_sha256": doc.get("trust_receipt_sha256"),
                        "audit_entry_sha256": entry["entry_sha256"],
                    },
                )
            except Exception as exc:
                return _json(400, {"error": type(exc).__name__, "message": str(exc)})

        if m == "POST" and clean_path == "/v1/harness/invoke":
            t0 = time.perf_counter()
            try:
                req = json.loads(body.decode("utf-8"))
                cap = str(req["capability"])
                cap_args = req.get("arguments", {})
                dry_run = bool(req.get("dry_run", False))
                env = self.harness.invoke(cap, cap_args, dry_run=dry_run)
                self.ledger.append(
                    event_kind=AuditEventKind.HARNESS_INVOKE,
                    bundle_name=cap,
                    mode="dry_run" if dry_run else "live",
                    verdict="OK" if env.ok else str(env.error_code),
                    inputs=cap_args if isinstance(cap_args, Mapping) else {},
                    trust_receipt_sha256=env.receipt_sha256,
                    latency_s=time.perf_counter() - t0,
                )
                return _json(200 if env.ok else 422, env.document())
            except Exception as exc:
                return _json(400, {"error": type(exc).__name__, "message": str(exc)})

        return _json(404, {"error": "not_found", "path": clean_path})


class FormaMechHttpServer:
    """Managed threaded HTTP server wrapping :class:`FormaMechApiApplication`."""

    def __init__(
        self,
        app: FormaMechApiApplication,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        self.app = app

        class _Handler(BaseHTTPRequestHandler):
            def _dispatch(self) -> None:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length > 0 else b""
                code, ctype, resp = app.handle_request(self.command, self.path, raw)
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

            def do_GET(self) -> None:  # noqa: N802
                self._dispatch()

            def do_POST(self) -> None:  # noqa: N802
                self._dispatch()

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                del format, args

        self._httpd = ThreadingHTTPServer((host, port), _Handler)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._httpd.server_address[:2]
        return str(host), int(port)

    @property
    def base_url(self) -> str:
        host, port = self.address
        return f"http://{host}:{port}"

    def start_background(self) -> None:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def serve_forever(self) -> None:
        self._httpd.serve_forever()

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


__all__ = (
    "FormaMechApiApplication",
    "FormaMechHttpServer",
)
