"""Zero-extra-dependency production verified HTTP/JSON & Prometheus serving API (V4-1)."""

from __future__ import annotations

import hmac
import json
import threading
import time
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

from piginot import __version__
from piginot.certify import verify_receipt_document
from piginot.contracts.canonical import strict_json_loads
from piginot.contracts.errors import (
    ContractValidationError,
    DuplicateKeyError,
    NonFiniteNumberError,
)
from piginot.harness import FormaMechAgentHarness, HarnessBudget
from piginot.serving.audit_ledger import AuditEventKind, SignedAuditLedger

DEFAULT_MAX_BODY_BYTES: int = 4 * 1024 * 1024  # 4 MiB


class FormaMechApiApplication:
    """Framework-agnostic HTTP router backing ``formamech serve-api``."""

    def __init__(
        self,
        workspace_root: Path | str,
        *,
        bundle_path: Path | str | None = None,
        ledger_path: Path | str | None = None,
        budget: HarnessBudget | None = None,
        auth_token: str | None = None,
        enable_harness_endpoint: bool = True,
        max_body_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if type(max_body_bytes) is not int or max_body_bytes < 1024:
            raise ValueError("max_body_bytes must be an integer >= 1024")
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
        self.auth_token = auth_token
        self.enable_harness_endpoint = bool(enable_harness_endpoint)
        self.max_body_bytes = max_body_bytes
        self._bundle_lock = threading.RLock()
        self._harness_lock = threading.RLock()
        self._loaded_bundle_pair: tuple[Any, Any] | None = None

    def _load_configured_bundle(self) -> tuple[Any, Any]:
        if self.bundle_path is None:
            raise ValueError("no default bundle configured on this serving instance")
        pair = self._loaded_bundle_pair
        if pair is not None:
            return pair
        with self._bundle_lock:
            if self._loaded_bundle_pair is None:
                from piginot.bundle import load_bundle
                from piginot.bundle.problem import PortableProblem
                from piginot.serving import load_recipe

                bundle = load_bundle(self.bundle_path)
                is_portable = isinstance(
                    getattr(bundle.model, "problem", None), PortableProblem
                )
                recipe = None if is_portable else load_recipe(bundle, self.bundle_path)
                self._loaded_bundle_pair = (bundle, recipe)
            return self._loaded_bundle_pair

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

    def _check_auth(self, headers: Mapping[str, str] | None) -> bool:
        if self.auth_token is None:
            return True
        if not headers:
            return False
        auth_val = ""
        for k, v in headers.items():
            if k.lower() == "authorization":
                auth_val = v.strip()
                break
        if not auth_val.startswith("Bearer "):
            return False
        supplied = auth_val[len("Bearer ") :].strip()
        return hmac.compare_digest(supplied, self.auth_token)

    def _parse_json_object(self, body: bytes) -> dict[str, Any]:
        parsed = strict_json_loads(body)
        if not isinstance(parsed, dict):
            raise ValueError("JSON request body must be an object")
        return parsed

    def handle_request(
        self,
        method: str,
        path: str,
        body: bytes = b"",
        *,
        headers: Mapping[str, str] | None = None,
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

        if len(body) > self.max_body_bytes:
            return _json(
                413,
                {
                    "error": "payload_too_large",
                    "max_body_bytes": self.max_body_bytes,
                },
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

        if not self._check_auth(headers):
            return _json(401, {"error": "unauthorized"})

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
                req = self._parse_json_object(body)
                receipt = req.get("receipt", req)
                if not isinstance(receipt, dict):
                    return _json(
                        400,
                        {
                            "error": "ValueError",
                            "message": "receipt must be a JSON object",
                        },
                    )
                ok, errs = verify_receipt_document(receipt)
                sha_val = receipt.get("receipt_sha256")
                self.ledger.append(
                    event_kind=AuditEventKind.VERIFY_RECEIPT,
                    bundle_name=str(receipt.get("bundle_name", "unknown")),
                    mode="verify",
                    verdict="VERIFIED" if ok else "INVALID",
                    inputs={},
                    trust_receipt_sha256=sha_val if isinstance(sha_val, str) else None,
                )
                return _json(200, {"verified": ok, "errors": list(errs)})
            except (
                json.JSONDecodeError,
                ContractValidationError,
                DuplicateKeyError,
                NonFiniteNumberError,
                ValueError,
                TypeError,
                KeyError,
            ) as exc:
                return _json(400, {"error": str(exc)})
            except Exception as exc:
                return _json(400, {"error": str(exc)})

        if m == "POST" and clean_path == "/v1/predict":
            t0 = time.perf_counter()
            try:
                from piginot.cli import parse_request
                from piginot.serving import ServingMode, ServingPath, serve

                req = self._parse_json_object(body)
                raw_inputs = req.get("inputs")
                if not isinstance(raw_inputs, dict):
                    raise ValueError("'inputs' must be a JSON object")
                escalate_raw = req.get("escalate", False)
                if not isinstance(escalate_raw, bool):
                    raise ValueError("'escalate' must be a boolean")
                mode = ServingMode(str(req.get("mode", "certified")))
                bundle, recipe = self._load_configured_bundle()
                parse_payload: dict[str, Any] = {"inputs": dict(raw_inputs)}
                if "queries" in req and req["queries"] is not None:
                    parse_payload["queries"] = req["queries"]
                parsed_inputs, parsed_queries = parse_request(parse_payload)
                served = serve(
                    bundle,
                    parsed_inputs,
                    cast("Sequence[Sequence[float]] | None", parsed_queries),
                    mode=mode,
                    recipe=recipe if mode is not ServingMode.FAST else None,
                    corrected=mode is not ServingMode.FAST,
                    escalate=escalate_raw,
                    emit_receipt=True,
                )
                doc = served.document()
                verdict = str(doc["prediction"]["verdict"])
                serving_path = str(doc["path"])
                fem_paths = {
                    ServingPath.WARM_NEWTON.value,
                    ServingPath.REPAIRED_WARM_NEWTON.value,
                    ServingPath.CONTINUATION.value,
                    ServingPath.COLD_FEM.value,
                    ServingPath.SOLVE_FAILED.value,
                }
                if verdict == "REFUSED" or serving_path == ServingPath.REFUSED.value:
                    ev_kind = AuditEventKind.OOD_REFUSED
                elif serving_path in fem_paths or mode is ServingMode.EXACT:
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
            if not self.enable_harness_endpoint:
                return _json(403, {"error": "harness_endpoint_disabled"})
            t0 = time.perf_counter()
            try:
                req = self._parse_json_object(body)
                cap_raw = req.get("capability")
                if not isinstance(cap_raw, str) or not cap_raw.strip():
                    raise ValueError("'capability' must be a non-empty string")
                cap_args = req.get("arguments", {})
                if not isinstance(cap_args, dict):
                    raise ValueError("'arguments' must be a JSON object")
                dry_run = req.get("dry_run", False)
                if not isinstance(dry_run, bool):
                    raise ValueError("'dry_run' must be a boolean")
                with self._harness_lock:
                    env = self.harness.invoke(cap_raw, cap_args, dry_run=dry_run)
                self.ledger.append(
                    event_kind=AuditEventKind.HARNESS_INVOKE,
                    bundle_name=cap_raw,
                    mode="dry_run" if dry_run else "live",
                    verdict="OK" if env.ok else str(env.error_code),
                    inputs=cap_args,
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
        request_timeout_s: float = 30.0,
    ) -> None:
        self.app = app

        class _Handler(BaseHTTPRequestHandler):
            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(request_timeout_s)

            def _send_bytes(self, code: int, ctype: str, resp: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

            def _dispatch(self) -> None:
                te = (self.headers.get("Transfer-Encoding") or "").lower()
                raw_cl = self.headers.get("Content-Length")
                if "chunked" in te and raw_cl is None:
                    err = b'{"error":"length_required"}\n'
                    self._send_bytes(411, "application/json", err)
                    return
                if raw_cl is None or not raw_cl.strip():
                    length = 0
                else:
                    try:
                        length = int(raw_cl.strip())
                    except ValueError:
                        err = b'{"error":"invalid_content_length"}\n'
                        self._send_bytes(400, "application/json", err)
                        return
                    if length < 0:
                        err = b'{"error":"invalid_content_length"}\n'
                        self._send_bytes(400, "application/json", err)
                        return
                    if length > app.max_body_bytes:
                        err = (
                            json.dumps(
                                {
                                    "error": "payload_too_large",
                                    "max_body_bytes": app.max_body_bytes,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        ).encode("utf-8")
                        self._send_bytes(413, "application/json", err)
                        return
                raw = self.rfile.read(length) if length > 0 else b""
                hdrs = {str(k): str(v) for k, v in self.headers.items()}
                code, ctype, resp = app.handle_request(
                    self.command, self.path, raw, headers=hdrs
                )
                self._send_bytes(code, ctype, resp)

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
