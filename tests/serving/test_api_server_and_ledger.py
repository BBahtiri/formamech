"""Verification tests for Phase 1 (V4-1) HTTP/JSON API server & SignedAuditLedger."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

from piginot.serving.api_server import FormaMechApiApplication, FormaMechHttpServer
from piginot.serving.audit_ledger import AuditEventKind, SignedAuditLedger

ROOT = Path(__file__).resolve().parents[2]
NOTCHED_BUNDLE = ROOT / "bundles" / "notched-v1-cp-freeze"
NOTCHED_REQ = ROOT / "bundles" / "requests" / "notched-in-domain.json"
NOTCHED_OOD = ROOT / "bundles" / "requests" / "notched-out-of-envelope.json"


def _http_json(
    url: str, payload: dict[str, object] | None = None
) -> tuple[int, dict[str, object]]:
    if payload is None:
        req = Request(url, method="GET")
    else:
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urlopen(req, timeout=20) as resp:
        return int(resp.status), json.loads(resp.read().decode("utf-8"))


def test_http_serving_api_predict_receipt_metrics_harness_and_ledger_tamper(
    tmp_path: Path,
) -> None:
    app = FormaMechApiApplication(
        tmp_path / "api_ws",
        bundle_path=NOTCHED_BUNDLE,
    )
    server = FormaMechHttpServer(app, host="127.0.0.1", port=0)
    server.start_background()
    try:
        base = server.base_url

        # 1) /health & /v1/capabilities
        st_h, health = _http_json(f"{base}/health")
        assert st_h == 200 and health["status"] == "ok"
        st_c, caps = _http_json(f"{base}/v1/capabilities")
        assert st_c == 200 and int(caps["count"]) >= 15  # type: ignore[arg-type]

        # 2) POST /v1/predict (in-domain certified -> verified Trust Receipt)
        in_inputs = json.loads(NOTCHED_REQ.read_text(encoding="utf-8"))["inputs"]
        st_p, pred = _http_json(
            f"{base}/v1/predict", {"inputs": in_inputs, "mode": "certified"}
        )
        assert st_p == 200
        assert pred["verdict"] == "IN_DOMAIN"
        assert pred["trust_receipt"] is not None

        # 3) POST /v1/verify-receipt confirms Trust Receipt
        st_v, ver = _http_json(
            f"{base}/v1/verify-receipt", {"receipt": pred["trust_receipt"]}
        )
        assert st_v == 200 and ver["verified"] is True

        # 4) Out-of-envelope request logs OOD_REFUSED in audit ledger & Prometheus metrics
        ood_inputs = json.loads(NOTCHED_OOD.read_text(encoding="utf-8"))["inputs"]
        st_o, ood = _http_json(
            f"{base}/v1/predict", {"inputs": ood_inputs, "mode": "fast"}
        )
        assert st_o == 200 and ood["verdict"] == "REFUSED"

        # 5) POST /v1/harness/invoke runs capabilities through the server harness
        st_hi, hi = _http_json(
            f"{base}/v1/harness/invoke",
            {"capability": "discover.backends", "arguments": {}},
        )
        assert st_hi == 200 and hi["ok"] is True

        # 6) GET /metrics returns Prometheus counters reflecting OOD refusal
        with urlopen(f"{base}/metrics", timeout=10) as mresp:
            metrics_txt = mresp.read().decode("utf-8")
        assert "formamech_requests_total 4" in metrics_txt
        assert "formamech_ood_refusals_total 1" in metrics_txt

        # 7) GET /v1/ledger/verify confirms intact SHA-256 chain
        st_l, led = _http_json(f"{base}/v1/ledger/verify")
        assert st_l == 200 and led["verified"] is True
    finally:
        server.shutdown()

    # 8) Query fallbacks & detect single-byte tampering on SignedAuditLedger
    ledger = SignedAuditLedger(app.ledger.path)
    fallbacks = ledger.query_fallbacks()
    assert len(fallbacks) == 1
    assert fallbacks[0]["event_kind"] == AuditEventKind.OOD_REFUSED.value
    ok_clean, _ = ledger.verify_chain()
    assert ok_clean is True

    lines = ledger.path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("IN_DOMAIN", "TAMPERED")
    ledger.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok_tampered, errs = ledger.verify_chain()
    assert ok_tampered is False and errs


def test_v4_1_api_server_hardening_auth_limits_strict_json_and_corrupt_ledger(
    tmp_path: Path,
) -> None:
    """API rejects NaN JSON, oversized bodies, missing auth, disabled harness, and corrupt ledgers cleanly."""
    app = FormaMechApiApplication(
        tmp_path / "ws_hardened",
        auth_token="secret-token-42",
        enable_harness_endpoint=False,
        max_body_bytes=2048,
    )
    # 1) /health open for liveness, protected endpoints require Bearer token
    st_h, _, _ = app.handle_request("GET", "/health")
    assert st_h == 200
    st_unauth, _, _ = app.handle_request("GET", "/v1/capabilities")
    assert st_unauth == 401
    hdr = {"Authorization": "Bearer secret-token-42"}
    st_auth, _, _ = app.handle_request("GET", "/v1/capabilities", headers=hdr)
    assert st_auth == 200

    # 2) Disabled /v1/harness/invoke returns 403
    st_forb, _, _ = app.handle_request(
        "POST",
        "/v1/harness/invoke",
        body=b'{"capability":"discover.capabilities","arguments":{}}',
        headers=hdr,
    )
    assert st_forb == 403

    # 3) Oversized body returns 413
    st_big, _, _ = app.handle_request(
        "POST", "/v1/verify-receipt", body=b"{" + b" " * 3000 + b"}", headers=hdr
    )
    assert st_big == 413

    # 4) Non-finite NaN JSON rejected with 400
    st_nan, _, _ = app.handle_request(
        "POST",
        "/v1/predict",
        body=b'{"inputs":{"x":NaN},"mode":"fast"}',
        headers=hdr,
    )
    assert st_nan == 400

    # 5) Corrupt JSON line in audit ledger returns 409 (never unhandled 500)
    app.ledger.path.write_text("{corrupt-json-line\n", encoding="utf-8")
    st_ver, _, body_ver = app.handle_request("GET", "/v1/ledger/verify", headers=hdr)
    assert st_ver == 409
    parsed = json.loads(body_ver.decode("utf-8"))
    assert parsed["verified"] is False
    assert parsed["metrics"]["corrupt_lines"] == 1
