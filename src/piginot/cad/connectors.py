"""Live parametric CAD connector registry (OpenCASCADE, CadQuery, FreeCAD, Onshape) (V4-1)."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable
from urllib.parse import urlparse

CAD_CONNECTOR_GROUP: Final = "formamech.cad.connectors"
CAD_CONNECTOR_RESULT_SCHEMA: Final = "formamech.cad-connector-result-v1"
_SAFE_PARAM_KEY_RE: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ONSHAPE_MODEL_REF_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")
_SAFE_SUBPROCESS_ENV_KEYS: Final[tuple[str, ...]] = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "SYSTEMROOT",
    "COMSPEC",
    "TEMP",
    "TMP",
    "TMPDIR",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "PYTHONPATH",
)


def _validate_params(parameters: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in parameters.items():
        ks = str(k)
        fv = float(v)
        if not math.isfinite(fv):
            raise ValueError(f"CAD parameter {ks!r} must be finite, got {v!r}")
        out[ks] = fv
    return out


class CadConnectorUnavailableError(RuntimeError):
    """Raised when an optional CAD engine or remote API credential is unavailable."""


@dataclass(frozen=True, slots=True)
class CadConnectorResult:
    """Content-addressed result returned by any live parametric CAD connector."""

    connector_name: str
    template_or_model: str
    step_path: str
    step_sha256: str
    parameters: Mapping[str, float]
    metadata: Mapping[str, Any]

    def document(self) -> dict[str, Any]:
        return {
            "schema": CAD_CONNECTOR_RESULT_SCHEMA,
            "connector_name": self.connector_name,
            "template_or_model": self.template_or_model,
            "step_path": self.step_path,
            "step_sha256": self.step_sha256,
            "parameters": {k: float(v) for k, v in sorted(self.parameters.items())},
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class CadConnectorProtocol(Protocol):
    connector_name: str

    def is_available(self) -> bool: ...

    def generate_step(
        self,
        model_ref: str,
        parameters: Mapping[str, float],
        output_step: Path,
    ) -> CadConnectorResult: ...


class OpenCascadeTemplateConnector:
    """Built-in pure-MIT OpenCASCADE (Gmsh OCC) live parametric 3D STEP connector."""

    connector_name: str = "opencascade-builtin"

    def is_available(self) -> bool:
        return importlib.util.find_spec("gmsh") is not None

    def generate_step(
        self,
        model_ref: str,
        parameters: Mapping[str, float],
        output_step: Path,
    ) -> CadConnectorResult:
        from piginot.cad.importing_3d import (
            generate_3d_box_step,
            generate_3d_l_bracket_step,
            generate_3d_mounting_lug_step,
        )
        from piginot.cad.parametric_3d import _builtin_3d_generator

        dest = Path(output_step).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        params = _validate_params(parameters)

        if model_ref == "box":
            allowed = {"lx", "ly", "lz"}
            unknown = set(params) - allowed
            if unknown:
                raise ValueError(
                    f"unknown parameters for 'box' template: {sorted(unknown)}"
                )
            for k, v in params.items():
                if v <= 0.0:
                    raise ValueError(f"box parameter {k!r} must be positive")
            generate_3d_box_step(
                dest,
                lx=params.get("lx", 20.0),
                ly=params.get("ly", 8.0),
                lz=params.get("lz", 8.0),
            )
        elif model_ref == "l-bracket":
            allowed = {"arm_x", "arm_y", "thickness", "depth", "hole_radius"}
            unknown = set(params) - allowed
            if unknown:
                raise ValueError(
                    f"unknown parameters for 'l-bracket' template: {sorted(unknown)}"
                )
            for k, v in params.items():
                if v <= 0.0:
                    raise ValueError(f"l-bracket parameter {k!r} must be positive")
            generate_3d_l_bracket_step(dest, **params)
        elif model_ref == "mounting-lug":
            allowed = {
                "base_lx",
                "base_ly",
                "base_lz",
                "boss_height",
                "boss_radius",
                "boss_thickness",
                "bore_radius",
                "bolt_radius",
            }
            unknown = set(params) - allowed
            if unknown:
                raise ValueError(
                    f"unknown parameters for 'mounting-lug' template: {sorted(unknown)}"
                )
            for k, v in params.items():
                if v <= 0.0:
                    raise ValueError(f"mounting-lug parameter {k!r} must be positive")
            generate_3d_mounting_lug_step(dest, **params)
        else:
            _builtin_3d_generator(model_ref, params, dest)

        sha = hashlib.sha256(dest.read_bytes()).hexdigest()
        return CadConnectorResult(
            connector_name=self.connector_name,
            template_or_model=model_ref,
            step_path=str(dest),
            step_sha256=sha,
            parameters=params,
            metadata={"engine": "gmsh-opencascade"},
        )


class CadQueryConnector:
    """Optional CadQuery live Python script connector (fail-closed by default)."""

    connector_name: str = "cadquery"

    def __init__(
        self,
        *,
        allow_arbitrary_scripts: bool = False,
        allowed_root: Path | str | None = None,
    ) -> None:
        self.allow_arbitrary_scripts = bool(allow_arbitrary_scripts)
        self.allowed_root = (
            None if allowed_root is None else Path(allowed_root).resolve()
        )

    def is_available(self) -> bool:
        return (
            self.allow_arbitrary_scripts
            and importlib.util.find_spec("cadquery") is not None
        )

    def generate_step(
        self,
        model_ref: str,
        parameters: Mapping[str, float],
        output_step: Path,
    ) -> CadConnectorResult:
        if not self.allow_arbitrary_scripts:
            raise PermissionError(
                "CadQueryConnector executes Python scripts and is disabled by default; "
                "instantiate with allow_arbitrary_scripts=True (and allowed_root) to opt in."
            )
        if importlib.util.find_spec("cadquery") is None:
            raise CadConnectorUnavailableError(
                "CadQuery is not installed (`pip install cadquery`)."
            )
        script_path = Path(model_ref).resolve()
        if self.allowed_root is not None and not script_path.is_relative_to(
            self.allowed_root
        ):
            raise PermissionError(
                f"CadQuery script {script_path} escapes allowed_root {self.allowed_root}"
            )
        if not script_path.is_file():
            raise FileNotFoundError(f"CadQuery script not found: {script_path}")
        params = _validate_params(parameters)
        dest = Path(output_step).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".cq-", suffix=".step", dir=str(dest.parent)
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        tmp_path.unlink(missing_ok=True)
        try:
            ns: dict[str, Any] = {
                "PARAMETERS": dict(params),
                "OUTPUT_STEP": str(tmp_path),
            }
            exec(
                compile(
                    script_path.read_text(encoding="utf-8"), str(script_path), "exec"
                ),
                ns,
            )
            if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
                raise RuntimeError(f"CadQuery script did not emit STEP file {dest}")
            os.replace(tmp_path, dest)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return CadConnectorResult(
            connector_name=self.connector_name,
            template_or_model=str(script_path),
            step_path=str(dest),
            step_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
            parameters=params,
            metadata={"engine": "cadquery"},
        )


class FreeCadSubprocessConnector:
    """Optional headless FreeCADCmd subprocess STEP exporter with env scrubbing."""

    connector_name: str = "freecad-cli"

    def __init__(
        self,
        executable: str = "FreeCADCmd",
        timeout_s: float = 60.0,
        *,
        allowed_root: Path | str | None = None,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        self.executable = executable
        self.timeout_s = float(timeout_s)
        self.allowed_root = (
            None if allowed_root is None else Path(allowed_root).resolve()
        )

    def is_available(self) -> bool:
        return (
            shutil.which(self.executable) is not None or Path(self.executable).is_file()
        )

    def generate_step(
        self,
        model_ref: str,
        parameters: Mapping[str, float],
        output_step: Path,
    ) -> CadConnectorResult:
        exe = (
            self.executable
            if Path(self.executable).is_file()
            else shutil.which(self.executable)
        )
        if exe is None:
            raise CadConnectorUnavailableError(
                f"FreeCAD executable {self.executable!r} not found on PATH."
            )
        if not model_ref or model_ref.startswith("-"):
            raise ValueError(
                f"invalid FreeCAD model_ref {model_ref!r}; must be a file path, not a CLI flag"
            )
        model_path = Path(model_ref).resolve()
        if self.allowed_root is not None and not model_path.is_relative_to(
            self.allowed_root
        ):
            raise PermissionError(
                f"FreeCAD model {model_path} escapes allowed_root {self.allowed_root}"
            )
        if not model_path.is_file():
            raise FileNotFoundError(f"FreeCAD model/script not found: {model_path}")

        params = _validate_params(parameters)
        for k in params:
            if not _SAFE_PARAM_KEY_RE.match(k):
                raise ValueError(
                    f"invalid FreeCAD parameter key {k!r}; must match {_SAFE_PARAM_KEY_RE.pattern!r}"
                )

        dest = Path(output_step).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".fc-", suffix=".step", dir=str(dest.parent)
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        tmp_path.unlink(missing_ok=True)

        # Scrub parent environment to safe allowlist so secrets are never leaked to subprocess
        env = {k: os.environ[k] for k in _SAFE_SUBPROCESS_ENV_KEYS if k in os.environ}
        env["FORMAMECH_STEP_OUT"] = str(tmp_path)
        for k, v in params.items():
            env[f"FORMAMECH_PARAM_{k}"] = str(v)
        try:
            proc = subprocess.run(
                [exe, str(model_path)],
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
            if (
                proc.returncode != 0
                or not tmp_path.is_file()
                or tmp_path.stat().st_size == 0
            ):
                raise RuntimeError(
                    f"FreeCAD export failed (code={proc.returncode}): {proc.stderr.strip()}"
                )
            os.replace(tmp_path, dest)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return CadConnectorResult(
            connector_name=self.connector_name,
            template_or_model=str(model_path),
            step_path=str(dest),
            step_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
            parameters=params,
            metadata={"engine": "freecad-cli"},
        )


HttpFetchFn = Callable[[str, Mapping[str, str], Mapping[str, float]], bytes]


class OnshapeRestConnector:
    """Cloud CAD REST connector with injectable HTTP transport and explicit token guard."""

    connector_name: str = "onshape-rest"

    def __init__(
        self,
        *,
        access_key: str | None = None,
        secret_key: str | None = None,
        base_url: str = "https://cad.onshape.com/api/v6",
        http_fetch: HttpFetchFn | None = None,
    ) -> None:
        parsed_url = urlparse(base_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError(
                f"Onshape base_url must be a valid https:// URL, got {base_url!r}"
            )
        self.access_key = access_key or os.environ.get("ONSHAPE_ACCESS_KEY")
        self.secret_key = secret_key or os.environ.get("ONSHAPE_SECRET_KEY")
        self.base_url = base_url.rstrip("/")
        self._http_fetch = http_fetch

    def is_available(self) -> bool:
        return bool(
            self._http_fetch is not None and self.access_key and self.secret_key
        )

    def generate_step(
        self,
        model_ref: str,
        parameters: Mapping[str, float],
        output_step: Path,
    ) -> CadConnectorResult:
        if not (self.access_key and self.secret_key):
            raise CadConnectorUnavailableError(
                "Onshape API credentials missing; set ONSHAPE_ACCESS_KEY and ONSHAPE_SECRET_KEY."
            )
        if self._http_fetch is None:
            raise CadConnectorUnavailableError(
                "OnshapeRestConnector requires an explicit HTTP transport client in offline environments."
            )
        if (
            not isinstance(model_ref, str)
            or not _ONSHAPE_MODEL_REF_RE.match(model_ref)
            or ".." in model_ref
            or "//" in model_ref
        ):
            raise ValueError(
                f"invalid Onshape model_ref {model_ref!r}; must not contain traversal or query tokens"
            )
        params = _validate_params(parameters)
        dest = Path(output_step).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/parts/{model_ref}/export/step"
        token = base64.b64encode(
            f"{self.access_key}:{self.secret_key}".encode()
        ).decode("ascii")
        headers = {
            "Accept": "application/octet-stream",
            "Authorization": f"Basic {token}",
            "X-Onshape-Access-Key": self.access_key,
        }
        payload = self._http_fetch(url, headers, params)
        if not payload:
            raise RuntimeError("Onshape REST export returned empty STEP bytes")
        if b"ISO-10303-21" not in payload[:512]:
            raise RuntimeError(
                "Onshape REST export did not return a valid ISO-10303-21 STEP payload"
            )
        dest.write_bytes(payload)
        return CadConnectorResult(
            connector_name=self.connector_name,
            template_or_model=model_ref,
            step_path=str(dest),
            step_sha256=hashlib.sha256(payload).hexdigest(),
            parameters=params,
            metadata={"engine": "onshape-rest", "url": url},
        )


class CadConnectorRegistry:
    """Registry of live parametric CAD connectors (built-in + entry-point plugins)."""

    def __init__(self) -> None:
        self._connectors: dict[str, CadConnectorProtocol] = {
            "opencascade-builtin": OpenCascadeTemplateConnector(),
            "cadquery": CadQueryConnector(),
            "freecad-cli": FreeCadSubprocessConnector(),
            "onshape-rest": OnshapeRestConnector(),
        }
        for ep in entry_points(group=CAD_CONNECTOR_GROUP):
            obj = ep.load()
            inst = obj() if callable(obj) else obj
            if isinstance(inst, CadConnectorProtocol):
                self._connectors[ep.name] = inst

    def register(self, connector: CadConnectorProtocol) -> None:
        self._connectors[connector.connector_name] = connector

    def get(self, name: str) -> CadConnectorProtocol:
        if name not in self._connectors:
            raise KeyError(
                f"unknown CAD connector {name!r}; available: {sorted(self._connectors)}"
            )
        return self._connectors[name]

    def list_connectors(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "available": conn.is_available(),
            }
            for name, conn in sorted(self._connectors.items())
        ]


__all__ = (
    "CAD_CONNECTOR_GROUP",
    "CAD_CONNECTOR_RESULT_SCHEMA",
    "CadConnectorProtocol",
    "CadConnectorRegistry",
    "CadConnectorResult",
    "CadConnectorUnavailableError",
    "FreeCadSubprocessConnector",
    "OnshapeRestConnector",
    "OpenCascadeTemplateConnector",
)
