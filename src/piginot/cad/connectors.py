"""Live parametric CAD connector registry (OpenCASCADE, CadQuery, FreeCAD, Onshape) (V4-1)."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

CAD_CONNECTOR_GROUP: Final = "formamech.cad.connectors"
CAD_CONNECTOR_RESULT_SCHEMA: Final = "formamech.cad-connector-result-v1"


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
        params = {str(k): float(v) for k, v in parameters.items()}

        if model_ref == "box":
            generate_3d_box_step(
                dest,
                lx=params.get("lx", 20.0),
                ly=params.get("ly", 8.0),
                lz=params.get("lz", 8.0),
            )
        elif model_ref == "l-bracket":
            generate_3d_l_bracket_step(dest)
        elif model_ref == "mounting-lug":
            generate_3d_mounting_lug_step(dest)
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
    """Optional CadQuery live Python script connector (fail-closed when uninstalled)."""

    connector_name: str = "cadquery"

    def is_available(self) -> bool:
        return importlib.util.find_spec("cadquery") is not None

    def generate_step(
        self,
        model_ref: str,
        parameters: Mapping[str, float],
        output_step: Path,
    ) -> CadConnectorResult:
        if not self.is_available():
            raise CadConnectorUnavailableError(
                "CadQuery is not installed (`pip install cadquery`)."
            )
        script_path = Path(model_ref).resolve()
        if not script_path.is_file():
            raise FileNotFoundError(f"CadQuery script not found: {script_path}")
        dest = Path(output_step).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        ns: dict[str, Any] = {
            "PARAMETERS": {str(k): float(v) for k, v in parameters.items()},
            "OUTPUT_STEP": str(dest),
        }
        exec(
            compile(script_path.read_text(encoding="utf-8"), str(script_path), "exec"),
            ns,
        )
        if not dest.is_file():
            raise RuntimeError(f"CadQuery script did not emit STEP file {dest}")
        return CadConnectorResult(
            connector_name=self.connector_name,
            template_or_model=str(script_path),
            step_path=str(dest),
            step_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
            parameters={str(k): float(v) for k, v in parameters.items()},
            metadata={"engine": "cadquery"},
        )


class FreeCadSubprocessConnector:
    """Optional headless FreeCADCmd subprocess STEP exporter."""

    connector_name: str = "freecad-cli"

    def __init__(self, executable: str = "FreeCADCmd", timeout_s: float = 60.0) -> None:
        self.executable = executable
        self.timeout_s = timeout_s

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
        dest = Path(output_step).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "FORMAMECH_STEP_OUT": str(dest)}
        for k, v in parameters.items():
            env[f"FORMAMECH_PARAM_{k}"] = str(float(v))
        proc = subprocess.run(
            [exe, model_ref],
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
        )
        if proc.returncode != 0 or not dest.is_file():
            raise RuntimeError(
                f"FreeCAD export failed (code={proc.returncode}): {proc.stderr.strip()}"
            )
        return CadConnectorResult(
            connector_name=self.connector_name,
            template_or_model=model_ref,
            step_path=str(dest),
            step_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
            parameters={str(k): float(v) for k, v in parameters.items()},
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
        self.access_key = access_key or os.environ.get("ONSHAPE_ACCESS_KEY")
        self.secret_key = secret_key or os.environ.get("ONSHAPE_SECRET_KEY")
        self.base_url = base_url.rstrip("/")
        self._http_fetch = http_fetch

    def is_available(self) -> bool:
        return bool(
            self._http_fetch is not None or (self.access_key and self.secret_key)
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
        dest = Path(output_step).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/parts/{model_ref}/export/step"
        headers = {
            "Accept": "application/octet-stream",
            "X-Onshape-Access-Key": self.access_key,
        }
        payload = self._http_fetch(url, headers, parameters)
        if not payload:
            raise RuntimeError("Onshape REST export returned empty STEP bytes")
        dest.write_bytes(payload)
        return CadConnectorResult(
            connector_name=self.connector_name,
            template_or_model=model_ref,
            step_path=str(dest),
            step_sha256=hashlib.sha256(payload).hexdigest(),
            parameters={str(k): float(v) for k, v in parameters.items()},
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
