"""Installed compiler command; numerical stages are adapter capabilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from piginot.compiler.adapters import (
    AdapterError,
    resolve_adapter,
    resolve_configured_adapter,
)
from piginot.compiler.config import CampaignConfig, CompilerConfigError
from piginot.compiler.planning import PlanError, build_plan
from piginot.compiler.records import write_plan


def compile_command(
    spec: Path,
    config: Path,
    output: Path,
    *,
    plan_only: bool,
    resume: bool,
    prog: str = "piginot",
) -> int:
    try:
        campaign = CampaignConfig.load(config)
        adapter = (
            resolve_adapter(campaign.adapter.id)
            if campaign.adapter.geometry_resource is None
            else resolve_configured_adapter(campaign.adapter, config)
        )
        plan = build_plan(spec, config, output, descriptor=adapter.descriptor)
        if plan_only:
            write_plan(plan)
            print(
                json.dumps(plan.document(), indent=2, sort_keys=True, allow_nan=False)
            )
            return 0
        executor: Any = getattr(adapter, "execute", None)
        if not callable(executor):
            raise AdapterError(
                "execution",
                "this installed adapter has no numerical execution capability",
            )
        code = executor(plan, resume=resume)
        if type(code) is not int or code not in (0, 1, 2, 3):
            raise RuntimeError(
                "installed adapter returned an invalid compiler exit code"
            )
        return code
    except (CompilerConfigError, PlanError, AdapterError, OSError) as error:
        print(f"{prog} compile: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"{prog} compile: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
