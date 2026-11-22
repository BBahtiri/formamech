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
    qualify_policy: Path | None = None,
    qualify_dev_samples: int = 9,
    qualify_holdout_samples: int = 0,
    qualify_backend: str = "scikit-fem",
    qualify_seed: int = 0,
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
        if code == 0 and qualify_policy is not None:
            candidate = output / "candidate-bundle"
            if not candidate.is_dir():
                raise RuntimeError(
                    "qualification requested but candidate-bundle was not exported"
                )
            from piginot.bundle import load_bundle
            from piginot.qualify import (
                QualificationPolicy,
                qualify_bundle,
                sample_qualification_cases,
            )

            policy = QualificationPolicy.load(qualify_policy)
            bundle = load_bundle(candidate)
            dev_cases, ho_cases = sample_qualification_cases(
                bundle,
                dev_count=qualify_dev_samples,
                holdout_count=qualify_holdout_samples,
                seed=qualify_seed,
            )
            qres = qualify_bundle(
                candidate,
                policy,
                dev_cases=dev_cases,
                holdout_cases=ho_cases,
                backend=qualify_backend,
                output_dir=output / "qualification",
            )
            return 0 if qres.qualified else 3
        return code
    except (CompilerConfigError, PlanError, AdapterError, OSError) as error:
        print(f"{prog} compile: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"{prog} compile: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
