"""Handlers for CAD geometry preparation and metadata-only inspection."""

from __future__ import annotations

import argparse
import json
import sys

from piginot.cad.configuration import prepare_campaign
from piginot.cad.mesh_policy import MeshPolicy
from piginot.cad.project import inspect_geometry, prepare_geometry
from piginot.contracts import Unit


def run(args: argparse.Namespace) -> int:
    try:
        if args.cad_command == "import":
            sizing = {"mesh_size_mm": args.mesh_size_mm}
            if args.mesh_policy:
                sizing = {"mesh_policy": MeshPolicy.load(args.mesh_policy)}
            result = prepare_geometry(
                args.source,
                args.output,
                unit=Unit(args.unit),
                **sizing,
            )
        elif args.cad_command == "inspect":
            result = inspect_geometry(args.project)
        elif args.cad_command == "prepare":
            result = prepare_campaign(
                args.project,
                args.settings,
                args.output,
                execution_profile=args.execution_profile,
            )
        else:
            raise ValueError(f"unknown CAD action: {args.cad_command}")
    except (OSError, ValueError, ImportError) as error:
        print(
            json.dumps({"status": "REFUSED", "reason": str(error)}, allow_nan=False),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0
