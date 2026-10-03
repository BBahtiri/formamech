"""DogBone v1 campaign plugin for the ``piginot`` command line.

``piginot <command> --campaign dogbone_v1 [-- arguments]`` dispatches here. Each
entry calls the registered campaign script of its step, unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from types import MappingProxyType


def _no_arguments(name: str, arguments: list[str]) -> None:
    if arguments:
        raise SystemExit(f"{name} takes no arguments, got {arguments}")


def family_compile(arguments: list[str]) -> int:
    """Compile the family and print its identities."""

    from campaigns.dogbone_v1.family import compiled_family
    from piginot.contracts import hash_spec

    _no_arguments("family compile", arguments)
    family = compiled_family()
    print(
        json.dumps(
            {
                "recipe_id": family.recipe_id,
                "compiled_family_hash": hash_spec(family, "content"),
                "membership_hash": hash_spec(family.membership, "content"),
                "reference_mesh_id": family.reference_mesh.mesh_id,
                "parameters": [
                    bound.name for bound in family.membership.primitive_bounds
                ],
            },
            indent=2,
        )
    )
    return 0


def pilot(arguments: list[str]) -> int:
    from campaigns.dogbone_v1.strain_pilot import main

    _no_arguments("pilot", arguments)
    return main()


def campaign(arguments: list[str]) -> int:
    from campaigns.dogbone_v1.campaign_proposal import main

    _no_arguments("campaign", arguments)
    return main()


def simulate(arguments: list[str]) -> int:
    from campaigns.dogbone_v1.s10_production import main

    return main(tuple(arguments))


def train(arguments: list[str]) -> int:
    from campaigns.dogbone_v1.s13c_trial import main

    return main(arguments)


def verify(arguments: list[str]) -> int:
    from campaigns.dogbone_v1.s14_validate import main

    if arguments[:1] == ["holdout"]:
        raise SystemExit("use holdout-reveal to consume HOLDOUT")
    return main(arguments)


def holdout_reveal(arguments: list[str]) -> int:
    from campaigns.dogbone_v1.s14_validate import main

    return main(["holdout", *arguments])


def bundle_build(arguments: list[str]) -> int:
    from campaigns.dogbone_v1.s15_bundle import main

    return main(arguments)


COMMANDS: Mapping[str, Callable[[list[str]], int]] = MappingProxyType(
    {
        "family compile": family_compile,
        "pilot": pilot,
        "campaign": campaign,
        "simulate": simulate,
        "train": train,
        "verify": verify,
        "holdout-reveal": holdout_reveal,
        "bundle build": bundle_build,
    }
)

__all__ = ("COMMANDS",)
