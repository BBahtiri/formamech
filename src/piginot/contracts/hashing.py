"""Canonical, domain-separated specification identities."""

from __future__ import annotations

import hashlib
from typing import Final, Literal, TypeAlias

from piginot.contracts.canonical import canonical_json_bytes
from piginot.contracts.errors import InvalidHashDomainError

HashDomain: TypeAlias = Literal["problem", "recipe", "content", "bundle"]
HASH_DOMAINS: Final = ("problem", "recipe", "content", "bundle")


def hash_spec(obj: object, domain: HashDomain | str) -> str:
    """Hash one canonical payload inside one of four explicit domains."""

    if domain not in HASH_DOMAINS:
        allowed = ", ".join(HASH_DOMAINS)
        raise InvalidHashDomainError(
            f"invalid hash domain {domain!r}; expected one of: {allowed}"
        )
    framed = {"domain": domain, "payload": obj}
    return hashlib.blake2b(canonical_json_bytes(framed), digest_size=32).hexdigest()
