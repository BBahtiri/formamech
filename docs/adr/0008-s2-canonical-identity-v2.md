# ADR 0008: S2 canonical identity format v2

- Status: Accepted
- Date: 2026-08-20
- Decision owner: Betim Bahtiri
- Gate: `G2` (pending)

## Context

S2 defines the immutable, hashable, unit-safe contract layer that later steps use
for recipes, content identity and bundle identity. Before S3 introduces a content
store, the canonical byte format must be stable enough that hashes are reproducible
and semantically meaningful.

The first S2 contract draft exposed several format problems before any S3 storage
or migration burden existed:

- dataclass type identity was implicit, so two structurally similar spec classes
  could collide unless callers supplied external context;
- field participation risked being inferred from `repr` or `compare`, which are
  Python object-behaviour controls rather than canonical-format controls;
- quantities carried unit and value but not an explicit physical dimension, making
  dimension-sensitive identity ambiguous;
- `ENERGY` and `MOMENT` both canonicalize to the same base unit, `N·mm`, but they
  are not interchangeable physical dimensions;
- permissive integer parsing could admit values outside the RFC 8785 / JCS safe
  integer domain before canonicalization;
- derived expressions needed a reproducible identity, but algebraic normalization
  would add an evaluator or simplifier that belongs outside S2.

This decision is made while only S2 files and test vectors are affected. No S3
content-addressed store exists yet, so no stored recipes, contents or bundles need
migration.

## Decision

Accept canonical identity format v2 for S2 and all downstream hash inputs.

Canonical dataclass objects include a stable public discriminator field named
`$type`. The discriminator is part of the canonical payload and is not inferred from
the hash domain. Its value is a public contract identifier, stable across supported
runtime versions unless an ADR records a deliberate format change.
Plain mappings may not provide `$type`; that name is reserved for the canonicalizer
so user-supplied dictionaries cannot spoof typed contract payloads.

Dataclass field membership is controlled only by field metadata:

```python
field(metadata={"canonical": False})
```

Fields are canonical by default. `repr=False`, `compare=False`, private naming and
other Python dataclass behaviours do not remove a field from canonical identity.

Every dimensional quantity carries three pieces of identity in canonical form:

- numeric value in the canonical internal unit system;
- canonical unit;
- physical dimension.

`ENERGY` and `MOMENT` remain distinct dimensions even though both use the canonical
unit `N·mm`. A value derived from joules and a value derived from newton-millimetres
therefore may share a numeric canonical unit representation while still hashing
differently when their dimensions differ.

Strict JSON parsing rejects integers outside the JCS safe integer range at
`parse_int`, before object construction. It also keeps duplicate-key rejection at
the parse boundary rather than relying on Python dictionaries after duplicate keys
have already been collapsed.

Derived-expression identity is structural, not algebraic. The canonical expression
payload records a validated syntax tree over the S2 grammar: names, finite numeric
literals, unary sign, binary `+`, `-`, `*`, `/`, and parentheses as represented by
the parsed tree. Whitespace and numeric spelling normalize through parsing, but
equivalent algebraic forms such as `width / notch_depth` and
`width * (1 / notch_depth)` remain different identities. S2 validates syntax,
referenced parameter names and static dimensions; it does not evaluate expressions
against parameter values, simplify algebra, sample a DOE or check envelope
membership.

The existing v1 hashes and vectors are superseded before any S3 store exists.
`vectors-v1` is retained only as historical evidence of the replaced draft format.
`vectors-v2` is the required vector set for `G2.vectors` and later consumers.

No S3 migration code, content-store compatibility shim, lifecycle state machine or
bundle upgrade path is introduced by this ADR.

## Consequences

Format v2 makes canonical identity more explicit and less dependent on incidental
Python object behaviour. Later recipes can compare public contract payloads without
guessing which spec class produced a structurally similar JSON object.

The cost is that every serialized dataclass now carries `$type`, and every quantity
now carries dimension even when the unit appears sufficient. This is intentional:
hash stability is more important than minimizing payload size at the contract layer.

The S2 implementation and fixtures must treat v1 vectors as obsolete. A test or
artifact still deriving a passing claim from v1 vectors is a bug. Keeping v1 vectors
as history is allowed only if they are clearly labelled as superseded and excluded
from the active G2 vector check.

Because this decision happens before S3, there is no migration obligation. Any later
format change after S3 must record how existing content-addressed objects are
handled.

## Verification

This decision is verified by S2 tests and the `G2` artifact:

- canonical dataclass bytes include stable `$type` discriminators;
- canonical field inclusion follows only `metadata={"canonical": False}`;
- quantities include dimension, and `ENERGY` hashes distinctly from `MOMENT`;
- strict parsing rejects out-of-domain integers and duplicate object names before
  canonicalization;
- derived-expression fixtures prove structural identity and explicitly reject
  algebraic normalization;
- `vectors-v2` is the active committed vector set for `G2.vectors`;
- `vectors-v1` remains historical and cannot satisfy `G2.vectors`;
- no S3 store, migration code or bundle compatibility path is present.

Revisit this ADR if S3 introduces a content-store compatibility policy, or if any
future step needs algebraic expression equivalence rather than structural contract
identity.
