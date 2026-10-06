# S18 plan — release 0.1.0

- Status: In progress. The decision owner signed off `limitations.md` on
  2026-10-06. G18 waits for the `v0.1.0` tag and the TestPyPI run.
- Date: 2026-10-06
- Base: `codex/s17-examples` (G17 PASS)
- Gate: `G18` (`STEPS-v1.0.md` §S18, ⚑ human sign-off)
- Decision record: [ADR 0032](../adr/0032-release-0.1.0.md)

## 1. What S18 must deliver

| Check | Threshold | How it is checked (`campaigns/g18_gate.py`) |
| --- | --- | --- |
| `G18.all-gates` | G0–G17 all `PASS`, or `NOT_RUN` with a written reason accepted in the release ADR | reads `artifacts/gates/S0.json` … `S17.json` (and `S11-ENERGY.json`) |
| `G18.claim-ceiling` | no shipped bundle claims above `HOLDOUT_VERIFIED`; `SEALED_RELEASE_VERIFIED` is not emittable | `ClaimLevel` members, `tests/test_release_claim_ceiling.py`, and loading both bundles |
| `G18.licenses` | every adapted component from GNOT, GINO or PointNet++ recorded with source, commit, digest and licence compatibility | [upstream sources](../adr/upstream-sources.md); the GINOT modules untracked; rebuilt wheel and sdist file lists; no "All Rights Reserved" text in them |
| `G18.naming` | README disclaims faithful reproduction of GNOT and GINO and gives the exact `architecture_id` | README text; the serving registry; both bundles' `architecture_id` |
| `G18.citation` | `CITATION.cff` validates | `cffconvert 2.0.0 --validate`; version matches |
| `G18.install` | install from TestPyPI succeeds on Linux, macOS and Windows | `artifacts/release/TESTPYPI-INSTALL.json` from `.github/workflows/release.yml` |
| `G18.lock` | lockfile committed and tool-generated | `uv lock --check` |
| `G18.changelog` | version tagged, changelog written | `CHANGELOG.md` entry; tag `v0.1.0` at or before `HEAD` |
| ⚑ `G18.signoff` | the decision owner confirms every sentence of `limitations.md` is true of the code as built | `artifacts/release/LIMITATIONS-SIGNOFF.json` naming the file's digest |

## 2. What was done

- **Licences.** The new [upstream sources](../adr/upstream-sources.md) record
  lists the components:
  - the served operator re-implements `BBahtiri/PI-GINOT@12c9c1c`;
  - the S11 operator re-implements `BBahtiri/PI-GINOT-DogBone@9f67941`;
  - both upstreams are MIT, and a line scan finds no copied upstream line in
    `src/`;
  - GNOT, GINO and PointNet++ contribute no shipped code.

  The five GINOT-derived modules of the S1 spike leave the tree, by the owner's
  decision. Two of them carry "All Rights Reserved", and the GINOT repository
  has no licence. They are pinned in `THIRD_PARTY.json`, and
  `python -m experiments.spike.fetch_upstream` fetches them. With the modules
  fetched, the spike's own pin test passes as before.
- **Distributions.** The wheel holds `piginot` only. The sdist holds the
  package, licence, README, citation, changelog, `pyproject.toml` and
  `uv.lock`. Hatchling is pinned below 1.30 so both carry metadata 2.4, and
  `twine check --strict` passes.
- **Naming.** The README section "Name and architecture" disclaims a faithful
  reproduction of GNOT, GINO and GINOT. It gives
  `architecture_id: pi-ginot-energy-parametric-film-v1`.
- **Claim ceiling.** `tests/test_release_claim_ceiling.py` checks three
  things: the claim levels stop at `HOLDOUT_VERIFIED`; no source file names a
  sealed-release claim; a bundle claiming more is refused on load.
- **Citation.** `CITATION.cff` names Betim Bahtiri (owner's decision), the
  repository and three references (the two PI-GINOT repositories and the GINOT
  paper). It validates against CFF 1.2.0.
- **Lockfile.** `uv.lock` was stale since the S15 `runtime` extra; it is
  regenerated with `uv lock`. CI now runs `uv lock --check`, the citation
  validator and `twine check` in a `release-metadata` job.
- **Release workflow.** `.github/workflows/release.yml` builds and checks the
  distributions, publishes to TestPyPI with the secret
  `TEST_PYPI_API_TOKEN`, then installs `piginot[runtime]==0.1.0` from TestPyPI
  on Linux, macOS and Windows. On each it runs `scripts/release_smoke.py`,
  which checks the installed location, the version, the console script, the
  claim ceiling and a refusal.
- **Changelog and limitations.** `CHANGELOG.md` records 0.1.0. The
  limitations page drops statements that were no longer true of the code as
  built. These were: S11 "may begin", `G17.non-dogbone` "NOT_RUN", "the
  bundle serves one family", and "one frozen candidate". It gains the release
  limits.

## 3. The owner's steps

1. ~~Sign off `docs/limitations.md`~~ — done on 2026-10-06
   (`artifacts/release/LIMITATIONS-SIGNOFF.json`).
2. **TestPyPI token.** Create a TestPyPI API token and add it to the GitHub
   repository as the Actions secret `TEST_PYPI_API_TOKEN`.
3. **Push** `codex/s18-release` and the tag `v0.1.0`. The tag starts the
   release workflow.
4. **Report the run.** Send the workflow run URL and the three JSON lines from
   the install jobs' summaries. They become
   `artifacts/release/TESTPYPI-INSTALL.json`, and G18 is run again.

## 4. Results

Recorded when G18 is final.
