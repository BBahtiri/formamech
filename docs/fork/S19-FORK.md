# S19 — fork and baseline lock

Decision record: [ADR 0100](../adr/0100-s19-fork-for-the-s19-line.md).
Gate: `python -m campaigns.g19_gate --write` → `artifacts/gates/S19.json`.

## Source and fork

| Item | Value |
| --- | --- |
| Source | `codex/s18-release` of `pi_ginot_large`, sent as the Git bundle `vno-s18.bundle` (complete history, verified) |
| Fork base | `399a14659cec6812e24980ece8f81ab2d94a4753`, tree `4985578321cc54de0306fc5bb25d5e7992ecfb6c` |
| Tags | `v0.1.0` (tag object `d7c0300`) and `v0.2.0-base` (annotated, new) both resolve to the base |
| Branch | `codex/s19-fork-baseline` |
| Isolation | no remote, no object alternates, `.git` a directory, one worktree |
| Identity | Betim Bahtiri `<betimbahtiri@outlook.de>`; the 71 historical `S9 test` commits are kept (ADR 0100) |
| Environment | `uv sync --frozen --all-extras`, Python 3.11.15, torch 2.13.0+cu130, CPU capability AVX512 (the registered S13c profile) |

## Baseline evidence recorded in the cloud workspace

| Evidence | File | Result |
| --- | --- | --- |
| Full suite at the base tree | `artifacts/fork/S19-SUITE.json` | 3,076 collected; 3,075 passed, 1 skipped, 0 failed, 0 errors (identical to S18) |
| Byte manifest of `artifacts/` and `bundles/` at the base | `artifacts/fork/S19-BASE-MANIFEST.json` | 129 files, BLAKE2b table `e14e8ea8…` |
| Bundle example outputs, per execution profile | `artifacts/fork/S19-BUNDLE-OUTPUTS.json` | SHA-256 of each served example document |

The suite ran file by file without coverage, because the cloud workspace has
one core and branch coverage slows the slowest fixtures about eightfold.
`src/`, `tests/` and `campaigns/` are identical between `12b3138`, where S18
measured 91% branch coverage, and the base.

## Checks to run on your machine

Three G19 checks need either a multi-core machine or data that is not in Git.
Until they run, G19 is `FAIL` with `NOT_RUN`, never `PASS`. All commands are
PowerShell, from `C:\Users\betim\Documents\Data_Science`.

### 0. Fetch the S19 bundle

```powershell
cd pi_ginot_large
git fetch ..\vno-s19.bundle "+refs/heads/codex/s19-fork-baseline:refs/heads/codex/s19-fork-baseline" "+refs/tags/v0.2.0-base:refs/tags/v0.2.0-base"
git rev-parse "v0.2.0-base^{commit}"     # 399a14659cec...
cd ..
git clone --no-local --branch v0.2.0-base pi_ginot_large vno-base      # clean base, for checks 1 and 3
git clone --no-local --branch codex/s19-fork-baseline pi_ginot_large vno-s19
```

### 1. `G19.coverage` — full suite with branch coverage (about 1 h)

```powershell
cd vno-base
uv sync --frozen --all-extras
uv run python -m pytest -q -p no:cacheprovider --cov=src/piginot --cov-report=json:..\s19-coverage.json 2>&1 | Tee-Object ..\s19-coverage.log
cd ..\vno-s19
uv sync --frozen --all-extras
uv run python -m campaigns.g19_gate --record-coverage ..\vno-base ..\s19-coverage.json ..\s19-coverage.log
```

### 2. `G19.s13c-rescore` — re-score the registered checkpoint (about 12 min)

```powershell
cd vno-s19
uv run python -m campaigns.dogbone_v1.s13c_evaluate `
  --s10-run-root <...\pi_ginot_large_s10\.piginot-runs\dogbone-s10-public-production-v1-b29c4af-d3ede9186f33> `
  --checkpoint-path <...\.piginot-runs\s13c-registered-8736c1a\checkpoint-step-2400> `
  > ..\s19-rescore.json
```

### 3. `G19.resume` — interrupted and resumed replay to step 400

Needs a CPU whose torch dispatches AVX2 or AVX512 (the pinned initial-weights
hash), and a clean checkout, because the revision is bound into `run_hash`.
The run root must be a new folder outside the repository.

```powershell
cd vno-base
uv sync --frozen --all-extras
uv run python -m campaigns.dogbone_v1.s13c_trial --s10-run-root <S10 run root> --run-root ..\s19-replay --segment-steps 200 --stop-after 200
uv run python -m campaigns.dogbone_v1.s13c_trial --s10-run-root <S10 run root> --run-root ..\s19-replay --segment-steps 200 --stop-after 400
```

The second command resumes from `checkpoint-step-0200`. The gate compares the
model-state hash of `checkpoint-step-0400` with the registered replay probe
(`e45edbdc…`).

### 4. Gate

```powershell
cd vno-s19
uv run python -m campaigns.g19_gate --record-golden --write `
  --s13c-evaluation ..\s19-rescore.json --replay-checkpoint ..\s19-replay\checkpoint-step-0400
git add artifacts/fork artifacts/gates/S19.json
git commit -m "chore(S19): G19 with the local coverage, rescore and replay checks"
git bundle create ..\vno-s19-local.bundle codex/s19-fork-baseline ^<HEAD of the cloud bundle>
```

`--record-golden` adds the example-output digests of your execution profile
(Windows, your CPU) next to the cloud profile's, on first use. Send
`vno-s19-local.bundle` back, or the four files.

## Comparing the bytes of the source and the fork

```powershell
cd pi_ginot_large
uv run python ..\vno-s19\scripts\verify_fork.py manifest --root . --include artifacts bundles --out ..\source-manifest.json
uv run python ..\vno-s19\scripts\verify_fork.py compare ..\source-manifest.json ..\vno-s19\artifacts\fork\S19-BASE-MANIFEST.json
```

Run it on the `codex/s18-release` checkout; it prints `identical`.
