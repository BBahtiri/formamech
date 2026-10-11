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

## Checks run on the owner's machine (WSL2 Ubuntu)

Three G19 checks need a multi-core machine or data that is not in Git. They
run under **WSL2 Ubuntu**, not native Windows: on Windows the pinned S13c
initial weights do not reproduce and ten more tests fail for platform reasons
(ADR 0100, findings). Until they run, G19 is `FAIL` with `NOT_RUN`.

Recorded on 10 Oct 2026 (Lunar Lake, Linux, Python 3.11.15, torch
2.13.0+cu130, AVX2):

| Check | Result |
| --- | --- |
| `G19.coverage` | 3,075 passed, 1 skipped, 0 failed; branch coverage 91.44% |
| `G19.s13c-rescore` | re-score of the registered checkpoint completed (`selected`) |
| `G19.resume` | stopped at 200, resumed to 400 = straight to 400: model state `9492c444…`, manifest `d9568fc8…` |

### Setup (bash, once)

```bash
sudo apt-get update && sudo apt-get install -y git libglu1-mesa libxcursor1 libxft2 libxinerama1 libxrender1 libgl1
command -v uv || { curl -LsSf https://astral.sh/uv/install.sh | sh; source ~/.local/bin/env; }
D=/mnt/c/Users/betim/Documents/Data_Science
S10=$D/pi_ginot_large_s10/.piginot-runs/dogbone-s10-public-production-v1-b29c4af-d3ede9186f33
S13C=$D/pi_ginot_large_s13b_enhanced/.piginot-runs/s13c-registered-8736c1a/checkpoint-step-2400
mkdir -p ~/s19 && cd ~/s19
git clone --branch codex/s18-release "$D/vno-s18.bundle" vno-s19
cd vno-s19
git fetch "$D/vno-s19.bundle" "+refs/heads/codex/s19-fork-baseline:refs/heads/codex/s19-fork-baseline" "+refs/tags/v0.2.0-base:refs/tags/v0.2.0-base" "+refs/tags/s19-cloud:refs/tags/s19-cloud"
git switch codex/s19-fork-baseline
cd ~/s19 && git clone --no-local --branch v0.2.0-base vno-s19 vno-base
for d in vno-s19 vno-base; do git -C $d remote remove origin; git -C $d config user.name "Betim Bahtiri"; git -C $d config user.email "betimbahtiri@outlook.de"; done
```

### 1. `G19.coverage`

```bash
cd ~/s19/vno-base && uv sync --frozen --all-extras --python 3.11
uv run --frozen python -m pytest -q -p no:cacheprovider --cov=src/piginot --cov-report=json:$HOME/s19/s19-coverage.json 2>&1 | tee ~/s19/s19-coverage.log
cd ~/s19/vno-s19 && uv sync --frozen --all-extras --python 3.11
uv run --frozen python -m campaigns.g19_gate --record-coverage ../vno-base ../s19-coverage.json ../s19-coverage.log
```

### 2. `G19.s13c-rescore`

```bash
cd ~/s19/vno-s19
uv run --frozen python -m campaigns.dogbone_v1.s13c_evaluate --s10-run-root "$S10" --checkpoint-path "$S13C" > ../s19-rescore.json
```

### 3. `G19.resume`

```bash
cd ~/s19/vno-base
uv run --frozen python -m campaigns.dogbone_v1.s13c_trial --s10-run-root "$S10" --run-root ~/s19/s19-replay   --segment-steps 200 --stop-after 200
uv run --frozen python -m campaigns.dogbone_v1.s13c_trial --s10-run-root "$S10" --run-root ~/s19/s19-replay   --segment-steps 200 --stop-after 400
uv run --frozen python -m campaigns.dogbone_v1.s13c_trial --s10-run-root "$S10" --run-root ~/s19/s19-straight --segment-steps 400 --stop-after 400
```

### 4. Gate

```bash
cd ~/s19/vno-s19
uv run --frozen python -m campaigns.g19_gate --record-golden --write \
  --s13c-evaluation ../s19-rescore.json \
  --replay-checkpoint ~/s19/s19-replay/checkpoint-step-0400 \
  --straight-checkpoint ~/s19/s19-straight/checkpoint-step-0400
git add artifacts/fork artifacts/gates/S19.json
git commit -m "chore(S19): G19 with the WSL coverage, rescore and replay checks"
git bundle create ~/s19/vno-s19-local.bundle codex/s19-fork-baseline ^s19-cloud-2
cp ~/s19/vno-s19-local.bundle "$D/"
```

`--record-golden` adds the example-output digests of the Linux/AVX2 profile
next to the cloud's Linux/AVX512 golden.

## Comparing the bytes of the source and the fork

```powershell
cd pi_ginot_large
uv run python ..\vno-s19\scripts\verify_fork.py manifest --root . --include artifacts bundles --out ..\source-manifest.json
uv run python ..\vno-s19\scripts\verify_fork.py compare ..\source-manifest.json ..\vno-s19\artifacts\fork\S19-BASE-MANIFEST.json
```

Run it on the `codex/s18-release` checkout; it prints `identical`. (On a
native Windows checkout use `-c core.autocrlf=false` when cloning, so text files
keep their committed line endings.)
