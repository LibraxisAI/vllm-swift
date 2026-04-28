# vllm-swift v0.3.0 — Release Plan

**Status:** PLANNING. Nothing built or pushed yet.
**Target version:** v0.3.0 (minor bump from v0.2.2)
**Target date:** TBD

## Why v0.3.0 (minor, not patch)

The two headline items each justify a minor on their own:

1. **Metal `Invalid Resource` race fix** — closes a long-standing buffer-aliasing crash at concurrent custom-kernel workloads (TurboQuant B-path on MoE B≥8). Fix lands in `ml-explore/mlx#3461 / #3462` (and the in-fork mirror `ekryski/mlx#19`) as a `MTL::Buffer` retain on first sighting in the bind path. Foundational stability.
2. **~10% throughput recovery on TurboQuant MoE** — once retain lands, the swift-side `stopGradient(output) + asyncEval(output)` boundary in `compressedAttention` is redundant and was costing ~10% throughput. Removed in alpha. Measured 119.9 t/s vs 108.7 t/s on Qwen3.5-35B-A3B B=17 4K turbo4v2.

Adjacent improvements also picked up:

- TurboQuant maturity in alpha: `useCompressedAttention=true` default, `prepareQueriesScaled` cache, A-path rotation bypass, signpost profiling.
- bf16 kernel + Gemma 4 dim=512 instantiation (`ekryski/mlx-swift-lm#107`).
- `prefillStepSize` per-model protocol cleanup.
- DeepSeek-V4 Phase 1 — initial foundation: `model_type: deepseek_v4` dispatch wired into `LLMTypeRegistry`, weight loading and engine creation work end-to-end (90GB DSV4-Flash-2bit-DQ loads cleanly on M5 Max). Forward pass hits a GPU kernel timeout in Phase 1 — not yet stable for production decode, but the surface area is in place for follow-up. PR target: merge `feat/deepseek-v4-initial-support` (`ekryski/mlx-swift-lm#109`) into alpha before snapshotting.

## Scope

**In scope**

- Swift bridge unchanged. No bridge ABI changes.
- mlx-swift-lm dependency snapshot bump (alpha tip with retain commit in C++ submodule).
- Bottle rebuild against new chain.
- README perf tables refresh against post-retain numbers.
- CHANGELOG entry.

**Out of scope**

- DeepSeek-V4 stable production decode (this release ships the initial foundation; Phase 2 work is a follow-up).
- Cap=4 default in mlx (we proved it redundant after retain — that decision belongs upstream in mlx#19, not in vllm-swift).
- Upstream coordination of mlx-swift-lm#232 (separate workstream, see [[TurboQuant Upstream PR 232 - Audit and Plan]] in obsidian).

## Submodule pin coordination

The chain that needs to converge on a tested set of SHAs:

```
vllm-swift                 v0.3.0 (this release)
  ↓ swift/Package.swift   .package(url: "TheTom/mlx-swift-lm.git", branch: "vllm-swift-stable")
mlx-swift-lm                <pinned to alpha tip we tested>
  ↓ Package.resolved       mlx-swift @ <SHA with turboBulkDequantRotated binding>
mlx-swift                   <published or local pin>
  ↓ submodule              mlx @ <SHA with retain commit>
                           mlx-c @ <alpha tip with 11 turbo bindings>
mlx (C++)                   retain commit present (from mlx#19, may not be merged yet)
mlx-c                       all 11 turbo C bindings present
```

**Open question:** mlx#19 (the retain commit upstream) is not merged at planning time. We have two options:

- **Option A (preferred):** Force-push `alpha` → `vllm-swift-stable` on `TheTom/mlx-swift-lm` so the snapshot pin captures the local mlx submodule with retain cherry-picked. Self-contained release. If/when mlx#19 merges upstream, follow up with a fresh snapshot in v0.3.1.
- **Option B:** Wait for mlx#19 to merge in `ml-explore/mlx`, then bump everything in lockstep. Lower autonomy, depends on Apple maintainer review timing.

Default plan: Option A. The retain commit is a local cherry-pick, well-tested, isolated change.

## Sequence

1. Author and merge any final cleanup PRs to `ekryski/mlx-swift-lm` alpha.
2. Merge `feat/deepseek-v4-initial-support` (PR #109) into alpha so DSV4 dispatch is part of the snapshot.
3. Snapshot `alpha` → `vllm-swift-stable` on `TheTom/mlx-swift-lm`.
4. In `vllm-swift`: `swift package update`, commit refreshed `Package.resolved`.
5. Run full `02-TEST-PLAN.md` matrix locally + on Mac Mini, including a DSV4 load + dispatch smoke test (no expectation of full decode).
5. Bump version in 4 files (see `01-VERSION-BUMP.md`).
6. Update `CHANGELOG.md` with v0.3.0 entry (use `03-RELEASE-NOTES-DRAFT.md`).
7. Update `README.md` perf tables if numbers shift.
8. Commit, tag `v0.3.0`, push.
9. Build bottle (`./scripts/build_bottle.sh`).
10. Update tap formula on `TheTom/homebrew-tap`.
11. Fresh-install verify on Mac Mini.
12. GitHub Release with notes.
13. Post on X (using Tom's voice — direct, lowercase, data-first).

## Risks

- **mlx#19 not merged at release time** — Option A above mitigates. If we go with Option A, document in release notes that the retain fix is shipped as a local pin until upstream merges.
- **Bottle build tooling drift** — last bottle was v0.2.2 (Apr 26). Build script may need tweaks against current toolchain. Test-build before tagging.
- **vllm-swift-stable branch divergence** — if anyone has been pushing to `vllm-swift-stable` independently, force-push from alpha will lose those commits. Sanity check git log before push.
- **Submodule SHA drift on user side** — `swift package resolve` users could end up on different SHAs than tested if branch-based pins move. Lock `Package.resolved` and ship it.

## Files in this folder

- `00-PLAN.md` — this file
- `01-VERSION-BUMP.md` — exact files to update with `0.2.2` → `0.3.0`
- `02-TEST-PLAN.md` — pre-release validation matrix
- `03-RELEASE-NOTES-DRAFT.md` — user-facing release notes draft
- `04-EXPECTED-GAINS.md` — concrete before/after numbers from today's testing
