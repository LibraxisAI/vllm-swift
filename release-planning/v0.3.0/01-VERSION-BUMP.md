# v0.3.0 — Version Bump Checklist

`0.2.2` → `0.3.0`. All four locations must be updated together.

## Files to update

### 1. `pyproject.toml`

```diff
-version = "0.2.2"
+version = "0.3.0"
```

### 2. `homebrew/vllm-swift.rb`

```diff
-  version "0.2.2"
+  version "0.3.0"
```

Also: zero out the bottle SHAs until the new bottle is built. Replace both `arm64_tahoe` and `arm64_sequoia` SHA strings with placeholders (or just delete the `bottle do … end` block until `build_bottle.sh` has run and produced the new SHA — the build script regenerates it).

### 3. `scripts/build_bottle.sh`

```diff
-VERSION="0.2.2"
+VERSION="0.3.0"
```

(Line 13. Single occurrence drives the rest of the script.)

### 4. Wrapper script version string

The `build_bottle.sh` writes a `vllm-swift` wrapper that responds to `vllm-swift version`. Search for the embedded version string in the wrapper-generation block of that script and update it from `0.2.2` to `0.3.0`. There is one location around the `141:  version)` switch arm in `build_bottle.sh`.

## Submodule pin update

Before bumping versions, lock the dependency snapshot.

### Step 1 — push tested alpha to vllm-swift-stable

```bash
cd ~/dev/mlx-swift-lm
git checkout alpha
git log --oneline HEAD -5   # confirm a5cad08 (or current alpha tip) is what we tested
git push fork alpha:vllm-swift-stable --force
```

This is the destructive step in `RELEASING.md` step 1. Confirm the snapshot SHA before push. Today's tested SHA on `ekryski/mlx-swift-lm/alpha` is `a5cad08` (read from `git log` output earlier; verify before push).

### Step 2 — refresh vllm-swift Package.resolved

```bash
cd ~/dev/vllm-swift/swift
swift package update                # pulls vllm-swift-stable to current
git status                          # Package.resolved should be the only change
git diff Package.resolved          # sanity check the new mlx-swift-lm SHA
```

### Step 3 — verify the chain

The Package.resolved should now point to:

| dep | branch / SHA | what to verify |
|---|---|---|
| `mlx-swift-lm` | `vllm-swift-stable` @ <new SHA> | matches alpha tip |
| `mlx-swift` | `8a5a74e` or newer | has `MLXFast.turboBulkDequantRotated` binding |

Verify mlx-swift checkout has the retain commit reachable in its `Source/Cmlx/mlx` submodule pointer:

```bash
cd swift/.build/checkouts/mlx-swift
git submodule status     # mlx pointer should resolve to a commit containing retain
git -C Source/Cmlx/mlx log --oneline HEAD -3 | grep -i retain
```

Expected: a commit titled `Retain bound buffers under untracked hazard mode`.

## Commit message conventions

```
release: v0.3.0 — Metal Invalid Resource race fix + ~10% TQ MoE perf
```

Follow the existing pattern: short hyphen-summary on the title. Co-author: `tturney@psyguard.ai`.

## Tag and push

```bash
git add -A
git commit -m "release: v0.3.0 — ..."
git push origin main
git tag v0.3.0
git push origin v0.3.0
```

## After version bump

Bottle build (`scripts/build_bottle.sh`) and tap formula update follow the existing `RELEASING.md` flow steps 5–7. No deviation needed.

## Sanity check before bumping

```bash
grep -rn "0\.2\.2" pyproject.toml homebrew/ scripts/build_bottle.sh
# Expected: 4 hits (one per file from the list above).
# After bump:
grep -rn "0\.3\.0" pyproject.toml homebrew/ scripts/build_bottle.sh
# Expected: 4 hits, same locations.
```
