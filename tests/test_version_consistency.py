# SPDX-License-Identifier: Apache-2.0
"""Version-string consistency tests.

Catches the class of bug where one version source (pyproject.toml,
vllm_swift/__init__.py, scripts/build_bottle.sh VERSION, the bottled
wrapper's `version` cmd, homebrew/vllm-swift.rb) drifts from the
others. Hit on 2026-05-13 when bumping 0.6.0 → 0.6.3 missed a
hardcoded "0.6.0" inside the wrapper template, shipping a bottle
whose `vllm-swift version` reported the wrong number.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text()


@pytest.fixture(scope="module")
def canonical_version() -> str:
    """The single source of truth: pyproject.toml [project] version.

    All other version strings in the repo MUST match this.
    """
    text = _read("pyproject.toml")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert m, "pyproject.toml has no [project] version line"
    return m.group(1)


def test_init_py_version_matches(canonical_version):
    """`vllm_swift/__init__.py:__version__` MUST match pyproject.toml.
    Drift here ships wrong version strings in `vllm-swift version` and
    in any code that imports `vllm_swift.__version__`."""
    text = _read("vllm_swift/__init__.py")
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert m, "vllm_swift/__init__.py has no __version__ line"
    assert m.group(1) == canonical_version, (
        f"vllm_swift/__init__.py __version__ = {m.group(1)!r} "
        f"!= pyproject.toml version = {canonical_version!r}"
    )


def test_build_bottle_sh_version_matches(canonical_version):
    """`scripts/build_bottle.sh:VERSION` MUST match pyproject.toml.
    Drift here builds a bottle in the wrong versioned subdir and the
    Homebrew formula's sha256 stamp lands on the wrong release."""
    text = _read("scripts/build_bottle.sh")
    m = re.search(r'^VERSION\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert m, "scripts/build_bottle.sh has no VERSION= line"
    assert m.group(1) == canonical_version, (
        f"scripts/build_bottle.sh VERSION = {m.group(1)!r} "
        f"!= pyproject.toml version = {canonical_version!r}"
    )


def test_homebrew_formula_version_matches(canonical_version):
    """`homebrew/vllm-swift.rb` `version` MUST match pyproject.toml."""
    text = _read("homebrew/vllm-swift.rb")
    m = re.search(r'^\s*version\s+"([^"]+)"', text, re.MULTILINE)
    assert m, "homebrew/vllm-swift.rb has no version line"
    assert m.group(1) == canonical_version, (
        f"homebrew/vllm-swift.rb version = {m.group(1)!r} "
        f"!= pyproject.toml version = {canonical_version!r}"
    )


def test_homebrew_formula_version_subcommand_matches(canonical_version):
    """The `version` subcommand string baked into homebrew/vllm-swift.rb
    must match pyproject.toml. (Brew formula has its own inline wrapper.)"""
    text = _read("homebrew/vllm-swift.rb")
    m = re.search(r'echo\s+"vllm-swift\s+([^"]+)"', text)
    assert m, 'homebrew/vllm-swift.rb has no `echo "vllm-swift X"` line'
    assert m.group(1) == canonical_version, (
        f"homebrew/vllm-swift.rb wrapper 'vllm-swift {m.group(1)}' "
        f"!= pyproject.toml version {canonical_version!r}"
    )


def test_homebrew_formula_test_assertion_matches(canonical_version):
    """The `test do ... assert_match \"X.Y.Z\"` in the formula must
    match. Otherwise `brew test vllm-swift` always fails."""
    text = _read("homebrew/vllm-swift.rb")
    m = re.search(r'assert_match\s+"([0-9]+\.[0-9]+\.[0-9]+)"', text)
    assert m, "homebrew/vllm-swift.rb has no assert_match version line"
    assert m.group(1) == canonical_version, (
        f"homebrew/vllm-swift.rb test assert = {m.group(1)!r} "
        f"!= pyproject.toml version = {canonical_version!r}"
    )


def test_published_tap_formula_matches_repo_formula(canonical_version):
    """Network-gated regression: the formula in `TheTom/homebrew-tap`
    MUST match the one in this repo's `homebrew/vllm-swift.rb`.

    The 0.6.2 → 0.6.3 release shipped with `homebrew/vllm-swift.rb`
    correctly bumped here but the tap formula at TheTom/homebrew-tap
    was left at 0.6.2 — `brew install TheTom/tap/vllm-swift` would
    have silently installed the old version. This test catches that
    class of split-brain drift.

    Opt out with `VLLM_SWIFT_SKIP_TAP_CHECK=1` (CI on no-network).
    """
    import os
    import urllib.error
    import urllib.request

    if os.environ.get("VLLM_SWIFT_SKIP_TAP_CHECK"):
        pytest.skip("VLLM_SWIFT_SKIP_TAP_CHECK set")

    import base64
    import json as _json

    # Use the GitHub Contents API (uncached) instead of
    # raw.githubusercontent (5-min CDN cache that false-positives this
    # test for ~5 min after every tap push).
    try:
        with urllib.request.urlopen(
            "https://api.github.com/repos/TheTom/homebrew-tap/contents/Formula/vllm-swift.rb",
            timeout=10,
        ) as r:
            payload = _json.loads(r.read())
            tap_text = base64.b64decode(payload["content"]).decode("utf-8")
    except (urllib.error.URLError, ConnectionError, OSError, KeyError, ValueError) as e:
        pytest.skip(f"network/api unavailable: {e}")

    m = re.search(r'^\s*version\s+"([^"]+)"', tap_text, re.MULTILINE)
    if m is None:
        pytest.skip("tap formula missing version line (api glitch?)")
    tap_version = m.group(1)

    assert tap_version == canonical_version, (
        f"Published tap formula version = {tap_version!r}, "
        f"repo formula version = {canonical_version!r}. "
        f"Push the updated formula to TheTom/homebrew-tap "
        f"before announcing the release."
    )

    # Also verify the bottle SHA stamp matches the repo's. Mismatch =
    # the tap is pointing at a different binary than our formula.
    repo_text = _read("homebrew/vllm-swift.rb")
    repo_sha = re.search(r'arm64_tahoe:\s*"([0-9a-f]{64})"', repo_text)
    tap_sha = re.search(r'arm64_tahoe:\s*"([0-9a-f]{64})"', tap_text)
    if repo_sha and tap_sha:
        assert repo_sha.group(1) == tap_sha.group(1), (
            f"Tap arm64_tahoe bottle SHA = {tap_sha.group(1)} "
            f"does not match repo formula SHA = {repo_sha.group(1)}. "
            f"Push the updated formula to TheTom/homebrew-tap."
        )


def test_bottle_wrapper_template_uses_substitution_placeholder():
    """The bottle wrapper template inside build_bottle.sh MUST use
    `__VERSION__` (substituted by `sed` at build time) for its
    `version` subcommand — NOT a hardcoded version string. Hardcoding
    is the exact bug we shipped on 0.6.0 → 0.6.3 (wrapper still said
    0.6.0 because the heredoc was single-quoted and never substituted).
    """
    text = _read("scripts/build_bottle.sh")
    # Inside the WRAPPER heredoc, find the `version)` case branch's echo.
    # Anchor on the case label so we don't match unrelated "vllm-swift X"
    # strings elsewhere in the wrapper (banner lines, env notes, etc.).
    m = re.search(
        r'version\)\s*\n\s*echo\s+"vllm-swift\s+(\S+)"',
        text,
    )
    assert m, 'build_bottle.sh wrapper has no `echo "vllm-swift X"` line'
    placeholder = m.group(1)
    assert placeholder == "__VERSION__", (
        f"Bottle wrapper bakes hardcoded version {placeholder!r}. "
        f"Use __VERSION__ instead so the sed substitution in "
        f"build_bottle.sh keeps the wrapper in lockstep with VERSION="
    )
    # And: the substitution step must actually exist.
    assert "sed -i" in text and "__VERSION__" in text, (
        "build_bottle.sh wrapper uses __VERSION__ placeholder but the "
        "`sed -i` substitution step is missing — the bottle will ship "
        "a literal '__VERSION__' in its version output."
    )
