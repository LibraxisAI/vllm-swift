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
    assert m, "homebrew/vllm-swift.rb has no `echo \"vllm-swift X\"` line"
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
    assert m, "build_bottle.sh wrapper has no `echo \"vllm-swift X\"` line"
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
