"""Run information written next to every result file (results/<name>/MANIFEST.txt)."""

from __future__ import annotations

import datetime
import pathlib
import platform
import shlex
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def command_line() -> str:
    """The command as run, with the script path relative to the repository root."""
    argv = list(sys.argv)
    p = pathlib.Path(argv[0]).resolve()
    if p.is_relative_to(REPO):
        argv[0] = str(p.relative_to(REPO))
    return "python " + " ".join(shlex.quote(a) for a in argv)


def write_manifest(out_dir: pathlib.Path, seed: int | None, extra: dict | None = None) -> None:
    """Write MANIFEST.txt: command, seed, git commit, working-tree state, versions, machine."""
    import numpy

    out_dir.mkdir(parents=True, exist_ok=True)
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--", "src", "tests", "cpp",
                 "experiments", "scripts", "docs", "CMakeLists.txt", "pyproject.toml")
    lines = [
        f"command: {command_line()}",
        f"seed: {seed if seed is not None else 'none (deterministic)'}",
        f"git_commit: {_git('rev-parse', 'HEAD')}",
        f"working_tree: {'modified code (uncommitted changes)' if dirty else 'clean code'}",
        f"date_utc: {datetime.datetime.now(datetime.UTC):%Y-%m-%dT%H:%M:%SZ}",
        f"python: {platform.python_version()}",
        f"numpy: {numpy.__version__}",
        f"machine: {platform.machine()} {platform.system()} {platform.release()}",
    ]
    for k, v in (extra or {}).items():
        lines.append(f"{k}: {v}")
    (out_dir / "MANIFEST.txt").write_text("\n".join(lines) + "\n")


def results_dir(name: str) -> pathlib.Path:
    d = REPO / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    return d
