from __future__ import annotations

import functools
import pathlib
import subprocess
from typing import Any


DEFAULT_VERSION = 'dev'


def app_version(config: dict[str, Any] | None = None) -> str:
    if isinstance(config, dict):
        value = str(config.get('version') or '').strip()
        if value:
            return value
    return DEFAULT_VERSION


@functools.lru_cache(maxsize=1)
def git_commit() -> str:
    root = pathlib.Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
    except Exception:
        return 'unknown'
    commit = (result.stdout or '').strip()
    return commit if result.returncode == 0 and commit else 'unknown'


def build_metadata(config: dict[str, Any] | None = None) -> dict[str, str]:
    return {
        'version': app_version(config),
        'git_commit': git_commit(),
    }
