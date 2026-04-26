"""Installation check for GraphRAG++.

Run:
  python scripts/check_install.py
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


REQUIRED_IMPORTS = [
    "fastapi",
    "pydantic",
    "pydantic_settings",
    "networkx",
    "numpy",
    "sklearn",
    "rank_bm25",
    "torch",
    "pypdf",
    "httpx",
    "bs4",
]

PROJECT_IMPORTS = [
    "graphrag_plus",
    "graphrag_plus.app.cli",
    "graphrag_plus.app.pipeline",
]


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def check_imports(modules: list[str]) -> None:
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - fail fast with context
            fail(f"Import failed: {module} ({exc})")
    ok(f"Imports succeeded ({len(modules)} modules)")


def check_cli_health() -> None:
    # repo_root is the package directory (contains __init__.py + app/).
    # Running from repo_root can make `import graphrag_plus` fail unless installed.
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    env = dict(os.environ)

    # Help avoid locked-down temp directories in some Windows environments.
    tmp_dir = repo_root / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("TEMP", str(tmp_dir))
    env.setdefault("TMP", str(tmp_dir))

    cmd = [sys.executable, "-m", "graphrag_plus.app.cli", "health_check"]
    proc = subprocess.run(cmd, cwd=str(workspace_root), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        fail(f"CLI health_check failed (exit {proc.returncode}): {proc.stderr.strip()}")
    try:
        payload = json.loads(proc.stdout)
    except Exception as exc:  # noqa: BLE001
        fail(f"CLI health_check did not return JSON ({exc}): {proc.stdout[:200]}")
    status = payload.get("status")
    if status not in {"healthy", "degraded"}:
        fail(f"Unexpected health_check status: {status}")
    ok(f"CLI health_check returned status={status}")


def main() -> None:
    print("GraphRAG++ install check")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")

    check_imports(REQUIRED_IMPORTS)
    check_imports(PROJECT_IMPORTS)
    check_cli_health()
    ok("Installation check complete")


if __name__ == "__main__":
    main()
