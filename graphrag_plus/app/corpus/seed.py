"""Seed the committed demo corpus into a (possibly fresh) corpora dir.

Both deployment targets start with an empty writable storage location —
Vercel's per-instance ``/tmp`` and a freshly-provisioned Render disk — so
the dashboard would render an empty graph until someone ingests. This
module copies the repo's committed ``corpus_demo_nova`` into the active
corpora directory on startup so there's always something to show.

The copy is idempotent (skips if the target already exists) and locates
the source relative to the installed package, so it works regardless of
the process working directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import graphrag_plus
from graphrag_plus.app.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEMO_CORPUS_ID = "corpus_demo_nova"

# The committed demo lives alongside the package: <pkg>/data/corpora/<id>.
_PACKAGE_ROOT = Path(graphrag_plus.__file__).resolve().parent
_DEMO_SOURCE = _PACKAGE_ROOT / "data" / "corpora" / DEMO_CORPUS_ID


def seed_demo_corpus(corpora_dir: Path | str) -> bool:
    """Copy the demo corpus into ``corpora_dir`` if it isn't already there.

    Returns ``True`` when a copy happened, ``False`` otherwise (already
    present, or the source is missing). Never raises on a missing source —
    a deployment without the demo files should still boot.
    """
    target = Path(corpora_dir) / DEMO_CORPUS_ID
    if target.exists():
        return False
    if not _DEMO_SOURCE.is_dir():
        logger.warning("demo_seed.source_missing path=%s", _DEMO_SOURCE)
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_DEMO_SOURCE, target)
    logger.info("demo_seed.copied into=%s", target.parent)
    return True
