"""Puts the repo root on sys.path so tests can `from src.common import ...`.

`src/` is deliberately a namespace package with no `__init__.py` (see
`test_src_has_no_init_files`), so there is nothing to install and nothing to
import-hook -- the root just has to be importable.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
