"""The handshake between an analysis script and its plotting counterpart.

Every `data/<slug>/<tag>/` directory carries a `manifest.json` describing how it
was produced: schema version, input files (with size + mtime, so a stale rebuild
is detectable), every parameter, the git SHA, and the list of files written.

The plotting side calls `require_manifest()`, which fails with an actionable
message -- naming the exact command to run -- rather than letting a missing file
surface as a bare FileNotFoundError three frames deep.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from src.common.paths import data_dir


MANIFEST_NAME = "manifest.json"


def git_sha(repo_root: Optional[Path] = None) -> Optional[str]:
    """Short SHA of the current commit, or None outside a repo / with no
    commits yet. Never raises -- provenance is nice to have, not a hard
    dependency of any analysis."""
    from src.common.paths import REPO_ROOT

    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root or REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def describe_inputs(paths: Iterable[Path]) -> list[Dict[str, Any]]:
    """Records each input as {path, bytes, mtime}. Missing files are recorded
    with `exists: false` rather than skipped, so a manifest never silently
    understates what a run depended on."""
    described = []
    for p in paths:
        p = Path(p)
        entry: Dict[str, Any] = {"path": str(p)}
        if p.exists():
            st = p.stat()
            entry.update(exists=True, bytes=st.st_size,
                         mtime=datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat())
        else:
            entry["exists"] = False
        described.append(entry)
    return described


def write_manifest(out_dir: Path, *, slug: str, tag: str, schema_version: int,
                   params: Dict[str, Any], inputs: Sequence[Path] = (),
                   outputs: Sequence[Path] = (), extra: Optional[Dict[str, Any]] = None) -> Path:
    """Writes `<out_dir>/manifest.json`. Call this LAST, after the artifacts are
    on disk -- its presence is what marks a data directory as complete, so
    writing it early would advertise a half-finished run as usable."""
    payload: Dict[str, Any] = {
        "slug": slug,
        "tag": tag,
        "schema_version": schema_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "params": params,
        "inputs": describe_inputs(inputs),
        "outputs": sorted(str(Path(p).name) for p in outputs),
    }
    if extra:
        payload.update(extra)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return path


def read_manifest(directory: Path) -> Dict[str, Any]:
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"No {MANIFEST_NAME} in {directory}")
    return json.loads(path.read_text())


def require_manifest(slug: str, tag: str, *, schema_version: int) -> tuple[Path, Dict[str, Any]]:
    """Resolves `data/<slug>/<tag>/`, verifies it is complete and readable by
    this version of the plotting script, and returns (dir, manifest)."""
    directory = data_dir(slug, tag)
    rerun = f"python analysis/{slug}.py --tag {tag}"

    if not directory.exists():
        raise SystemExit(
            f"No analysis data at {directory}\n"
            f"Run the analysis first:\n    {rerun}")
    try:
        manifest = read_manifest(directory)
    except FileNotFoundError:
        raise SystemExit(
            f"{directory} exists but has no {MANIFEST_NAME} -- the analysis did not "
            f"finish (the manifest is written last).\nRe-run it:\n    {rerun}")

    found = manifest.get("schema_version")
    if found != schema_version:
        raise SystemExit(
            f"{directory} was written with schema_version {found}, but this plotting "
            f"script expects {schema_version}.\nRe-run the analysis:\n    {rerun}")
    return directory, manifest
