"""The analysis -> plotting handshake: deterministic paths and manifests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common import manifest as manifest_mod
from src.common import paths as paths_mod
from src.common.manifest import read_manifest, require_manifest, write_manifest
from src.common.paths import cache_dir, data_dir, figures_dir


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Repoints the four roots at a tmp dir so tests never touch real output."""
    for name, sub in (("DATA_ROOT", "data"), ("FIGURES_ROOT", "figures"),
                      ("CACHE_ROOT", "cache")):
        monkeypatch.setattr(paths_mod, name, tmp_path / sub)
    monkeypatch.setattr(manifest_mod, "data_dir", paths_mod.data_dir)
    return tmp_path


def test_paths_are_deterministic(sandbox):
    """The property the old `resolve_experiment_dir` lacked: calling twice with
    the same name returns the same directory instead of forking a `_1` sibling
    that the plotting side could never find."""
    first = data_dir("04_pca", "main", create=True)
    second = data_dir("04_pca", "main", create=True)
    assert first == second
    assert first.name == "main" and first.parent.name == "04_pca"


def test_tag_separates_variants(sandbox):
    assert data_dir("04_pca", "main") != data_dir("04_pca", "sweep")


def test_data_and_figures_are_separate_trees(sandbox):
    d, f = data_dir("04_pca", "main"), figures_dir("04_pca", "main")
    assert d != f
    assert "data" in d.parts and "figures" in f.parts


def test_no_create_means_no_mkdir(sandbox):
    """A plotting script resolving a path must not conjure an empty directory --
    that would turn 'the analysis never ran' into a confusing empty read."""
    assert not data_dir("99_missing", "main").exists()


@pytest.mark.parametrize("slug,tag", [("", "main"), ("04_pca", ""),
                                       ("a/b", "main"), ("04_pca", "..")])
def test_rejects_path_traversal(sandbox, slug, tag):
    with pytest.raises(ValueError):
        data_dir(slug, tag)


def test_manifest_roundtrip(sandbox):
    out = data_dir("04_pca", "main", create=True)
    (out / "pcs.npz").write_bytes(b"")
    src = sandbox / "cotrain_test_global_cond.pt"
    src.write_bytes(b"x" * 7)

    write_manifest(out, slug="04_pca", tag="main", schema_version=1,
                   params={"n_components": 8}, inputs=[src], outputs=[out / "pcs.npz"])

    got = read_manifest(out)
    assert got["schema_version"] == 1
    assert got["params"]["n_components"] == 8
    assert got["outputs"] == ["pcs.npz"]
    assert got["inputs"][0]["exists"] is True and got["inputs"][0]["bytes"] == 7


def test_missing_input_is_recorded_not_dropped(sandbox):
    """A manifest that silently omits an input understates what a run depended
    on, which is worse than recording that it was absent."""
    out = data_dir("04_pca", "main", create=True)
    write_manifest(out, slug="04_pca", tag="main", schema_version=1,
                   params={}, inputs=[sandbox / "nope.pt"])
    assert read_manifest(out)["inputs"][0]["exists"] is False


def test_require_manifest_names_the_command_to_run(sandbox):
    with pytest.raises(SystemExit) as e:
        require_manifest("04_pca", "main", schema_version=1)
    assert "analysis/04_pca.py --tag main" in str(e.value)


def test_require_manifest_rejects_a_half_finished_run(sandbox):
    """The manifest is written last, so a directory without one means the
    analysis died partway -- its artifacts must not be drawn as if complete."""
    data_dir("04_pca", "main", create=True)
    with pytest.raises(SystemExit) as e:
        require_manifest("04_pca", "main", schema_version=1)
    assert "did not finish" in str(e.value)


def test_require_manifest_rejects_a_stale_schema(sandbox):
    out = data_dir("04_pca", "main", create=True)
    write_manifest(out, slug="04_pca", tag="main", schema_version=1, params={})
    with pytest.raises(SystemExit) as e:
        require_manifest("04_pca", "main", schema_version=2)
    assert "schema_version 1" in str(e.value)


def test_require_manifest_accepts_a_matching_run(sandbox):
    out = data_dir("04_pca", "main", create=True)
    write_manifest(out, slug="04_pca", tag="main", schema_version=3, params={"k": 1})
    directory, got = require_manifest("04_pca", "main", schema_version=3)
    assert directory == out and got["params"] == {"k": 1}


def test_cache_is_its_own_tree(sandbox):
    assert cache_dir("umap").parent.name == "cache"
