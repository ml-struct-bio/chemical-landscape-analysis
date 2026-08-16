"""The UMAP cache is keyed by a hash of CONFIG, not of the embedding content.

That makes `EmbeddingSpec.cache_key` load-bearing: two fits whose key and
hyperparameters agree are treated as the same fit, and one silently gets the
other's projection. Every other fingerprinted field (n_neighbors, min_dist,
metric, pca_dim, scale, seed, n_points, n_dims -- see
`src/analysis/umap_projection.py`) is identical across the layers and timesteps
of a single extraction, so the key is the ONLY thing separating them.
"""
import itertools

from src.analysis.corpus import EmbeddingSpec
from src.analysis.umap_cache import umap_fingerprint


STREAMS = ("x_hidden_mean", "y_hidden_mean")
LAYERS = range(12)
TIMESTEPS = (0.001, 0.5, 1.0)


def _decoder_specs():
    return [EmbeddingSpec(kind="decoder", stream=s, layer=l, timestep=t)
            for s, l, t in itertools.product(STREAMS, LAYERS, TIMESTEPS)]


def _fingerprint(spec, n_points=75_000, n_dims=768):
    """The fields `project_and_store` hashes, with everything but the key fixed."""
    return umap_fingerprint({
        "embedding_key": spec.cache_key,
        "datasets": None,
        "n_neighbors": 30,
        "min_dist": 0.1,
        "metric": "cosine",
        "pca_dim": 50,
        "scale": True,
        "seed": 0,
        "n_points": n_points,
        "n_dims": n_dims,
    })


def test_every_decoder_spec_has_a_distinct_cache_key():
    specs = _decoder_specs()
    keys = [s.cache_key for s in specs]
    assert len(set(keys)) == len(specs), (
        "decoder cache keys collide -- fits from different layers/timesteps "
        "would share one cache entry")


def test_decoder_fits_of_one_extraction_do_not_share_a_fingerprint():
    """The regression this file exists for: 12 layers x 3 timesteps x 2 streams
    of one extraction all have the same shape, so only the key separates them."""
    specs = _decoder_specs()
    fingerprints = [_fingerprint(s) for s in specs]
    assert len(set(fingerprints)) == len(specs)


def test_encoder_and_ecfp_keys_are_unchanged():
    """These name the previous pipeline's expensive full-corpus fits. Changing
    them silently discards those cache entries and forces multi-hour refits."""
    assert EmbeddingSpec(kind="encoder").cache_key == "global_cond"
    assert EmbeddingSpec(kind="ecfp").cache_key == "ecfp"


def test_decoder_key_is_distinct_from_the_corpus_keys():
    for spec in _decoder_specs():
        assert spec.cache_key not in ("global_cond", "ecfp")


def test_cache_key_tracks_the_output_tag():
    """A cache entry that cannot be traced back to the `data/05_umap/<tag>/` it
    produced is not auditable after the fact."""
    for spec in _decoder_specs():
        assert spec.cache_key == spec.slug
