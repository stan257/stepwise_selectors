from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.market_chunks import (
    MarketChunkConfig,
    generate_market_gram_chunks_known,
    generate_market_gram_chunks_unknown,
    load_market_chunk_dataset,
    save_market_chunk_dataset,
)


def _small_config() -> MarketChunkConfig:
    return MarketChunkConfig(
        seed=12345,
        n_chunks=6,
        bars_per_chunk=180,
        warmup_bars=80,
        n_features=32,
        n_regimes=3,
        support_size=6,
        signal_scale=0.28,
        target_noise_std=0.01,
    )


def _rss_for_support(gram_data, support: tuple[int, ...]) -> float:
    if not support:
        return float(gram_data.y_norm)
    idx = np.array(support, dtype=int)
    gram_ss = gram_data.gram[np.ix_(idx, idx)]
    cov_s = gram_data.cov[idx]
    beta = np.linalg.pinv(gram_ss) @ cov_s
    rss = (
        gram_data.y_norm
        - 2.0 * float(cov_s @ beta)
        + float(beta @ (gram_ss @ beta))
    )
    return float(max(rss, 0.0))


def test_market_chunk_known_is_deterministic():
    cfg = _small_config()
    left = generate_market_gram_chunks_known(cfg)
    right = generate_market_gram_chunks_known(cfg)

    assert left.feature_names == right.feature_names
    assert left.chunk_ranges == right.chunk_ranges
    assert left.regime_by_chunk == right.regime_by_chunk
    assert left.support_by_chunk == right.support_by_chunk
    assert left.beta_by_chunk is not None
    assert right.beta_by_chunk is not None

    for g_l, g_r in zip(left.gram_chunks, right.gram_chunks):
        np.testing.assert_allclose(g_l.gram, g_r.gram, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(g_l.cov, g_r.cov, atol=0.0, rtol=0.0)
        assert g_l.y_norm == g_r.y_norm
        assert g_l.n_samples == g_r.n_samples

    for b_l, b_r in zip(left.beta_by_chunk, right.beta_by_chunk):
        np.testing.assert_allclose(b_l, b_r, atol=0.0, rtol=0.0)


def test_market_chunk_known_shape_contract():
    cfg = _small_config()
    data = generate_market_gram_chunks_known(cfg)
    assert len(data.gram_chunks) == cfg.n_chunks
    assert len(data.chunk_ranges) == cfg.n_chunks
    assert len(data.regime_by_chunk) == cfg.n_chunks
    assert len(data.feature_names) == cfg.n_features
    for chunk in data.gram_chunks:
        assert chunk.gram.shape == (cfg.n_features, cfg.n_features)
        assert chunk.cov.shape == (cfg.n_features,)
        assert chunk.n_samples == cfg.bars_per_chunk


def test_market_chunk_known_truth_metadata_matches_support():
    cfg = _small_config()
    data = generate_market_gram_chunks_known(cfg)
    assert data.support_by_chunk is not None
    assert data.beta_by_chunk is not None
    for support, beta in zip(data.support_by_chunk, data.beta_by_chunk):
        assert len(support) == cfg.support_size
        assert len(set(support)) == cfg.support_size
        assert all(0 <= idx < cfg.n_features for idx in support)
        nz = tuple(int(i) for i in np.flatnonzero(np.abs(beta) > 1e-14).tolist())
        assert nz == support


def test_market_chunk_unknown_hides_truth_by_default_and_can_expose():
    cfg = _small_config()
    hidden = generate_market_gram_chunks_unknown(cfg)
    assert hidden.support_by_chunk is None
    assert hidden.beta_by_chunk is None

    exposed = generate_market_gram_chunks_unknown(cfg, expose_truth=True)
    assert exposed.support_by_chunk is not None
    assert exposed.beta_by_chunk is not None
    assert len(exposed.support_by_chunk) == cfg.n_chunks
    assert len(exposed.beta_by_chunk) == cfg.n_chunks


def test_market_chunk_known_has_regime_driven_support_variation():
    cfg = MarketChunkConfig(
        seed=22,
        n_chunks=10,
        bars_per_chunk=120,
        warmup_bars=60,
        n_features=36,
        n_regimes=5,
        support_size=7,
        support_overlap_ratio=0.0,
        signal_scale=0.25,
    )
    data = generate_market_gram_chunks_known(cfg)
    assert data.support_by_chunk is not None
    unique_supports = {tuple(s) for s in data.support_by_chunk}
    assert len(unique_supports) > 1
    assert any(
        data.support_by_chunk[i] != data.support_by_chunk[i - 1]
        for i in range(1, len(data.support_by_chunk))
    )


def test_market_chunk_centering_and_finiteness_sanity():
    cfg = _small_config()
    data = generate_market_gram_chunks_known(cfg)
    for chunk in data.gram_chunks:
        assert np.isfinite(chunk.gram).all()
        assert np.isfinite(chunk.cov).all()
        assert np.isfinite(chunk.y_norm)

    feat_means = data.meta["max_abs_centered_feature_mean_per_chunk"]
    target_means = data.meta["max_abs_centered_target_mean_per_chunk"]
    assert isinstance(feat_means, list)
    assert isinstance(target_means, list)
    assert len(feat_means) == cfg.n_chunks
    assert len(target_means) == cfg.n_chunks
    assert max(float(v) for v in feat_means) < 1e-10
    assert max(float(v) for v in target_means) < 1e-10


def test_market_chunk_known_true_support_beats_random_support_most_chunks():
    cfg = _small_config()
    data = generate_market_gram_chunks_known(cfg)
    assert data.support_by_chunk is not None

    rng = np.random.default_rng(991)
    wins = 0
    for chunk_data, true_support in zip(data.gram_chunks, data.support_by_chunk):
        true_rss = _rss_for_support(chunk_data, true_support)
        random_rss = []
        for _ in range(8):
            candidate = tuple(
                sorted(
                    int(v)
                    for v in rng.choice(
                        cfg.n_features, size=len(true_support), replace=False
                    ).tolist()
                )
            )
            random_rss.append(_rss_for_support(chunk_data, candidate))
        if true_rss < float(np.median(np.asarray(random_rss, dtype=float))):
            wins += 1

    assert wins >= int(math.ceil(0.67 * cfg.n_chunks))


def test_market_chunk_npz_roundtrip_preserves_core_payload(tmp_path):
    cfg = _small_config()
    original = generate_market_gram_chunks_unknown(cfg, expose_truth=True)
    out = tmp_path / "market_chunks_unknown.npz"
    save_market_chunk_dataset(original, out)
    loaded = load_market_chunk_dataset(out)

    assert loaded.feature_names == original.feature_names
    assert loaded.chunk_ranges == original.chunk_ranges
    assert loaded.regime_by_chunk == original.regime_by_chunk
    assert loaded.support_by_chunk == original.support_by_chunk
    assert loaded.meta == original.meta
    assert loaded.beta_by_chunk is not None
    assert original.beta_by_chunk is not None

    for chunk_left, chunk_right in zip(original.gram_chunks, loaded.gram_chunks):
        np.testing.assert_allclose(chunk_left.gram, chunk_right.gram)
        np.testing.assert_allclose(chunk_left.cov, chunk_right.cov)
        assert chunk_left.y_norm == chunk_right.y_norm
        assert chunk_left.n_samples == chunk_right.n_samples

    for beta_left, beta_right in zip(original.beta_by_chunk, loaded.beta_by_chunk):
        np.testing.assert_allclose(beta_left, beta_right)


def test_market_chunk_meta_marks_last_chunk_as_oos():
    cfg = _small_config()
    data = generate_market_gram_chunks_unknown(cfg)
    assert data.meta["train_chunk_count_recommended"] == cfg.n_chunks - 1
    assert data.meta["oos_chunk_index_recommended"] == cfg.n_chunks - 1


def test_market_chunk_requires_at_least_two_chunks():
    with pytest.raises(ValueError, match="n_chunks must be >= 2"):
        generate_market_gram_chunks_known(MarketChunkConfig(n_chunks=1, n_regimes=1))
