from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks.microstructure_chunks import (
    CANONICAL_MICROSTRUCTURE_FEATURE_NAMES,
    MicrostructureChunkConfig,
    MicrostructureFeatureRegistry,
    MicrostructureFeatureSpec,
    apply_microstructure_preset,
    build_microstructure_feature_table,
    build_microstructure_target,
    default_microstructure_feature_registry,
    generate_microstructure_gram_chunks_known,
    generate_microstructure_gram_chunks_unknown,
    load_microstructure_chunk_dataset,
    microstructure_feature_registry,
    save_microstructure_chunk_dataset,
    simulate_microstructure_observables,
    supported_microstructure_feature_counts,
)


ROOT = Path(__file__).resolve().parents[2]
_LOOKBACK_EVENTS = 40
_PRE_REFACTOR_HASHES = {
    "mid_price": "207d8e41eeb56c72417c06873b860c2d42d6fb60a9cc25600dccdf6bd292e4c7",
    "spread_ticks": "028957d74117adfcc73256c636337a569d0baf34b760b34678cac8ce631eab52",
    "q_bid": "62038007a3a512d36d9109c88e107891c4691035f5f258f9ee1f857291e83026",
    "q_ask": "424623e6444701131f954ff7189f4a5da47739406e9387dc665f82c039aa2162",
    "signed_trade": "d9fb6e63fbf97ae9e2fd29dea7c3880a255b7b45b34f25f14c01a1678188d146",
    "signed_volume": "3bfe600ccbffbb1ca472e310f917ef722864977718bb88736242a6063d2055f9",
    "price_changed": "e319082c23faa07865238e06844382d181ad3cdab44bcfaea7836fd601dba857",
    "spread_changed": "08251bcb57739d4c63d5b5e3a54ab1d8866158669b242e668a66b86340724b1b",
    "depletion_dir": "e319082c23faa07865238e06844382d181ad3cdab44bcfaea7836fd601dba857",
    "ofi": "f1dd4e09f8d3d4cf60a46c704b8560141278985642343561b9f8252957742b69",
    "X": "a41767b9ad3c370716fd4cd9b1c31c25502dc43b495a8f37c2b970008e79b693",
    "y_base": "423fdfd63cdb0a3bd7cb7c798b1dc532e41d0d98234fad78e2cba05b49f97911",
    "known_dataset": "d42cfb904916ad4956b24ad9b8bf606a1448089828ee74bacd8bfbcb2561c6a9",
    "unknown_dataset_exposed": "f63b042eaa2727fe6a3d0bcbf99fc7f918d3711974fbeeff612d8ce65d4ca07d",
}



def _small_config() -> MicrostructureChunkConfig:
    return MicrostructureChunkConfig(
        seed=12345,
        n_chunks=6,
        events_per_chunk=180,
        warmup_events=90,
        n_features=64,
        n_regimes=3,
        target_horizon_events=6,
        support_size=6,
        signal_scale=0.24,
        target_noise_std=0.006,
    )



def _rss_for_support(gram_data, support: tuple[int, ...]) -> float:
    if not support:
        return float(gram_data.y_norm)
    idx = np.array(support, dtype=int)
    gram_ss = gram_data.gram[np.ix_(idx, idx)]
    cov_s = gram_data.cov[idx]
    beta = np.linalg.pinv(gram_ss) @ cov_s
    rss = gram_data.y_norm - 2.0 * float(cov_s @ beta) + float(beta @ (gram_ss @ beta))
    return float(max(rss, 0.0))



def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).view(np.uint8)).hexdigest()



def _dataset_digest(dataset) -> str:
    dig = hashlib.sha256()
    for chunk in dataset.gram_chunks:
        dig.update(np.ascontiguousarray(chunk.gram).view(np.uint8))
        dig.update(np.ascontiguousarray(chunk.cov).view(np.uint8))
        dig.update(np.asarray([chunk.y_norm], dtype=float).view(np.uint8))
        dig.update(np.asarray([chunk.n_samples], dtype=np.int64).view(np.uint8))
    dig.update(json.dumps(dataset.meta, sort_keys=True).encode())
    if dataset.support_by_chunk is not None:
        dig.update(json.dumps(dataset.support_by_chunk).encode())
    if dataset.beta_by_chunk is not None:
        for beta in dataset.beta_by_chunk:
            dig.update(np.ascontiguousarray(beta).view(np.uint8))
    return dig.hexdigest()



def test_microstructure_chunk_known_is_deterministic():
    cfg = _small_config()
    left = generate_microstructure_gram_chunks_known(cfg)
    right = generate_microstructure_gram_chunks_known(cfg)

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



def test_microstructure_chunk_shape_contract():
    cfg = _small_config()
    data = generate_microstructure_gram_chunks_known(cfg)
    assert len(data.gram_chunks) == cfg.n_chunks
    assert len(data.chunk_ranges) == cfg.n_chunks
    assert len(data.regime_by_chunk) == cfg.n_chunks
    assert len(data.feature_names) == cfg.n_features
    for chunk in data.gram_chunks:
        assert chunk.gram.shape == (cfg.n_features, cfg.n_features)
        assert chunk.cov.shape == (cfg.n_features,)
        assert chunk.n_samples == cfg.events_per_chunk



def test_microstructure_chunk_known_truth_metadata_matches_support():
    cfg = _small_config()
    data = generate_microstructure_gram_chunks_known(cfg)
    assert data.support_by_chunk is not None
    assert data.beta_by_chunk is not None
    for support, beta in zip(data.support_by_chunk, data.beta_by_chunk):
        assert len(support) == cfg.support_size
        assert len(set(support)) == cfg.support_size
        assert all(0 <= idx < cfg.n_features for idx in support)
        nz = tuple(int(i) for i in np.flatnonzero(np.abs(beta) > 1e-14).tolist())
        assert nz == support



def test_microstructure_chunk_unknown_hides_truth_by_default_and_can_expose():
    cfg = _small_config()
    hidden = generate_microstructure_gram_chunks_unknown(cfg)
    assert hidden.support_by_chunk is None
    assert hidden.beta_by_chunk is None

    exposed = generate_microstructure_gram_chunks_unknown(cfg, expose_truth=True)
    assert exposed.support_by_chunk is not None
    assert exposed.beta_by_chunk is not None
    assert len(exposed.support_by_chunk) == cfg.n_chunks
    assert len(exposed.beta_by_chunk) == cfg.n_chunks


def test_unknown_wide_support_preset_widens_hidden_support():
    base = _small_config()
    preset = apply_microstructure_preset(base, preset="unknown_wide_support")

    assert preset.unknown_support_size == 20
    assert preset.unknown_support_overlap_ratio == pytest.approx(0.15)
    assert preset.unknown_chunk_jitter == 5
    assert preset.signal_scale == pytest.approx(0.155)

    exposed = generate_microstructure_gram_chunks_unknown(preset, expose_truth=True)
    assert exposed.support_by_chunk is not None
    assert all(len(support) == 20 for support in exposed.support_by_chunk)


def test_unknown_multiscale_192_presets_expand_feature_bank():
    base = _small_config()

    multiscale = apply_microstructure_preset(base, preset="unknown_multiscale_192")
    assert multiscale.n_features == 192
    assert multiscale.n_regimes == min(base.n_chunks, 7)
    assert multiscale.target_horizon_events == 10
    assert multiscale.unknown_support_size == 24

    rotating = apply_microstructure_preset(
        base, preset="unknown_multiscale_192_rapid_rotation"
    )
    assert rotating.n_features == 192
    assert rotating.n_regimes == min(base.n_chunks, 8)
    assert rotating.target_horizon_events == 12
    assert rotating.unknown_support_size == 32
    assert rotating.unknown_chunk_jitter == 10

    exposed = generate_microstructure_gram_chunks_unknown(
        multiscale, expose_truth=True
    )
    assert len(exposed.feature_names) == 192
    assert exposed.support_by_chunk is not None
    assert all(len(support) == 24 for support in exposed.support_by_chunk)


def test_microstructure_chunk_known_has_regime_driven_support_variation():
    cfg = MicrostructureChunkConfig(
        seed=22,
        n_chunks=10,
        events_per_chunk=150,
        warmup_events=80,
        n_features=64,
        n_regimes=5,
        target_horizon_events=6,
        support_size=7,
        support_overlap_ratio=0.0,
        signal_scale=0.25,
    )
    data = generate_microstructure_gram_chunks_known(cfg)
    assert data.support_by_chunk is not None
    unique_supports = {tuple(s) for s in data.support_by_chunk}
    assert len(unique_supports) > 1
    assert any(
        data.support_by_chunk[i] != data.support_by_chunk[i - 1]
        for i in range(1, len(data.support_by_chunk))
    )



def test_microstructure_chunk_centering_and_finiteness_sanity():
    cfg = _small_config()
    data = generate_microstructure_gram_chunks_known(cfg)
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



def test_microstructure_chunk_representativeness_signals_hold():
    cfg = MicrostructureChunkConfig(
        seed=77,
        n_chunks=8,
        events_per_chunk=220,
        warmup_events=120,
        n_features=64,
        n_regimes=4,
        target_horizon_events=8,
        support_size=8,
        signal_scale=0.24,
        target_noise_std=0.005,
    )
    data = generate_microstructure_gram_chunks_known(cfg)
    spreads = [float(v) for v in data.meta["chunk_mean_spread"]]
    sign_autocorr = [float(v) for v in data.meta["chunk_trade_sign_autocorr_lag1"]]
    ofi_corr = [float(v) for v in data.meta["chunk_ofi_target_corr"]]
    micro_corr = [float(v) for v in data.meta["chunk_microdev_target_corr"]]
    target_std = np.asarray(data.meta["chunk_target_std"], dtype=float)

    assert min(spreads) > 0.0
    assert sum(v > 0.0 for v in sign_autocorr) >= int(math.ceil(0.60 * cfg.n_chunks))
    assert sum((o > 0.0) or (m > 0.0) for o, m in zip(ofi_corr, micro_corr)) >= int(
        math.ceil(0.60 * cfg.n_chunks)
    )
    spread_corr = np.corrcoef(target_std, np.asarray(spreads, dtype=float))[0, 1]
    assert spread_corr > 0.0



def test_microstructure_chunk_known_true_support_beats_random_support_most_chunks():
    cfg = MicrostructureChunkConfig(
        seed=91,
        n_chunks=6,
        events_per_chunk=200,
        warmup_events=100,
        n_features=64,
        n_regimes=3,
        target_horizon_events=6,
        support_size=6,
        signal_scale=0.30,
        target_noise_std=0.004,
    )
    data = generate_microstructure_gram_chunks_known(cfg)
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



def test_microstructure_observables_and_dataset_match_pre_refactor_hashes():
    cfg = _small_config()
    observables = simulate_microstructure_observables(cfg)
    assert _hash_array(observables.mid_price) == _PRE_REFACTOR_HASHES["mid_price"]
    assert _hash_array(observables.spread_ticks) == _PRE_REFACTOR_HASHES["spread_ticks"]
    assert _hash_array(observables.q_bid) == _PRE_REFACTOR_HASHES["q_bid"]
    assert _hash_array(observables.q_ask) == _PRE_REFACTOR_HASHES["q_ask"]
    assert _hash_array(observables.signed_trade) == _PRE_REFACTOR_HASHES["signed_trade"]
    assert _hash_array(observables.signed_volume) == _PRE_REFACTOR_HASHES["signed_volume"]
    assert _hash_array(observables.price_changed) == _PRE_REFACTOR_HASHES["price_changed"]
    assert _hash_array(observables.spread_changed) == _PRE_REFACTOR_HASHES["spread_changed"]
    assert _hash_array(observables.depletion_dir) == _PRE_REFACTOR_HASHES["depletion_dir"]
    assert _hash_array(observables.ofi) == _PRE_REFACTOR_HASHES["ofi"]

    table = build_microstructure_feature_table(observables)
    valid_stop = observables.mid_price.shape[0] - cfg.target_horizon_events
    usable_start = _LOOKBACK_EVENTS + cfg.warmup_events
    usable_end = usable_start + cfg.n_chunks * cfg.events_per_chunk
    X = np.ascontiguousarray(table.matrix[:valid_stop][usable_start:usable_end], dtype=float)
    y_base_full = build_microstructure_target(
        observables, horizon_events=cfg.target_horizon_events
    )
    y_base = np.asarray(y_base_full[usable_start:usable_end], dtype=float)

    assert _hash_array(X) == _PRE_REFACTOR_HASHES["X"]
    assert _hash_array(y_base) == _PRE_REFACTOR_HASHES["y_base"]

    known = generate_microstructure_gram_chunks_known(cfg)
    unknown = generate_microstructure_gram_chunks_unknown(cfg, expose_truth=True)
    assert _dataset_digest(known) == _PRE_REFACTOR_HASHES["known_dataset"]
    assert _dataset_digest(unknown) == _PRE_REFACTOR_HASHES["unknown_dataset_exposed"]



def test_microstructure_registry_matches_canonical_order_and_subset_generation():
    cfg = _small_config()
    observables = simulate_microstructure_observables(cfg)
    registry = default_microstructure_feature_registry()

    assert registry.feature_names() == CANONICAL_MICROSTRUCTURE_FEATURE_NAMES
    assert len(registry.feature_names()) == 64

    subset = build_microstructure_feature_table(
        observables,
        names=("ofi_mean_20", "microdev_mean_10", "imbalance_lag_3"),
    )
    assert subset.feature_names == (
        "ofi_mean_20",
        "microdev_mean_10",
        "imbalance_lag_3",
    )
    assert subset.matrix.shape[1] == 3

    family_subset = build_microstructure_feature_table(
        observables,
        families=("imbalance", "flow"),
    )
    assert set(family_subset.feature_families) == {"imbalance", "flow"}
    assert all(
        family in {"imbalance", "flow"} for family in family_subset.feature_families
    )


def test_microstructure_registry_supports_expanded_feature_counts_and_new_families():
    assert supported_microstructure_feature_counts() == (64, 128, 192)

    registry_128 = microstructure_feature_registry(128)
    registry_192 = microstructure_feature_registry(192)

    assert len(registry_128.feature_names()) == 128
    assert len(registry_192.feature_names()) == 192
    assert registry_128.feature_names()[:64] == CANONICAL_MICROSTRUCTURE_FEATURE_NAMES
    assert registry_192.feature_names()[:64] == CANONICAL_MICROSTRUCTURE_FEATURE_NAMES
    assert "depth" in set(registry_192.families())
    assert "intensity" in set(registry_192.families())


def test_microstructure_feature_table_supports_expanded_registry():
    cfg = MicrostructureChunkConfig(
        seed=12345,
        n_chunks=4,
        events_per_chunk=80,
        warmup_events=40,
        n_features=192,
        n_regimes=2,
        target_horizon_events=6,
        support_size=6,
    )
    observables = simulate_microstructure_observables(cfg)
    registry = microstructure_feature_registry(192)
    table = build_microstructure_feature_table(observables, registry=registry)

    assert table.matrix.shape[1] == 192
    assert "depth_total_log_mean_80" in table.name_to_index
    assert "trade_rate_80" in table.name_to_index
    assert "depth_pressure_x_flow_20" in table.name_to_index


def test_microstructure_observables_surface_and_feature_table_accessors():
    cfg = _small_config()
    observables = simulate_microstructure_observables(cfg)
    as_dict = observables.as_dict()
    assert "mid_price" in as_dict
    assert "signed_trade_abs" in as_dict
    np.testing.assert_allclose(observables.series("mid_price"), observables.mid_price)
    np.testing.assert_allclose(
        observables.series("signed_trade_abs"), np.abs(observables.signed_trade)
    )

    table = build_microstructure_feature_table(
        observables,
        names=("ofi_mean_10", "mid_ret_vol_20"),
    )
    np.testing.assert_allclose(table.column("ofi_mean_10"), table.matrix[:, 0])
    as_feature_dict = table.as_dict()
    assert set(as_feature_dict) == {"ofi_mean_10", "mid_ret_vol_20"}
    selected = table.select(names=("mid_ret_vol_20",))
    assert selected.feature_names == ("mid_ret_vol_20",)



def test_microstructure_custom_registry_feature_can_be_built():
    cfg = _small_config()
    observables = simulate_microstructure_observables(cfg)
    custom_spec = MicrostructureFeatureSpec(
        name="signed_trade_minus_imbalance",
        family="custom",
        lookback=1,
        required_series=("signed_trade", "imbalance"),
        description="Signed trade minus imbalance.",
        builder=lambda obs: obs.signed_trade - obs.imbalance,
    )
    custom_registry = MicrostructureFeatureRegistry(specs=(custom_spec,))
    table = build_microstructure_feature_table(observables, registry=custom_registry)
    assert table.feature_names == ("signed_trade_minus_imbalance",)
    np.testing.assert_allclose(
        table.column("signed_trade_minus_imbalance"),
        observables.signed_trade - observables.imbalance,
    )



def test_microstructure_chunk_npz_roundtrip_preserves_core_payload(tmp_path):
    cfg = _small_config()
    original = generate_microstructure_gram_chunks_unknown(cfg, expose_truth=True)
    out = tmp_path / "microstructure_chunks_unknown.npz"
    save_microstructure_chunk_dataset(original, out)
    loaded = load_microstructure_chunk_dataset(out)

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



def test_microstructure_cli_smoke_roundtrip(tmp_path):
    out = tmp_path / "cli_microstructure_chunks.npz"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_microstructure_chunks.py",
            "--flavor",
            "unknown",
            "--n-chunks",
            "4",
            "--events-per-chunk",
            "80",
            "--warmup-events",
            "40",
            "--n-regimes",
            "2",
            "--output",
            str(out),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert out.exists()
    loaded = load_microstructure_chunk_dataset(out)
    assert len(loaded.gram_chunks) == 4
    assert len(loaded.feature_names) == 64
    assert loaded.support_by_chunk is None
    assert "Target horizon" in result.stdout



def test_microstructure_feature_inspection_cli_smoke(tmp_path):
    out = tmp_path / "microstructure_features_subset.npz"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_microstructure_features.py",
            "--n-chunks",
            "4",
            "--events-per-chunk",
            "80",
            "--warmup-events",
            "40",
            "--n-regimes",
            "2",
            "--families",
            "imbalance,flow",
            "--output",
            str(out),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert out.exists()
    with np.load(out, allow_pickle=False) as npz:
        feature_names = tuple(str(v) for v in np.asarray(npz["feature_names"], dtype=str))
        feature_families = tuple(str(v) for v in np.asarray(npz["feature_families"], dtype=str))
        matrix = np.asarray(npz["matrix"], dtype=float)
    assert matrix.shape[1] == len(feature_names)
    assert set(feature_families) == {"imbalance", "flow"}
    assert "Generated" in result.stdout



def test_microstructure_chunk_requires_valid_config():
    with pytest.raises(ValueError, match="n_chunks must be >= 2"):
        generate_microstructure_gram_chunks_known(
            MicrostructureChunkConfig(n_chunks=1, n_regimes=1)
        )

    with pytest.raises(ValueError, match="events_per_chunk must be > target_horizon_events"):
        generate_microstructure_gram_chunks_known(
            MicrostructureChunkConfig(events_per_chunk=8, target_horizon_events=8)
        )

    with pytest.raises(ValueError, match="n_features must be one of 64, 128, 192"):
        generate_microstructure_gram_chunks_known(
            MicrostructureChunkConfig(n_features=32)
        )
