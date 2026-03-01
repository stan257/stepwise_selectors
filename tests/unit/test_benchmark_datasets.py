import numpy as np

from benchmarks.datasets import build_dataset


def test_synthetic_support_recovery_kind_builds_dataset():
    cfg = {
        "kind": "synthetic_support_recovery",
        "name": "test_case",
        "seed": 123,
        "n_samples": 120,
        "n_features": 24,
        "support_size": 5,
        "noise_std": 0.4,
        "signal_scale": 2.0,
        "correlation": 0.7,
        "train_fraction": 0.6,
        "val_fraction": 0.2,
    }
    ds = build_dataset(cfg)

    assert ds.name == "test_case"
    assert ds.X_train.shape[1] == 24
    assert ds.true_support.shape == (5,)
    assert np.all(np.diff(ds.true_support) >= 0)


def test_support_seed_keeps_true_support_fixed_across_run_seeds():
    base_cfg = {
        "kind": "synthetic_support_recovery",
        "name": "fixed_support",
        "n_samples": 100,
        "n_features": 20,
        "support_size": 4,
        "noise_std": 0.6,
        "signal_scale": 1.8,
        "correlation": 0.5,
        "support_seed": 999,
        "train_fraction": 0.6,
        "val_fraction": 0.2,
    }

    cfg_a = dict(base_cfg)
    cfg_a["seed"] = 1
    cfg_b = dict(base_cfg)
    cfg_b["seed"] = 2

    ds_a = build_dataset(cfg_a)
    ds_b = build_dataset(cfg_b)

    np.testing.assert_array_equal(ds_a.true_support, ds_b.true_support)


def test_synthetic_support_recovery_with_twins_and_nonlinear_term_builds():
    cfg = {
        "kind": "synthetic_support_recovery",
        "name": "twins_nonlinear",
        "seed": 321,
        "n_samples": 160,
        "n_features": 30,
        "support_size": 5,
        "noise_std": 0.5,
        "signal_scale": 2.2,
        "correlation": 0.6,
        "twin_decoys_per_signal": 1,
        "twin_strength": 0.98,
        "twin_noise_std": 0.1,
        "nonlinear_strength": 0.7,
        "train_fraction": 0.6,
        "val_fraction": 0.2,
    }
    ds = build_dataset(cfg)

    assert ds.X_train.shape == (96, 30)
    assert ds.X_val.shape == (32, 30)
    assert ds.X_test.shape == (32, 30)
    assert ds.true_support.shape == (5,)
