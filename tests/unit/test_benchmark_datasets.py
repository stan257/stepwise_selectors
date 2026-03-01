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
