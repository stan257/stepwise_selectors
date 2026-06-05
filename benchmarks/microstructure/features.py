"""Inspectable feature registry for the microstructure benchmark pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .types import (
    MicrostructureFeatureSpec,
    MicrostructureFeatureTable,
    MicrostructureObservables,
)
from .utils import lag, rolling_mean, rolling_std

_SUPPORTED_FEATURE_COUNTS: tuple[int, ...] = (64, 128, 192)


@dataclass(frozen=True)
class MicrostructureFeatureRegistry:
    """Named registry of canonical microstructure feature specifications."""

    specs: tuple[MicrostructureFeatureSpec, ...]

    def __post_init__(self) -> None:
        names = [spec.name for spec in self.specs]
        if len(names) != len(set(names)):
            raise ValueError("Feature registry contains duplicate feature names.")

    def all_specs(self) -> tuple[MicrostructureFeatureSpec, ...]:
        return self.specs

    def feature_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def families(self) -> tuple[str, ...]:
        return tuple(spec.family for spec in self.specs)

    def spec(self, name: str) -> MicrostructureFeatureSpec:
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise KeyError(f"Unknown feature name: {name!r}")

    def select(
        self,
        *,
        names: tuple[str, ...] | None = None,
        families: tuple[str, ...] | None = None,
    ) -> "MicrostructureFeatureRegistry":
        name_filter = None if names is None else tuple(str(name) for name in names)
        family_filter = None if families is None else set(str(family) for family in families)
        specs_by_name = {spec.name: spec for spec in self.specs}
        if name_filter is not None:
            unknown = [name for name in name_filter if name not in specs_by_name]
            if unknown:
                raise KeyError(f"Unknown feature name(s): {unknown!r}")
        selected: list[MicrostructureFeatureSpec] = []
        if name_filter is None:
            for spec in self.specs:
                if family_filter is not None and spec.family not in family_filter:
                    continue
                selected.append(spec)
        else:
            for name in name_filter:
                spec = specs_by_name[name]
                if family_filter is not None and spec.family not in family_filter:
                    continue
                selected.append(spec)
        return MicrostructureFeatureRegistry(specs=tuple(selected))

    def build(self, observables: MicrostructureObservables) -> MicrostructureFeatureTable:
        columns = [spec.compute(observables) for spec in self.specs]
        if columns:
            matrix = np.column_stack(columns)
        else:
            matrix = np.zeros((observables.event_index.shape[0], 0), dtype=float)
        feature_names = tuple(spec.name for spec in self.specs)
        feature_families = tuple(spec.family for spec in self.specs)
        feature_descriptions = tuple(spec.description for spec in self.specs)
        name_to_index = {name: idx for idx, name in enumerate(feature_names)}
        return MicrostructureFeatureTable(
            matrix=np.ascontiguousarray(matrix, dtype=float),
            feature_names=feature_names,
            feature_families=feature_families,
            feature_descriptions=feature_descriptions,
            name_to_index=name_to_index,
        )


def supported_microstructure_feature_counts() -> tuple[int, ...]:
    return _SUPPORTED_FEATURE_COUNTS


def _custom_spec(
    *,
    name: str,
    family: str,
    lookback: int,
    required_series: tuple[str, ...],
    description: str,
    builder,
) -> MicrostructureFeatureSpec:
    return MicrostructureFeatureSpec(
        name=name,
        family=family,
        lookback=lookback,
        required_series=required_series,
        description=description,
        builder=builder,
    )


def _lag_spec(
    *,
    name: str,
    family: str,
    source: str,
    steps: int,
    description: str,
) -> MicrostructureFeatureSpec:
    return _custom_spec(
        name=name,
        family=family,
        lookback=steps,
        required_series=(source,),
        description=description,
        builder=lambda obs, source=source, steps=steps: lag(obs.series(source), steps),
    )


def _rolling_mean_spec(
    *,
    name: str,
    family: str,
    source: str,
    window: int,
    description: str,
) -> MicrostructureFeatureSpec:
    return _custom_spec(
        name=name,
        family=family,
        lookback=window,
        required_series=(source,),
        description=description,
        builder=lambda obs, source=source, window=window: rolling_mean(
            obs.series(source), window
        ),
    )


def _rolling_std_spec(
    *,
    name: str,
    family: str,
    source: str,
    window: int,
    description: str,
) -> MicrostructureFeatureSpec:
    return _custom_spec(
        name=name,
        family=family,
        lookback=window,
        required_series=(source,),
        description=description,
        builder=lambda obs, source=source, window=window: rolling_std(
            obs.series(source), window
        ),
    )


def _product_spec(
    *,
    name: str,
    family: str,
    left_builder,
    right_builder,
    lookback: int,
    required_series: tuple[str, ...],
    description: str,
) -> MicrostructureFeatureSpec:
    return _custom_spec(
        name=name,
        family=family,
        lookback=lookback,
        required_series=required_series,
        description=description,
        builder=lambda obs, left_builder=left_builder, right_builder=right_builder: left_builder(obs)
        * right_builder(obs),
    )


def _rolling_transformed_mean_spec(
    *,
    name: str,
    family: str,
    source: str,
    window: int,
    description: str,
    transform,
) -> MicrostructureFeatureSpec:
    return _custom_spec(
        name=name,
        family=family,
        lookback=window,
        required_series=(source,),
        description=description,
        builder=lambda obs, source=source, window=window, transform=transform: rolling_mean(
            transform(obs.series(source)), window
        ),
    )


def _rolling_transformed_std_spec(
    *,
    name: str,
    family: str,
    source: str,
    window: int,
    description: str,
    transform,
) -> MicrostructureFeatureSpec:
    return _custom_spec(
        name=name,
        family=family,
        lookback=window,
        required_series=(source,),
        description=description,
        builder=lambda obs, source=source, window=window, transform=transform: rolling_std(
            transform(obs.series(source)), window
        ),
    )


def _depth_total_log(obs: MicrostructureObservables) -> np.ndarray:
    return np.log(np.maximum(obs.q_bid + obs.q_ask, 1e-12))


def _depth_pressure(obs: MicrostructureObservables) -> np.ndarray:
    return np.log(np.maximum(obs.q_bid, 1e-12)) - np.log(np.maximum(obs.q_ask, 1e-12))


def _mid_vs_micro_ret_gap(obs: MicrostructureObservables) -> np.ndarray:
    return obs.micro_ret - obs.mid_ret


def _signed_direction(arr: np.ndarray) -> np.ndarray:
    return np.sign(np.asarray(arr, dtype=float))


def _canonical_specs() -> tuple[MicrostructureFeatureSpec, ...]:
    specs: list[MicrostructureFeatureSpec] = []
    for steps in (1, 2, 3, 5, 8):
        specs.append(
            _lag_spec(
                name=f"mid_ret_lag_{steps}",
                family="momentum",
                source="mid_ret",
                steps=steps,
                description=f"Lag {steps} of mid log return.",
            )
        )
    for steps in (1, 2, 3, 5):
        specs.append(
            _lag_spec(
                name=f"micro_ret_lag_{steps}",
                family="momentum",
                source="micro_ret",
                steps=steps,
                description=f"Lag {steps} of microprice log return.",
            )
        )
    for steps in (1, 2, 3, 5):
        specs.append(
            _lag_spec(
                name=f"microdev_lag_{steps}",
                family="microdev",
                source="microdev",
                steps=steps,
                description=f"Lag {steps} of microprice minus midprice.",
            )
        )
    for window in (10, 20):
        specs.append(
            _rolling_mean_spec(
                name=f"microdev_mean_{window}",
                family="microdev",
                source="microdev",
                window=window,
                description=f"Rolling mean of microprice deviation over {window} events.",
            )
        )
    for steps in (1, 2, 3, 5):
        specs.append(
            _lag_spec(
                name=f"spread_lag_{steps}",
                family="spread",
                source="spread_ticks",
                steps=steps,
                description=f"Lag {steps} of spread in ticks.",
            )
        )
    for steps in (1, 2, 3):
        specs.append(
            _lag_spec(
                name=f"rel_spread_lag_{steps}",
                family="spread",
                source="rel_spread",
                steps=steps,
                description=f"Lag {steps} of relative spread.",
            )
        )
    for window in (5, 10, 20):
        specs.append(
            _rolling_mean_spec(
                name=f"spread_change_rate_{window}",
                family="spread",
                source="spread_changed",
                window=window,
                description=f"Rolling spread-change rate over {window} events.",
            )
        )
    for steps in (1, 2, 3, 5, 8):
        specs.append(
            _lag_spec(
                name=f"imbalance_lag_{steps}",
                family="imbalance",
                source="imbalance",
                steps=steps,
                description=f"Lag {steps} of top-of-book imbalance.",
            )
        )
    for window in (5, 10, 20, 40):
        specs.append(
            _rolling_mean_spec(
                name=f"imbalance_mean_{window}",
                family="imbalance",
                source="imbalance",
                window=window,
                description=f"Rolling mean imbalance over {window} events.",
            )
        )
    for steps in (1, 2, 3, 5, 8):
        specs.append(
            _lag_spec(
                name=f"signed_trade_lag_{steps}",
                family="flow",
                source="signed_trade",
                steps=steps,
                description=f"Lag {steps} of signed trade direction.",
            )
        )
    for steps in (1, 2, 3, 5):
        specs.append(
            _lag_spec(
                name=f"signed_volume_lag_{steps}",
                family="flow",
                source="signed_volume",
                steps=steps,
                description=f"Lag {steps} of signed trade volume.",
            )
        )
    for window in (5, 10, 20, 40):
        specs.append(
            _rolling_mean_spec(
                name=f"signed_flow_mean_{window}",
                family="flow",
                source="signed_trade",
                window=window,
                description=f"Rolling mean signed trade flow over {window} events.",
            )
        )
    specs.append(
        _rolling_mean_spec(
            name="signed_flow_abs_mean_20",
            family="flow",
            source="signed_trade_abs",
            window=20,
            description="Rolling mean absolute signed trade flow over 20 events.",
        )
    )
    for window in (5, 10, 20, 40):
        specs.append(
            _rolling_mean_spec(
                name=f"ofi_mean_{window}",
                family="flow",
                source="ofi",
                window=window,
                description=f"Rolling mean order-flow imbalance over {window} events.",
            )
        )
    for window in (5, 10, 20, 40):
        specs.append(
            _rolling_std_spec(
                name=f"mid_ret_vol_{window}",
                family="volatility",
                source="mid_ret",
                window=window,
                description=f"Rolling volatility of mid log return over {window} events.",
            )
        )
    for window in (5, 10, 20):
        specs.append(
            _rolling_mean_spec(
                name=f"price_change_rate_{window}",
                family="volatility",
                source="price_changed",
                window=window,
                description=f"Rolling price-change rate over {window} events.",
            )
        )
    for window in (5, 10, 20):
        specs.append(
            _rolling_mean_spec(
                name=f"queue_depletion_diff_{window}",
                family="volatility",
                source="depletion_dir",
                window=window,
                description=f"Rolling signed queue-depletion balance over {window} events.",
            )
        )
    specs.append(
        _product_spec(
            name="imbalance_x_spread_1",
            family="interaction",
            left_builder=lambda obs: lag(obs.imbalance, 1),
            right_builder=lambda obs: lag(obs.spread_ticks, 1),
            lookback=1,
            required_series=("imbalance", "spread_ticks"),
            description="Lagged imbalance multiplied by lagged spread ticks.",
        )
    )
    specs.append(
        _product_spec(
            name="imbalance_x_signedflow_5",
            family="interaction",
            left_builder=lambda obs: lag(obs.imbalance, 1),
            right_builder=lambda obs: rolling_mean(obs.signed_trade, 5),
            lookback=5,
            required_series=("imbalance", "signed_trade"),
            description="Lagged imbalance multiplied by 5-event signed flow mean.",
        )
    )
    return tuple(specs)


def _expanded_specs_tier_2() -> tuple[MicrostructureFeatureSpec, ...]:
    specs: list[MicrostructureFeatureSpec] = []
    for steps in (13, 21, 34):
        specs.append(
            _lag_spec(
                name=f"mid_ret_lag_{steps}",
                family="momentum",
                source="mid_ret",
                steps=steps,
                description=f"Lag {steps} of mid log return.",
            )
        )
    for steps in (8, 13, 21):
        specs.append(
            _lag_spec(
                name=f"micro_ret_lag_{steps}",
                family="momentum",
                source="micro_ret",
                steps=steps,
                description=f"Lag {steps} of microprice log return.",
            )
        )
    for window in (5, 10, 20, 40):
        specs.append(
            _rolling_mean_spec(
                name=f"mid_ret_mean_{window}",
                family="momentum",
                source="mid_ret",
                window=window,
                description=f"Rolling mean mid return over {window} events.",
            )
        )
    for window in (5, 10, 20, 40):
        specs.append(
            _rolling_mean_spec(
                name=f"micro_ret_mean_{window}",
                family="momentum",
                source="micro_ret",
                window=window,
                description=f"Rolling mean microprice return over {window} events.",
            )
        )
    for steps in (8, 13, 21):
        specs.append(
            _lag_spec(
                name=f"microdev_lag_{steps}",
                family="microdev",
                source="microdev",
                steps=steps,
                description=f"Lag {steps} of microprice minus midprice.",
            )
        )
    for window in (40, 80):
        specs.append(
            _rolling_mean_spec(
                name=f"microdev_mean_{window}",
                family="microdev",
                source="microdev",
                window=window,
                description=f"Rolling mean of microprice deviation over {window} events.",
            )
        )
    for window in (10, 20, 40):
        specs.append(
            _rolling_std_spec(
                name=f"microdev_vol_{window}",
                family="microdev",
                source="microdev",
                window=window,
                description=f"Rolling volatility of microprice deviation over {window} events.",
            )
        )
    for steps in (8, 13):
        specs.append(
            _lag_spec(
                name=f"spread_lag_{steps}",
                family="spread",
                source="spread_ticks",
                steps=steps,
                description=f"Lag {steps} of spread in ticks.",
            )
        )
    for steps in (5, 8):
        specs.append(
            _lag_spec(
                name=f"rel_spread_lag_{steps}",
                family="spread",
                source="rel_spread",
                steps=steps,
                description=f"Lag {steps} of relative spread.",
            )
        )
    for window in (40, 80):
        specs.append(
            _rolling_mean_spec(
                name=f"spread_change_rate_{window}",
                family="spread",
                source="spread_changed",
                window=window,
                description=f"Rolling spread-change rate over {window} events.",
            )
        )
    for window in (10, 20, 40):
        specs.append(
            _rolling_std_spec(
                name=f"spread_vol_{window}",
                family="spread",
                source="spread_ticks",
                window=window,
                description=f"Rolling spread volatility over {window} events.",
            )
        )
    for steps in (13, 21):
        specs.append(
            _lag_spec(
                name=f"imbalance_lag_{steps}",
                family="imbalance",
                source="imbalance",
                steps=steps,
                description=f"Lag {steps} of top-of-book imbalance.",
            )
        )
    specs.append(
        _rolling_mean_spec(
            name="imbalance_mean_80",
            family="imbalance",
            source="imbalance",
            window=80,
            description="Rolling mean imbalance over 80 events.",
        )
    )
    for window in (5, 10, 20, 40):
        specs.append(
            _rolling_std_spec(
                name=f"imbalance_vol_{window}",
                family="imbalance",
                source="imbalance",
                window=window,
                description=f"Rolling imbalance volatility over {window} events.",
            )
        )
    for steps in (13, 21):
        specs.append(
            _lag_spec(
                name=f"signed_trade_lag_{steps}",
                family="flow",
                source="signed_trade",
                steps=steps,
                description=f"Lag {steps} of signed trade direction.",
            )
        )
    for steps in (8, 13):
        specs.append(
            _lag_spec(
                name=f"signed_volume_lag_{steps}",
                family="flow",
                source="signed_volume",
                steps=steps,
                description=f"Lag {steps} of signed trade volume.",
            )
        )
    specs.append(
        _rolling_mean_spec(
            name="signed_flow_mean_80",
            family="flow",
            source="signed_trade",
            window=80,
            description="Rolling mean signed trade flow over 80 events.",
        )
    )
    for window in (5, 10, 20, 40, 80):
        specs.append(
            _rolling_mean_spec(
                name=f"trade_rate_{window}",
                family="intensity",
                source="signed_trade_abs",
                window=window,
                description=f"Rolling trade-event rate over {window} events.",
            )
        )
    for window in (10, 20, 40, 80):
        specs.append(
            _rolling_transformed_mean_spec(
                name=f"signed_volume_abs_mean_{window}",
                family="flow",
                source="signed_volume",
                window=window,
                description=f"Rolling mean absolute signed volume over {window} events.",
                transform=np.abs,
            )
        )
    specs.append(
        _rolling_mean_spec(
            name="ofi_mean_80",
            family="flow",
            source="ofi",
            window=80,
            description="Rolling mean order-flow imbalance over 80 events.",
        )
    )
    for window in (5, 10, 20, 40):
        specs.append(
            _rolling_std_spec(
                name=f"ofi_vol_{window}",
                family="flow",
                source="ofi",
                window=window,
                description=f"Rolling OFI volatility over {window} events.",
            )
        )
    specs.append(
        _rolling_std_spec(
            name="mid_ret_vol_80",
            family="volatility",
            source="mid_ret",
            window=80,
            description="Rolling volatility of mid log return over 80 events.",
        )
    )
    for window in (40, 80):
        specs.append(
            _rolling_mean_spec(
                name=f"price_change_rate_{window}",
                family="volatility",
                source="price_changed",
                window=window,
                description=f"Rolling price-change rate over {window} events.",
            )
        )
    for window in (40, 80):
        specs.append(
            _rolling_mean_spec(
                name=f"queue_depletion_diff_{window}",
                family="volatility",
                source="depletion_dir",
                window=window,
                description=f"Rolling signed queue-depletion balance over {window} events.",
            )
        )
    for steps in (1, 3):
        specs.append(
            _custom_spec(
                name=f"depth_total_log_lag_{steps}",
                family="depth",
                lookback=steps,
                required_series=("q_bid", "q_ask"),
                description=f"Lag {steps} of total top-of-book depth in log scale.",
                builder=lambda obs, steps=steps: lag(_depth_total_log(obs), steps),
            )
        )
    return tuple(specs)


def _expanded_specs_tier_3() -> tuple[MicrostructureFeatureSpec, ...]:
    specs: list[MicrostructureFeatureSpec] = []
    for steps in (5, 8, 13, 21):
        specs.append(
            _custom_spec(
                name=f"depth_total_log_lag_{steps}",
                family="depth",
                lookback=steps,
                required_series=("q_bid", "q_ask"),
                description=f"Lag {steps} of total top-of-book depth in log scale.",
                builder=lambda obs, steps=steps: lag(_depth_total_log(obs), steps),
            )
        )
    for window in (5, 10, 20, 40, 80):
        specs.append(
            _custom_spec(
                name=f"depth_total_log_mean_{window}",
                family="depth",
                lookback=window,
                required_series=("q_bid", "q_ask"),
                description=f"Rolling mean log total depth over {window} events.",
                builder=lambda obs, window=window: rolling_mean(_depth_total_log(obs), window),
            )
        )
    for steps in (1, 2, 3, 5, 8, 13):
        specs.append(
            _custom_spec(
                name=f"depth_pressure_lag_{steps}",
                family="depth",
                lookback=steps,
                required_series=("q_bid", "q_ask"),
                description=f"Lag {steps} of bid-vs-ask depth pressure in log scale.",
                builder=lambda obs, steps=steps: lag(_depth_pressure(obs), steps),
            )
        )
    for window in (5, 10, 20, 40, 80):
        specs.append(
            _custom_spec(
                name=f"depth_pressure_mean_{window}",
                family="depth",
                lookback=window,
                required_series=("q_bid", "q_ask"),
                description=f"Rolling mean depth pressure over {window} events.",
                builder=lambda obs, window=window: rolling_mean(_depth_pressure(obs), window),
            )
        )
    for window in (10, 20, 40):
        specs.append(
            _custom_spec(
                name=f"depth_pressure_vol_{window}",
                family="depth",
                lookback=window,
                required_series=("q_bid", "q_ask"),
                description=f"Rolling depth-pressure volatility over {window} events.",
                builder=lambda obs, window=window: rolling_std(_depth_pressure(obs), window),
            )
        )
    for window in (10, 20, 40, 80):
        specs.append(
            _rolling_transformed_mean_spec(
                name=f"microdev_abs_mean_{window}",
                family="microdev",
                source="microdev",
                window=window,
                description=f"Rolling mean absolute microprice deviation over {window} events.",
                transform=np.abs,
            )
        )
    for window in (10, 20, 40):
        specs.append(
            _rolling_transformed_mean_spec(
                name=f"microdev_direction_rate_{window}",
                family="microdev",
                source="microdev",
                window=window,
                description=f"Rolling signed microprice-deviation direction over {window} events.",
                transform=_signed_direction,
            )
        )
    for window in (10, 20, 40, 80):
        specs.append(
            _rolling_std_spec(
                name=f"signed_volume_vol_{window}",
                family="flow",
                source="signed_volume",
                window=window,
                description=f"Rolling signed-volume volatility over {window} events.",
            )
        )
    for window in (10, 20, 40, 80):
        specs.append(
            _rolling_transformed_mean_spec(
                name=f"ofi_abs_mean_{window}",
                family="flow",
                source="ofi",
                window=window,
                description=f"Rolling mean absolute OFI over {window} events.",
                transform=np.abs,
            )
        )
    for window in (10, 20, 40):
        specs.append(
            _rolling_transformed_mean_spec(
                name=f"ofi_direction_rate_{window}",
                family="flow",
                source="ofi",
                window=window,
                description=f"Rolling OFI direction rate over {window} events.",
                transform=_signed_direction,
            )
        )
    specs.extend(
        (
            _product_spec(
                name="microdev_x_imbalance_1",
                family="interaction",
                left_builder=lambda obs: lag(obs.microdev, 1),
                right_builder=lambda obs: lag(obs.imbalance, 1),
                lookback=1,
                required_series=("microdev", "imbalance"),
                description="Lagged microprice deviation multiplied by lagged imbalance.",
            ),
            _product_spec(
                name="microdev_x_flow_10",
                family="interaction",
                left_builder=lambda obs: lag(obs.microdev, 1),
                right_builder=lambda obs: rolling_mean(obs.signed_trade, 10),
                lookback=10,
                required_series=("microdev", "signed_trade"),
                description="Lagged microprice deviation multiplied by 10-event signed flow mean.",
            ),
            _product_spec(
                name="ofi_x_spread_10",
                family="interaction",
                left_builder=lambda obs: rolling_mean(obs.ofi, 10),
                right_builder=lambda obs: rolling_mean(obs.spread_ticks, 10),
                lookback=10,
                required_series=("ofi", "spread_ticks"),
                description="10-event OFI mean multiplied by 10-event spread mean.",
            ),
            _product_spec(
                name="ofi_x_imbalance_10",
                family="interaction",
                left_builder=lambda obs: rolling_mean(obs.ofi, 10),
                right_builder=lambda obs: rolling_mean(obs.imbalance, 10),
                lookback=10,
                required_series=("ofi", "imbalance"),
                description="10-event OFI mean multiplied by 10-event imbalance mean.",
            ),
            _product_spec(
                name="spread_x_trade_rate_20",
                family="interaction",
                left_builder=lambda obs: rolling_mean(obs.spread_ticks, 20),
                right_builder=lambda obs: rolling_mean(obs.series("signed_trade_abs"), 20),
                lookback=20,
                required_series=("spread_ticks", "signed_trade_abs"),
                description="20-event spread mean multiplied by 20-event trade rate.",
            ),
            _product_spec(
                name="vol_x_flow_20",
                family="interaction",
                left_builder=lambda obs: rolling_std(obs.mid_ret, 20),
                right_builder=lambda obs: rolling_mean(obs.signed_trade, 20),
                lookback=20,
                required_series=("mid_ret", "signed_trade"),
                description="20-event mid-return volatility multiplied by 20-event signed flow mean.",
            ),
            _product_spec(
                name="depth_pressure_x_flow_20",
                family="interaction",
                left_builder=lambda obs: rolling_mean(_depth_pressure(obs), 20),
                right_builder=lambda obs: rolling_mean(obs.signed_trade, 20),
                lookback=20,
                required_series=("q_bid", "q_ask", "signed_trade"),
                description="20-event depth pressure multiplied by 20-event signed flow mean.",
            ),
            _product_spec(
                name="depth_pressure_x_microdev_10",
                family="interaction",
                left_builder=lambda obs: rolling_mean(_depth_pressure(obs), 10),
                right_builder=lambda obs: rolling_mean(obs.microdev, 10),
                lookback=10,
                required_series=("q_bid", "q_ask", "microdev"),
                description="10-event depth pressure multiplied by 10-event microprice deviation mean.",
            ),
        )
    )
    for steps in (1, 2, 3, 5, 8):
        specs.append(
            _custom_spec(
                name=f"mid_vs_micro_ret_gap_lag_{steps}",
                family="momentum",
                lookback=steps,
                required_series=("mid_ret", "micro_ret"),
                description=f"Lag {steps} of microprice-vs-mid return gap.",
                builder=lambda obs, steps=steps: lag(_mid_vs_micro_ret_gap(obs), steps),
            )
        )
    for window in (20, 40, 80):
        specs.append(
            _rolling_std_spec(
                name=f"price_change_burst_{window}",
                family="intensity",
                source="price_changed",
                window=window,
                description=f"Rolling burstiness of price-change arrivals over {window} events.",
            )
        )
    for window in (20, 40, 80):
        specs.append(
            _rolling_std_spec(
                name=f"spread_change_burst_{window}",
                family="intensity",
                source="spread_changed",
                window=window,
                description=f"Rolling burstiness of spread-change arrivals over {window} events.",
            )
        )
    for window in (10, 20, 40, 80):
        specs.append(
            _rolling_transformed_mean_spec(
                name=f"depletion_rate_{window}",
                family="depth",
                source="depletion_dir",
                window=window,
                description=f"Rolling queue-depletion event rate over {window} events.",
                transform=np.abs,
            )
        )
    return tuple(specs)


def _all_ordered_specs() -> tuple[MicrostructureFeatureSpec, ...]:
    return _canonical_specs() + _expanded_specs_tier_2() + _expanded_specs_tier_3()


@lru_cache(maxsize=None)
def microstructure_feature_registry(
    feature_count: int = 64,
) -> MicrostructureFeatureRegistry:
    if feature_count not in _SUPPORTED_FEATURE_COUNTS:
        supported = ", ".join(str(value) for value in _SUPPORTED_FEATURE_COUNTS)
        raise ValueError(
            f"Unsupported microstructure feature count: {feature_count}. "
            f"Supported values: {supported}."
        )
    specs = _all_ordered_specs()[:feature_count]
    return MicrostructureFeatureRegistry(specs=specs)


@lru_cache(maxsize=1)
def default_microstructure_feature_registry() -> MicrostructureFeatureRegistry:
    return microstructure_feature_registry(_SUPPORTED_FEATURE_COUNTS[0])


CANONICAL_MICROSTRUCTURE_FEATURE_NAMES = (
    default_microstructure_feature_registry().feature_names()
)


def build_microstructure_feature_table(
    observables: MicrostructureObservables,
    *,
    registry: MicrostructureFeatureRegistry | None = None,
    names: tuple[str, ...] | None = None,
    families: tuple[str, ...] | None = None,
) -> MicrostructureFeatureTable:
    active_registry = default_microstructure_feature_registry() if registry is None else registry
    if names is not None or families is not None:
        active_registry = active_registry.select(names=names, families=families)
    return active_registry.build(observables)


__all__ = [
    "CANONICAL_MICROSTRUCTURE_FEATURE_NAMES",
    "MicrostructureFeatureRegistry",
    "build_microstructure_feature_table",
    "default_microstructure_feature_registry",
    "microstructure_feature_registry",
    "supported_microstructure_feature_counts",
]
