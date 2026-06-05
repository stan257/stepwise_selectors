"""L1 microstructure simulator for the benchmark pipeline."""

from __future__ import annotations

import numpy as np

from .types import MicrostructureChunkConfig, MicrostructureObservables
from .utils import (
    build_event_regimes,
    sigmoid,
    student_unit_variance,
    validate_microstructure_config,
)


class L1MicrostructureSimulator:
    """Simulate an event-time L1 order book and derived observable series."""

    def simulate(
        self,
        config: MicrostructureChunkConfig,
        *,
        rng: np.random.Generator,
        regime_by_event: np.ndarray,
        n_events_total: int,
    ) -> MicrostructureObservables:
        tick = config.tick_size
        base_spread = max(1.0, float(round(2.0 * config.base_half_spread_ticks)))
        depth = config.base_queue_depth

        mid_price = np.empty(n_events_total, dtype=float)
        spread_ticks = np.empty(n_events_total, dtype=float)
        q_bid = np.empty(n_events_total, dtype=float)
        q_ask = np.empty(n_events_total, dtype=float)
        signed_trade = np.zeros(n_events_total, dtype=float)
        signed_volume = np.zeros(n_events_total, dtype=float)
        price_changed = np.zeros(n_events_total, dtype=float)
        spread_changed = np.zeros(n_events_total, dtype=float)
        depletion_dir = np.zeros(n_events_total, dtype=float)

        mid_price[0] = 100.0
        spread_ticks[0] = base_spread
        q_bid[0] = depth * (1.0 + 0.05 * rng.standard_normal())
        q_ask[0] = depth * (1.0 + 0.05 * rng.standard_normal())

        regime_activity_level = rng.normal(loc=0.0, scale=0.25, size=config.n_regimes)
        regime_trade_bias = rng.normal(loc=0.0, scale=0.20, size=config.n_regimes)
        regime_spread_shift = np.clip(
            rng.normal(loc=0.15, scale=0.25, size=config.n_regimes), -0.20, 0.90
        )
        regime_depth_shift = rng.normal(loc=0.0, scale=0.12, size=config.n_regimes)
        regime_drift = rng.normal(loc=0.0, scale=0.18, size=config.n_regimes)
        regime_impact = np.clip(
            1.0 + rng.normal(loc=0.0, scale=0.15, size=config.n_regimes), 0.7, 1.4
        )

        activity_state = 1.0
        fair_offset_ticks = 0.0
        flow_ema = 0.0
        abs_flow_ema = 0.0
        vol_ema = 0.0
        last_trade_sign = 0.0

        for t in range(1, n_events_total):
            prev_mid = float(mid_price[t - 1])
            prev_spread = float(spread_ticks[t - 1])
            prev_q_bid = float(q_bid[t - 1])
            prev_q_ask = float(q_ask[t - 1])
            reg = int(regime_by_event[t])

            activity_state = (
                config.activity_persistence * activity_state
                + (1.0 - config.activity_persistence) * (1.0 + regime_activity_level[reg])
                + 0.08 * abs(student_unit_variance(config.student_df, rng))
            )
            activity_state = max(activity_state, 0.15)
            fair_offset_ticks = (
                0.985 * fair_offset_ticks
                + 0.10 * regime_drift[reg]
                + 0.25 * config.impact_strength * regime_impact[reg] * flow_ema
                + 0.08 * student_unit_variance(config.student_df, rng)
            )
            fair_offset_ticks = float(np.clip(fair_offset_ticks, -4.0, 4.0))

            imbalance_prev = (prev_q_bid - prev_q_ask) / max(prev_q_bid + prev_q_ask, 1e-12)
            bid_prev = prev_mid - 0.5 * prev_spread * tick
            ask_prev = prev_mid + 0.5 * prev_spread * tick
            micro_prev = (ask_prev * prev_q_bid + bid_prev * prev_q_ask) / max(
                prev_q_bid + prev_q_ask, 1e-12
            )
            microdev_prev_ticks = (micro_prev - prev_mid) / tick

            trade_score = (
                2.0 * config.sign_persistence * last_trade_sign
                + 1.9 * imbalance_prev
                + 1.4 * np.tanh(microdev_prev_ticks)
                + 0.7 * fair_offset_ticks
                + 0.45 * flow_ema
                + regime_trade_bias[reg]
            )
            p_buy_given_trade = sigmoid(trade_score)

            market_scale = 0.28 + 0.12 * activity_state + 0.08 * abs(imbalance_prev)
            add_scale = 0.44 + config.replenish_rate_scale * (0.90 - 0.25 * activity_state)
            cancel_scale = 0.20 + config.cancel_rate_scale * (0.55 + 0.25 * activity_state)
            p_bid_given_add = sigmoid(-1.25 * imbalance_prev - 0.30 * flow_ema)
            p_bid_given_cancel = sigmoid(1.10 * imbalance_prev + 0.30 * flow_ema)

            weights = np.array(
                [
                    market_scale * p_buy_given_trade,
                    market_scale * (1.0 - p_buy_given_trade),
                    add_scale * p_bid_given_add,
                    add_scale * (1.0 - p_bid_given_add),
                    cancel_scale * p_bid_given_cancel,
                    cancel_scale * (1.0 - p_bid_given_cancel),
                ],
                dtype=float,
            )
            weights = np.maximum(weights, 1e-12)
            weights /= np.sum(weights)
            event_type = int(rng.choice(6, p=weights))

            scale_noise = abs(student_unit_variance(config.student_df, rng))
            trade_size = depth * (0.055 + 0.018 * activity_state + 0.020 * scale_noise)
            add_size = depth * (
                0.040 + 0.018 * config.replenish_rate_scale + 0.015 * scale_noise
            )
            cancel_size = depth * (
                0.030 + 0.020 * config.cancel_rate_scale + 0.012 * scale_noise
            )

            new_mid = prev_mid
            new_spread = prev_spread
            new_q_bid = prev_q_bid
            new_q_ask = prev_q_ask

            if event_type == 0:
                new_q_ask -= trade_size
                signed_trade[t] = 1.0
                signed_volume[t] = trade_size
            elif event_type == 1:
                new_q_bid -= trade_size
                signed_trade[t] = -1.0
                signed_volume[t] = -trade_size
            elif event_type == 2:
                new_q_bid += add_size
            elif event_type == 3:
                new_q_ask += add_size
            elif event_type == 4:
                new_q_bid -= cancel_size
            else:
                new_q_ask -= cancel_size

            replenish_base = depth * (1.0 + 0.12 * activity_state + regime_depth_shift[reg])
            replenish_base = max(replenish_base, 0.4 * depth)
            if new_q_ask <= 0.0:
                new_mid = prev_mid + tick
                depletion_dir[t] = 1.0
                new_q_ask = replenish_base * (1.0 + 0.10 * rng.random())
                new_q_bid = max(new_q_bid + 0.18 * replenish_base, 0.30 * depth)
            elif new_q_bid <= 0.0:
                new_mid = prev_mid - tick
                depletion_dir[t] = -1.0
                new_q_bid = replenish_base * (1.0 + 0.10 * rng.random())
                new_q_ask = max(new_q_ask + 0.18 * replenish_base, 0.30 * depth)

            desired_spread = (
                base_spread
                + regime_spread_shift[reg]
                + config.spread_vol_sensitivity * (0.55 * activity_state + 1.60 * vol_ema)
            )
            desired_spread = max(1.0, float(round(desired_spread)))
            if desired_spread > prev_spread and rng.random() < 0.20 + 0.08 * activity_state:
                new_spread = prev_spread + 1.0
            elif desired_spread < prev_spread and rng.random() < 0.16:
                new_spread = prev_spread - 1.0
            if depletion_dir[t] != 0.0 and rng.random() < 0.10 + 0.04 * activity_state:
                new_spread = max(1.0, new_spread + 1.0)
            new_spread = max(1.0, new_spread)

            target_bid_depth = depth * (1.0 + 0.08 * activity_state - 0.15 * imbalance_prev)
            target_ask_depth = depth * (1.0 + 0.08 * activity_state + 0.15 * imbalance_prev)
            new_q_bid = (
                config.queue_mean_reversion * new_q_bid
                + (1.0 - config.queue_mean_reversion) * max(target_bid_depth, 0.25 * depth)
            )
            new_q_ask = (
                config.queue_mean_reversion * new_q_ask
                + (1.0 - config.queue_mean_reversion) * max(target_ask_depth, 0.25 * depth)
            )
            new_q_bid = max(new_q_bid, 0.15 * depth)
            new_q_ask = max(new_q_ask, 0.15 * depth)

            mid_price[t] = max(new_mid, 10.0 * tick)
            spread_ticks[t] = new_spread
            q_bid[t] = new_q_bid
            q_ask[t] = new_q_ask
            price_changed[t] = 1.0 if mid_price[t] != prev_mid else 0.0
            spread_changed[t] = 1.0 if spread_ticks[t] != prev_spread else 0.0

            last_trade_sign = (
                float(np.sign(signed_trade[t]))
                if signed_trade[t] != 0.0
                else last_trade_sign
            )
            flow_ema = 0.92 * flow_ema + 0.08 * float(np.sign(signed_volume[t]))
            abs_flow_ema = 0.95 * abs_flow_ema + 0.05 * abs(float(signed_volume[t]) / depth)
            price_move_ticks = abs(mid_price[t] - prev_mid) / tick
            vol_ema = 0.94 * vol_ema + 0.06 * (price_move_ticks + 0.35 * abs_flow_ema)

        bid = mid_price - 0.5 * spread_ticks * tick
        ask = mid_price + 0.5 * spread_ticks * tick
        bid_levels = np.rint(bid / tick).astype(np.int64)
        ask_levels = np.rint(ask / tick).astype(np.int64)
        ofi = np.zeros(n_events_total, dtype=float)
        bid_move = np.where(
            bid_levels[1:] > bid_levels[:-1],
            q_bid[1:],
            np.where(
                bid_levels[1:] == bid_levels[:-1],
                q_bid[1:] - q_bid[:-1],
                -q_bid[:-1],
            ),
        )
        ask_move = np.where(
            ask_levels[1:] < ask_levels[:-1],
            -q_ask[1:],
            np.where(
                ask_levels[1:] == ask_levels[:-1],
                q_ask[:-1] - q_ask[1:],
                q_ask[:-1],
            ),
        )
        ofi[1:] = bid_move + ask_move

        microprice = (ask * q_bid + bid * q_ask) / np.maximum(q_bid + q_ask, 1e-12)
        microdev = microprice - mid_price
        imbalance = (q_bid - q_ask) / np.maximum(q_bid + q_ask, 1e-12)
        rel_spread = (spread_ticks * tick) / np.maximum(mid_price, 1e-12)
        mid_log = np.log(np.maximum(mid_price, 1e-12))
        micro_log = np.log(np.maximum(microprice, 1e-12))
        mid_ret = np.diff(mid_log, prepend=mid_log[0])
        micro_ret = np.diff(micro_log, prepend=micro_log[0])

        return MicrostructureObservables(
            event_index=np.arange(n_events_total, dtype=int),
            regime_by_event=np.asarray(regime_by_event, dtype=int),
            mid_price=mid_price,
            spread_ticks=spread_ticks,
            q_bid=q_bid,
            q_ask=q_ask,
            signed_trade=signed_trade,
            signed_volume=signed_volume,
            price_changed=price_changed,
            spread_changed=spread_changed,
            depletion_dir=depletion_dir,
            ofi=ofi,
            bid=bid,
            ask=ask,
            microprice=microprice,
            microdev=microdev,
            imbalance=imbalance,
            rel_spread=rel_spread,
            mid_log=mid_log,
            micro_log=micro_log,
            mid_ret=mid_ret,
            micro_ret=micro_ret,
            extras={"signed_trade_abs": np.abs(signed_trade)},
        )



def simulate_microstructure_observables(
    config: MicrostructureChunkConfig,
) -> MicrostructureObservables:
    validated = validate_microstructure_config(config)
    _, regime_by_event, _, _, total_events = build_event_regimes(validated)
    simulator = L1MicrostructureSimulator()
    return simulator.simulate(
        validated,
        rng=np.random.default_rng(validated.seed),
        regime_by_event=regime_by_event,
        n_events_total=total_events,
    )


__all__ = ["L1MicrostructureSimulator", "simulate_microstructure_observables"]
