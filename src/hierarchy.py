"""Hierarchical empirical-Bayes effects (partial pooling).

Levels, from coarse to fine:
    global -> region pair -> pickup city, delivery city -> lane -> lane+equipment

Each level estimates how much its groups *deviate from everything above it*:

    effect[group] = sum(partial residual in group) / (n_group + k_level)

The pseudo-count k shrinks small groups toward 0, i.e. toward the parent level's
prediction (not toward the global mean). A lane with 3 loads mostly inherits its
cities' and region's level; a lane with 300 loads mostly speaks for itself.
Levels are fitted by backfitting: each level is re-estimated on the residual
left by all other levels, for a few passes (like a mixed model with fixed
variance ratios, but in a few lines of numpy/pandas).

Unseen cities: their city deviation is borrowed from nearby known cities (geo.py);
an unseen lane has deviation 0, so it falls back to region + city effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .geo import borrow

LEVELS = ["region_pair", "pickup", "delivery", "lane", "lane_equipment"]


def fit_effects(df: pd.DataFrame, resid, passes: int = None) -> dict:
    """Return {level: Series(group -> shrunken deviation)}."""
    passes = passes or config.HIER_PASSES
    resid = np.asarray(resid, dtype=float)
    keys = {lvl: df[lvl].reset_index(drop=True) for lvl in LEVELS}
    contrib = {lvl: np.zeros(len(df)) for lvl in LEVELS}
    effects = {}
    for _ in range(passes):
        for lvl in LEVELS:
            others = sum(contrib[o] for o in LEVELS if o != lvl)
            g = pd.Series(resid - others).groupby(keys[lvl]).agg(["sum", "count"])
            effects[lvl] = g["sum"] / (g["count"] + config.HIER_SMOOTHING[lvl])
            contrib[lvl] = keys[lvl].map(effects[lvl]).values
    return effects


def encode(df: pd.DataFrame, effects: dict, pickup_unseen, delivery_unseen, neighbours: dict) -> pd.DataFrame:
    """Hierarchical encodings for any frame. *_unseen are boolean arrays: True
    for cities without history (or masked in training)."""
    pickup_unseen = np.asarray(pickup_unseen, bool)
    delivery_unseen = np.asarray(delivery_unseen, bool)
    region = df["region_pair"].map(effects["region_pair"]).fillna(0.0).values
    pick = np.where(pickup_unseen, borrow(effects["pickup"], df["pickup"], neighbours).values,
                    df["pickup"].map(effects["pickup"]).fillna(0.0).values)
    drop = np.where(delivery_unseen, borrow(effects["delivery"], df["delivery"], neighbours).values,
                    df["delivery"].map(effects["delivery"]).fillna(0.0).values)
    lane_known = ~(pickup_unseen | delivery_unseen)
    lane = np.where(lane_known, df["lane"].map(effects["lane"]).fillna(0.0).values, 0.0)
    lane_eq = np.where(lane_known, df["lane_equipment"].map(effects["lane_equipment"]).fillna(0.0).values, 0.0)
    return pd.DataFrame({
        "hte_region": region,                                  # region-pair level
        "hte_pickup": pick,                                    # city deviations
        "hte_delivery": drop,
        "hte_lane": region + pick + drop + lane,               # cumulative through lane
        "hte_total": region + pick + drop + lane + lane_eq,    # full hierarchical estimate
    }, index=df.index)
