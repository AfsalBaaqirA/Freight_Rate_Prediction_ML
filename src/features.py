"""Feature engineering.

Target: log(rate per mile). Rate per mile removes most of the distance effect,
and the log turns multiplicative effects (equipment premium, market level) into
additive ones.

City / lane information enters through *smoothed, out-of-fold target encodings*
of the residual of a tiny baseline (log rpm ~ log distance + equipment).
Encoding the residual (not the raw target) stops a city looking "expensive"
just because its loads are short. Unseen cities (8 in validation) borrow the
inverse-distance-weighted encoding of the 3 nearest known cities (geo.py), and
an indicator tells the model it was borrowed.

Time-level features (only used by the variants in the feature study):
level_30d / level_60d = mean baseline residual over the trailing 30 / 60 days
before each date; dates after the training period get the last known level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import config
from .data import EQUIPMENT_CODES
from .geo import borrow, nearest_known

DAY0 = pd.Timestamp("2025-01-01")


def target(df: pd.DataFrame) -> pd.Series:
    return np.log(df["posted_rate"] / df["distance"])


def region(lon: pd.Series) -> pd.Series:
    """Rough west / mountain / central / east bands (coordinates are noisy,
    so only coarse location is trusted)."""
    return pd.cut(lon, [-180, -110, -97, -85, 0], labels=False).astype(int)


def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features that need no fitted state."""
    out = df.copy()
    out["log_distance"] = np.log(out["distance"])
    out["equipment_code"] = out["equipment"].map(EQUIPMENT_CODES).astype(int)
    out["dayofweek"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["dayofyear"] = out["date"].dt.dayofyear
    out["t_days"] = (out["date"] - DAY0).dt.days
    out["d_lat"] = out["delivery_lat"] - out["pickup_lat"]
    out["d_lon"] = out["delivery_lon"] - out["pickup_lon"]
    out["pickup_region"] = region(out["pickup_lon"])
    out["delivery_region"] = region(out["delivery_lon"])
    out["lane"] = out["pickup"] + ">" + out["delivery"]
    out["lane_equipment"] = out["lane"] + "|" + out["equipment"]
    out["region_pair"] = out["pickup_region"].astype(str) + "-" + out["delivery_region"].astype(str)
    return out


def _base_design(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        np.ones(len(df)), np.log(df["distance"]),
        (df["equipment"] == "Reefer").astype(float), (df["equipment"] == "Flatbed").astype(float),
    ])


def _smoothed_means(keys: pd.Series, values: pd.Series) -> pd.Series:
    """Mean residual per group, shrunk toward 0 (the prior) by TE_SMOOTHING."""
    g = values.groupby(keys).agg(["sum", "count"])
    return g["sum"] / (g["count"] + config.TE_SMOOTHING)


TE_KEYS = {"te_pickup": "pickup", "te_delivery": "delivery",
           "te_lane": "lane", "te_lane_equipment": "lane_equipment"}


class FeatureBuilder:
    """Fit on a labelled training frame, then transform any frame.

    * transform_train(): out-of-fold encodings for the training rows (a row's
      encoding never uses its own target -> no leakage).
    * transform(): encodings computed from all training rows, for new data.
    """

    def __init__(self, cities: pd.DataFrame):
        self.cities = cities  # full city -> (lat, lon) lookup incl. unseen cities

    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        train = basic_features(train)
        coef, *_ = np.linalg.lstsq(_base_design(train), target(train).values, rcond=None)
        res = pd.Series(target(train).values - _base_design(train) @ coef, index=train.index)
        self.known_cities = set(train["pickup"]) | set(train["delivery"])
        self.neighbours = nearest_known(self.cities, self.known_cities)
        self.tables = self._tables(train, res)
        self.levels = self._levels(train, res)
        self._train, self._res = train, res
        return self

    @staticmethod
    def _tables(train, res) -> dict:
        return {name: _smoothed_means(train[key], res) for name, key in TE_KEYS.items()}

    @staticmethod
    def _levels(train, res) -> dict:
        """Trailing W-day mean residual before each date; after the last training
        day the level stays at the last known value (no look-ahead)."""
        days = pd.date_range(train["date"].min(), train["date"].max())
        s = res.groupby(train["date"]).sum().reindex(days, fill_value=0.0)
        n = res.groupby(train["date"]).size().reindex(days, fill_value=0)
        out = {}
        for w in (30, 60):
            level = (s.rolling(w, min_periods=1).sum() / n.rolling(w, min_periods=1).sum()).shift(1)
            out[w] = (level, float(s.iloc[-w:].sum() / n.iloc[-w:].sum()), days[-1])
        return out

    def _encode(self, df, tables) -> pd.DataFrame:
        out = df.copy()
        pick_unseen = ~out["pickup"].isin(self.known_cities).values
        drop_unseen = ~out["delivery"].isin(self.known_cities).values
        out["pickup_seen"] = (~pick_unseen).astype(int)
        out["delivery_seen"] = (~drop_unseen).astype(int)
        # unseen cities borrow the weighted encoding of their nearest known cities
        out["te_pickup"] = np.where(pick_unseen, borrow(tables["te_pickup"], out["pickup"], self.neighbours),
                                    out["pickup"].map(tables["te_pickup"]).fillna(0.0))
        out["te_delivery"] = np.where(drop_unseen, borrow(tables["te_delivery"], out["delivery"], self.neighbours),
                                      out["delivery"].map(tables["te_delivery"]).fillna(0.0))
        # lane encodings: NaN for a new lane (trees learn a default branch; the linear model fills 0)
        out["te_lane"] = out["lane"].map(tables["te_lane"])
        out["te_lane_equipment"] = out["lane_equipment"].map(tables["te_lane_equipment"])
        for w, (level, last, end) in self.levels.items():
            out[f"level_{w}d"] = np.where(out["date"] > end, last, out["date"].map(level))
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._encode(basic_features(df), self.tables)

    def transform_train(self) -> pd.DataFrame:
        parts = []
        folds = KFold(config.TE_FOLDS, shuffle=True, random_state=config.SEED)
        for fit_idx, enc_idx in folds.split(self._train):
            tables = self._tables(self._train.iloc[fit_idx], self._res.iloc[fit_idx])
            parts.append(self._encode(self._train.iloc[enc_idx], tables))
        return pd.concat(parts).loc[self._train.index]


def feature_list(extra=(), drop=()) -> list[str]:
    """BASE_FEATURES minus the FEATURE_GROUPS in `drop`, plus the EXTRA_FEATURES in `extra`."""
    dropped = {c for g in drop for c in config.FEATURE_GROUPS[g]}
    cols = [c for c in config.BASE_FEATURES if c not in dropped]
    for name in extra:
        cols += [c for c in config.EXTRA_FEATURES[name] if c not in cols]
    return cols
