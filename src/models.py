"""Candidate models. Every model predicts log(rate per mile); to_rate() converts to dollars.

All models share one interface: fit(X, y) and predict(X), where X is the encoded
frame from features.FeatureBuilder (it still carries the raw columns).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from . import config
from .data import distance_band
from .features import feature_list
from .geo import nearest_known
from .hierarchy import encode as hier_encode, fit_effects

# scikit-learn 1.6 + recent SciPy emit a harmless "Unknown solver options: iprint" warning
warnings.filterwarnings("ignore", message="Unknown solver options")


def to_rate(log_rpm: np.ndarray, distance: pd.Series) -> np.ndarray:
    """log rate-per-mile -> dollars, clipped to a sensible positive floor."""
    return np.maximum(np.exp(log_rpm) * np.asarray(distance), config.MIN_PREDICTED_RATE)


def recency_weights(X: pd.DataFrame, half_life) -> np.ndarray | None:
    """weight = 0.5 ** (days before the last training day / half_life)."""
    if half_life is None:
        return None
    age = X["t_days"].max() - X["t_days"]
    return 0.5 ** (age.values / half_life)


class LaneMedianBaseline:
    """Median rate-per-mile of the lane+equipment; fallback equipment + distance band.
    The 'what would a dispatcher do with a spreadsheet' benchmark."""

    def fit(self, X, y):
        self.lane = y.groupby(X["lane_equipment"]).median()
        self.band = y.groupby(self._band(X)).median()
        self.global_ = float(y.median())
        return self

    @staticmethod
    def _band(X):
        return X["equipment"] + "|" + distance_band(X["distance"]).astype(str)

    def predict(self, X):
        pred = X["lane_equipment"].map(self.lane).fillna(self._band(X).map(self.band))
        return pred.fillna(self.global_).values


class LinearGAM:
    """Spline on log distance, one-hot categoricals, Huber loss."""

    def __init__(self, features):
        self.features = features

    def fit(self, X, y):
        cats = ["equipment_code", "dayofweek", "pickup_region", "delivery_region"]
        nums = [c for c in self.features if c not in cats + ["log_distance"]]
        pre = ColumnTransformer([
            ("spline", SplineTransformer(n_knots=6, degree=3), ["log_distance"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
            ("num", make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0), StandardScaler()), nums),
        ])
        self.model = make_pipeline(pre, HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000))
        self.model.fit(X[self.features], y)
        return self

    def predict(self, X):
        return self.model.predict(X[self.features])


class HierarchicalShrinkage:
    """Transparent partial-pooling model (mixed-effects style):

        log rpm = base(x) + region_pair + pickup city + delivery city + lane + lane+equipment

    base(x) is least squares on log distance (+ squared), equipment, weight, the
    row-level market deviation and day of week; the effects are hierarchical
    empirical-Bayes deviations (hierarchy.py), each shrunk toward its parent.
    """

    def __init__(self, cities: pd.DataFrame):
        self.cities = cities

    @staticmethod
    def _design(X):
        dow = pd.get_dummies(X["dayofweek"].astype(int)).reindex(columns=range(1, 7), fill_value=0)
        return np.column_stack([
            np.ones(len(X)), X["log_distance"], X["log_distance"] ** 2,
            X["equipment_code"] == 1, X["equipment_code"] == 2,
            X["weight_clean"] / 10_000, X["weight_missing"], X["weight_capped"],
            X["mi_row_dev"], dow.values,
        ]).astype(float)

    def _effects(self, X, unseen_pick, unseen_drop):
        return hier_encode(X, self.effects, unseen_pick, unseen_drop, self.neighbours)["hte_total"].values

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        D = self._design(X)
        self.known = set(X["pickup"]) | set(X["delivery"])
        self.neighbours = nearest_known(self.cities, self.known)
        no = np.zeros(len(X), bool)
        effects = np.zeros(len(X))
        for _ in range(config.HIER_OUTER_ITERS):
            self.coef, *_ = np.linalg.lstsq(D, y - effects, rcond=None)
            self.effects = fit_effects(X, y - D @ self.coef)
            effects = self._effects(X, no, no)
        return self

    def predict(self, X):
        unseen_pick = ~X["pickup"].isin(self.known).values
        unseen_drop = ~X["delivery"].isin(self.known).values
        return self._design(X) @ self.coef + self._effects(X, unseen_pick, unseen_drop)


class GBM:
    """LightGBM or CatBoost on the centred target, with optional recency weights,
    an optional linear time trend (trees then fit the detrended target) and
    averaging over several seeds.

    trend=None | "extrapolate" (keep the line going) | "hold" (line frozen at the
    last training day, i.e. carry the trend's end level forward)."""

    def __init__(self, features, params, half_life=None, trend=None, seeds=1, kind="lightgbm"):
        self.features, self.params, self.half_life = features, params, half_life
        self.trend, self.seeds, self.kind = trend, seeds, kind

    def _trend(self, X):
        if not self.trend:
            return 0.0
        t = X["t_days"].values
        return self.slope * (np.minimum(t, self.t_end) if self.trend == "hold" else t)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.slope = np.polyfit(X["t_days"].values, y, 1)[0] if self.trend else 0.0
        self.t_end = X["t_days"].max()
        y = y - self._trend(X)
        self.offset = float(np.median(y))   # Huber boosting would otherwise start far from the mean
        w = recency_weights(X, self.half_life)
        self.models = []
        for s in range(self.seeds):
            if self.kind == "lightgbm":
                m = LGBMRegressor(**{**self.params, "random_state": config.SEED + s})
            else:
                m = CatBoostRegressor(**{**self.params, "random_seed": config.SEED + s})
            self.models.append(m.fit(X[self.features], y - self.offset, sample_weight=w))
        return self

    def predict(self, X):
        pred = np.mean([m.predict(X[self.features]) for m in self.models], axis=0)
        return pred + self.offset + self._trend(X)


V1_FEATURES = [c for c in feature_list(extra=("market_level", "season"), drop=("mi_row_dev",))] + ["market_index_missing"]


def build_candidates(cities: pd.DataFrame, lgbm_params: dict, half_life) -> dict:
    """name -> zero-argument factory. lgbm_params / half_life come from the tuning step."""
    base = feature_list()
    season = feature_list(extra=("season",))
    cands = {
        "baseline_lane_median": lambda: LaneMedianBaseline(),
        "linear_huber_gam": lambda: LinearGAM(base),
        "hierarchical_shrinkage": lambda: HierarchicalShrinkage(cities),
        "lightgbm": lambda: GBM(base, lgbm_params, seeds=config.N_SEEDS),
        "lightgbm+recency": lambda: GBM(base, lgbm_params, half_life, seeds=config.N_SEEDS),
        "lightgbm+season": lambda: GBM(season, lgbm_params, seeds=config.N_SEEDS),
        "lightgbm_v1": lambda: GBM(V1_FEATURES, config.V1_LGBM_PARAMS),
        "catboost+recency": lambda: GBM(base, config.CATBOOST_PARAMS, half_life, kind="catboost"),
    }
    if half_life is None:   # tuning preferred no weighting: the recency variants equal the plain ones
        cands.pop("lightgbm+recency")
        cands["catboost"] = cands.pop("catboost+recency")
    return cands


def blends(candidates: dict) -> dict:
    """Simple averages in log space of already-fitted members (no extra fitting)."""
    lgbm = "lightgbm+recency" if "lightgbm+recency" in candidates else "lightgbm"
    cat = "catboost+recency" if "catboost+recency" in candidates else "catboost"
    return {"blend_lgbm_catboost": [lgbm, cat], "blend_lgbm_hierarchical": [lgbm, "hierarchical_shrinkage"]}
