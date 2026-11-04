"""Time-based cross-validation: tuning, model comparison, feature study, calibration.

Folds (validation is Nov-Dec, so every fold trains on the past, tests on the future):
* rolling_M    : train on months < M, test on month M (M = Jul, Aug, Sep, Oct)
* forward      : train Jan-Aug, test Sep-Oct (the headline split)
* cold_start_i : like forward, but 8 random cities are removed from training,
                 mimicking the 8 brand-new cities in validation.csv. Metrics on the
                 rows touching those cities = "unseen-city" metrics (3 city sets).

Inside every fold the outlier filter and encodings are fitted on the training part
only. "Clean" test rows are those not flagged as injected rate noise by a reference
built from all labelled data; this is used only for scoring, never for training.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .data import flag_outliers, outlier_reference
from .features import FeatureBuilder, feature_list, target
from .models import GBM, LaneMedianBaseline, blends, build_candidates, to_rate


def make_folds(train: pd.DataFrame) -> list[dict]:
    month = train["date"].dt.month
    folds = [dict(name=f"rolling_{m:02d}", train=month < m, test=month == m, cities=[])
             for m in config.ROLLING_MONTHS]
    folds.append(dict(name="forward", train=month <= 8, test=month >= 9, cities=[]))
    cities = sorted(set(train["pickup"]) | set(train["delivery"]))
    for i in range(1, config.COLD_START_REPEATS + 1):
        rng = np.random.default_rng(config.SEED + 100 * i)      # a different city set per repeat
        held = list(rng.choice(cities, size=8, replace=False))
        touches = train["pickup"].isin(held) | train["delivery"].isin(held)
        folds.append(dict(name=f"cold_start_{i}", train=(month <= 8) & ~touches, test=month >= 9, cities=held))
    return folds


@dataclass
class Fold:
    """Everything a model needs for one fold."""
    name: str
    X_tr: pd.DataFrame       # out-of-fold encoded training rows (outliers dropped)
    y_tr: pd.Series
    X_te: pd.DataFrame
    actual: np.ndarray       # test posted_rate
    clean: np.ndarray        # test rows that are not injected rate noise
    unseen: np.ndarray       # test rows touching a held-out city
    X_tr_all: pd.DataFrame   # all training rows incl. outliers (for calibration factors)
    cold: bool


def prepare(train, fold, cities, eval_clean, drop_outliers=True) -> Fold:
    tr_all = train[fold["train"]]
    tr = tr_all[~flag_outliers(tr_all, outlier_reference(tr_all))] if drop_outliers else tr_all
    te = train[fold["test"]]
    fb = FeatureBuilder(cities).fit(tr)
    unseen = (te["pickup"].isin(fold["cities"]) | te["delivery"].isin(fold["cities"])).values
    return Fold(fold["name"], fb.transform_train(), target(tr), fb.transform(te), te["posted_rate"].values,
                eval_clean[te.index].values, unseen, fb.transform(tr_all), bool(fold["cities"]))


def prepare_all(train, cities):
    eval_clean = ~flag_outliers(train, outlier_reference(train))
    return [prepare(train, f, cities, eval_clean) for f in make_folds(train)], eval_clean


def metrics(y, p) -> dict:
    err = p - y
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.mean(np.abs(err))),
                mape=float(np.mean(np.abs(err) / y)), bias_pct=float(100 * np.mean(err / y)))


def score(f: Fold, rate: np.ndarray) -> list[dict]:
    subsets = [("all", np.ones(len(rate), bool)), ("clean", f.clean)] + ([("unseen_city", f.unseen)] if f.cold else [])
    return [dict(fold=f.name, subset=s, **metrics(f.actual[m], rate[m])) for s, m in subsets]


def time_summary(rows: pd.DataFrame) -> dict:
    """Mean over the 5 time folds of all MAE / RMSE and clean MAE / RMSE."""
    t = rows[~rows["fold"].str.startswith("cold")].pivot_table(index="fold", columns="subset", values=["mae", "rmse"])
    out = {f"{s}_{m}": t[(m, s)].mean() for m in ["mae", "rmse"] for s in ["all", "clean"]}
    out.update({f"{f}_clean_mae": t.loc[f, ("mae", "clean")] for f in t.index})
    b = rows[(rows["fold"] == "rolling_07") & (rows["subset"] == "clean")]["bias_pct"]
    out["rolling_07_clean_bias_pct"] = float(b.iloc[0]) if len(b) else np.nan
    return out


# ------------------------------------------------------------------ 1. tuning
def tune(folds: list[Fold]) -> tuple[pd.DataFrame, dict, object]:
    """Small grid (TUNING_GRID x HALF_LIVES) for LightGBM on the 5 time folds, single seed.
    Score = mean of (all MAE, clean RMSE) relative to the lane-median baseline."""
    time_folds = [f for f in folds if not f.cold]
    base = time_summary(pd.DataFrame([r for f in time_folds for r in score(
        f, to_rate(LaneMedianBaseline().fit(f.X_tr, f.y_tr).predict(f.X_te), f.X_te["distance"]))]))
    rows = []
    keys = list(config.TUNING_GRID)
    for values in itertools.product(*config.TUNING_GRID.values()):
        params = {**config.LGBM_PARAMS, **dict(zip(keys, values))}
        for hl in config.HALF_LIVES:
            res = []
            for f in time_folds:
                m = GBM(feature_list(), params, hl).fit(f.X_tr, f.y_tr)
                res += score(f, to_rate(m.predict(f.X_te), f.X_te["distance"]))
            s = time_summary(pd.DataFrame(res))
            s["tuning_score"] = (s["all_mae"] / base["all_mae"] + s["clean_rmse"] / base["clean_rmse"]) / 2
            rows.append(dict(zip(keys, values), half_life=hl if hl else "none", **s))
            print(f"    tune {dict(zip(keys, values))} half_life={hl}: score {s['tuning_score']:.4f}", flush=True)
    table = pd.DataFrame(rows).sort_values("tuning_score").reset_index(drop=True)
    best = table.iloc[0]
    params = {**config.LGBM_PARAMS, **{k: int(best[k]) for k in keys}}
    return table, params, (None if best["half_life"] == "none" else int(best["half_life"]))


# ------------------------------------------------------------------ 2. model comparison
def compare(folds: list[Fold], cities, params, half_life):
    """Fit every candidate on every fold. Returns (long metrics table, per-fold predictions)."""
    cands = build_candidates(cities, params, half_life)
    rows, preds = [], {}
    for f in folds:
        p = {}
        for name, make in cands.items():
            m = make().fit(f.X_tr, f.y_tr)
            p[name] = (m.predict(f.X_te), m.predict(f.X_tr_all))     # (test, all training rows)
        for name, members in blends(cands).items():
            p[name] = tuple(np.mean([p[k][i] for k in members], axis=0) for i in (0, 1))
        for name, (te, _) in p.items():
            rows += [dict(model=name, **r) for r in score(f, to_rate(te, f.X_te["distance"]))]
        preds[f.name] = p
        print(f"    fold {f.name:<13} train={len(f.y_tr):>6} test={len(f.actual):>6} done", flush=True)
    return pd.DataFrame(rows), preds


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """One row per model: time-fold averages (rolling + forward) and cold-start
    unseen-city metrics averaged over the 3 city sets, plus the selection score."""
    cold = results["fold"].str.startswith("cold_start")
    wide = results[~cold].pivot_table(index="model", columns="subset", values=["rmse", "mae", "mape"], aggfunc="mean")
    wide.columns = [f"{s}_{m}" for m, s in wide.columns]
    unseen = results[cold & (results["subset"] == "unseen_city")].groupby("model")[["rmse", "mae", "mape"]].mean()
    for m in ["rmse", "mae", "mape"]:
        wide[f"unseen_city_{m}"] = unseen[m]
    base = wide.loc["baseline_lane_median"]
    # mean of three error ratios vs the baseline (lower is better): MAE on all rows,
    # RMSE on clean rows, MAE on unseen-city rows (12% of validation touches a new city)
    wide["selection_score"] = (wide["all_mae"] / base["all_mae"] + wide["clean_rmse"] / base["clean_rmse"]
                               + wide["unseen_city_mae"] / base["unseen_city_mae"]) / 3
    cols = ["selection_score", "all_rmse", "all_mae", "all_mape", "clean_rmse", "clean_mae", "clean_mape",
            "unseen_city_rmse", "unseen_city_mae", "unseen_city_mape"]
    return wide[cols].sort_values("selection_score")


# ------------------------------------------------------------------ 3. calibration
def calibration_factor(method, train_pred_log, X_tr_all) -> float:
    """Global multiplier estimated on the fold's *training* rows (incl. outliers):
    rmse_multiplier = least-squares scale sum(y*p)/sum(p^2);
    smearing        = mean(y/p), Duan's mean-vs-median correction."""
    if method is None:
        return 1.0
    y = X_tr_all["posted_rate"].values
    p = to_rate(train_pred_log, X_tr_all["distance"])
    return float(np.sum(y * p) / np.sum(p ** 2)) if method == "rmse_multiplier" else float(np.mean(y / p))


def calibration_study(folds, preds, model) -> pd.DataFrame:
    rows = []
    for method in [None, "rmse_multiplier", "smearing"]:
        res, factors = [], []
        for f in folds:
            te, tr_all = preds[f.name][model]
            c = calibration_factor(method, tr_all, f.X_tr_all)
            factors.append(c)
            res += [dict(model=method or "none", **r) for r in score(f, c * to_rate(te, f.X_te["distance"]))]
        s = summarise_like(pd.DataFrame(res))
        s["mean_factor"] = np.mean(factors)
        rows.append(s)
    return pd.DataFrame(rows).set_index("calibration")


def summarise_like(res: pd.DataFrame) -> dict:
    cold = res["fold"].str.startswith("cold")
    t = res[~cold].groupby("subset")[["mae", "rmse"]].mean()
    u = res[cold & (res["subset"] == "unseen_city")][["mae", "rmse"]].mean()
    return dict(calibration=res["model"].iloc[0], all_mae=t.loc["all", "mae"], all_rmse=t.loc["all", "rmse"],
                clean_mae=t.loc["clean", "mae"], clean_rmse=t.loc["clean", "rmse"],
                unseen_city_mae=u["mae"], unseen_city_rmse=u["rmse"])


# ------------------------------------------------------------------ 4. feature study
def feature_study(folds, train, cities, eval_clean, params, half_life) -> pd.DataFrame:
    """Refit LightGBM (single seed, tuned settings) on the 5 time folds with one change
    at a time. Refitting (not permuting) is the honest test for time features: a
    permuted month/date column breaks its link with the target in a way the model
    never saw, while a refit shows what the model can do without it."""
    time_folds = [f for f in folds if not f.cold]
    base, hl = feature_list(), half_life
    variants = [  # (section, name, features, half_life, trend)
        ("time level", "pooled level (no time feature, no weights)", base, None, None),
        ("time level", f"recency weights, half-life {hl} days", base, hl, None),
        ("time level", "linear trend, extrapolated + trees on detrended target", base, None, "extrapolate"),
        ("time level", "linear trend, held at last training day + trees on detrended target", base, None, "hold"),
        ("time level", "trailing 30-day recent level", feature_list(("recent_level_30",)), None, None),
        ("time level", "trailing 60-day recent level", feature_list(("recent_level_60",)), None, None),
        ("time level", "season (month + day of year)", feature_list(("season",)), None, None),
    ]
    for name in ["market_level", "mi_lag_1", "mi_lag_7", "mi_lag_14", "mi_lead_1", "mi_lead_7", "mi_lead_14",
                 "mi_change_7d", "mi_ratio_28d", "mi_x_equipment", "quote_signal"]:
        variants.append(("added to shipped recipe", f"+ {name}", feature_list((name,)), hl, None))
    for group in config.FEATURE_GROUPS:
        variants.append(("removed from shipped recipe", f"- {group}", feature_list(drop=(group,)), hl, None))
    rows = []
    for section, name, feats, h, trend in variants:
        res = []
        for f in time_folds:
            m = GBM(feats, params, h, trend).fit(f.X_tr, f.y_tr)
            res += score(f, to_rate(m.predict(f.X_te), f.X_te["distance"]))
        rows.append(dict(section=section, variant=name, **time_summary(pd.DataFrame(res))))
        print(f"    study {name}: clean MAE {rows[-1]['clean_mae']:.1f}", flush=True)
    # training on every row (outliers kept) instead of dropping the flagged ones
    res = []
    for fold in [f for f in make_folds(train) if not f["cities"]]:
        f = prepare(train, fold, cities, eval_clean, drop_outliers=False)
        m = GBM(base, params, hl).fit(f.X_tr, f.y_tr)
        res += score(f, to_rate(m.predict(f.X_te), f.X_te["distance"]))
    rows.append(dict(section="cleaning", variant="keep flagged outlier rows in training", **time_summary(pd.DataFrame(res))))
    return pd.DataFrame(rows)
