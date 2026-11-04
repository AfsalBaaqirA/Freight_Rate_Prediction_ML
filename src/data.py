"""ETL: loading and cleaning.

Rules (from the data audit, see reports/summary.md):
* weight: negative values are sign flips -> abs(); 47,500 is a cap -> flag it;
  missing -> training median of the same equipment type + a missing flag.
* market_index: one national value per day plus small row noise, so a missing
  value is filled with that date's national value (mean over train + validation).
* posted_rate outliers (training data only): rate-per-mile far from the robust
  lane+equipment reference. They are flagged and dropped before fitting.
* quote_signal is read but never used as a feature (it leaks the label, see EDA).
Validation rows are never altered except for these imputations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

NUMERIC = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance",
           "weight", "market_index", "quote_signal"]
EQUIPMENT_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}


def read_loads(path) -> pd.DataFrame:
    """Read a loads CSV with explicit dtypes and a parsed date column."""
    df = pd.read_csv(path)
    for col in NUMERIC + ["posted_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    for col in ["pickup", "delivery", "equipment"]:
        df[col] = df[col].astype(str).str.strip()
    return df


# ---------------------------------------------------------------- weight
def weight_fill_table(train: pd.DataFrame) -> dict:
    """Median cleaned weight per equipment type (training data only) + overall fallback."""
    w = train["weight"].abs().clip(config.WEIGHT_FLOOR, config.WEIGHT_CAP)
    return {"by_equipment": w.groupby(train["equipment"]).median().to_dict(),
            "overall": float(w.median())}


def clean_weight(df: pd.DataFrame, fill_value: dict) -> pd.DataFrame:
    out = df.copy()
    w = out["weight"]
    out["weight_missing"] = w.isna().astype(int)
    out["weight_was_negative"] = (w < 0).astype(int)
    w = w.abs()
    out["weight_capped"] = (w >= config.WEIGHT_CAP).astype(int)
    per_type = out["equipment"].map(fill_value["by_equipment"]).fillna(fill_value["overall"])
    out["weight_clean"] = w.clip(config.WEIGHT_FLOOR, config.WEIGHT_CAP).fillna(per_type)
    return out


# ---------------------------------------------------------------- market_index
def market_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """National daily market_index series (train + validation files, so Nov-Dec and
    the December chart dates are covered) and the variants tested in the feature study."""
    both = pd.concat([f[["date", "market_index"]] for f in frames])
    daily = both.groupby("date")["market_index"].mean()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max())).interpolate()
    t = pd.DataFrame({"mi_day": daily})
    t["mi_7d"] = daily.rolling(7, center=True, min_periods=1).mean()   # level without the weekly cycle
    t["mi_week_dev"] = t["mi_day"] - t["mi_7d"]
    for k in (1, 7, 14):
        t[f"mi_lag_{k}"] = t["mi_7d"].shift(k)
        t[f"mi_lead_{k}"] = t["mi_7d"].shift(-k)
    t["mi_change_7d"] = t["mi_7d"] - t["mi_7d"].shift(7)
    t["mi_ratio_28d"] = daily / daily.rolling(28, min_periods=7).mean()
    return t


def clean_market_index(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Missing market_index -> that date's national value (+ a flag).
    mi_row_dev = this row's value minus the national daily value (0 when missing)."""
    out = df.copy()
    out["market_index_missing"] = out["market_index"].isna().astype(int)
    out = out.join(market, on="date")
    out["market_index_clean"] = out["market_index"].fillna(out["mi_day"])
    out["mi_row_dev"] = out["market_index_clean"] - out["mi_day"]
    out["mi_7d_x_equipment"] = out["mi_7d"] * out["equipment"].map(EQUIPMENT_CODES)
    return out


# ---------------------------------------------------------------- outliers
def distance_band(distance: pd.Series) -> pd.Series:
    return pd.cut(distance, [0, 150, 300, 500, 800, 1200, 2000, 5000], labels=False)


def outlier_reference(train: pd.DataFrame) -> dict:
    """Robust reference rate-per-mile tables built from labelled rows only."""
    rpm = train["posted_rate"] / train["distance"]
    lane = train["pickup"] + ">" + train["delivery"] + "|" + train["equipment"]
    band = train["equipment"] + "|" + distance_band(train["distance"]).astype(str)
    return {"lane": rpm.groupby(lane).agg(["median", "size"]), "band": rpm.groupby(band).median()}


def outlier_ratio(df: pd.DataFrame, ref: dict) -> pd.Series:
    """rate-per-mile / reference: the lane+equipment median when the lane has
    >= MIN_LANE_ROWS rows, otherwise the equipment + distance-band median."""
    rpm = df["posted_rate"] / df["distance"]
    lane = df["pickup"] + ">" + df["delivery"] + "|" + df["equipment"]
    lane_n = lane.map(ref["lane"]["size"]).fillna(0)
    band = df["equipment"] + "|" + distance_band(df["distance"]).astype(str)
    reference = lane.map(ref["lane"]["median"]).where(lane_n >= config.MIN_LANE_ROWS, band.map(ref["band"]))
    return rpm / reference


def flag_outliers(df: pd.DataFrame, ref: dict) -> pd.Series:
    """True where posted_rate is outside [LOW, HIGH] x the robust reference."""
    ratio = outlier_ratio(df, ref)
    return (ratio < config.OUTLIER_LOW) | (ratio > config.OUTLIER_HIGH)


# ---------------------------------------------------------------- loading
def load_all():
    """Cleaned train / validation, the raw December inputs, weight-fill table, market table."""
    train = read_loads(config.TRAIN_PATH)
    valid = read_loads(config.VALID_PATH)
    december = pd.read_csv(config.DECEMBER_PATH)
    weight_fill = weight_fill_table(train)
    market = market_table(train, valid)
    train = clean_market_index(clean_weight(train, weight_fill), market)
    valid = clean_market_index(clean_weight(valid, weight_fill), market)
    return train.reset_index(drop=True), valid.reset_index(drop=True), december, weight_fill, market


def city_lookup(*frames: pd.DataFrame) -> pd.DataFrame:
    """One (lat, lon) per city. The audit found exactly one pair per city."""
    parts = []
    for f in frames:
        parts.append(f[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
        parts.append(f[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).drop_duplicates("city").set_index("city")


def december_frame(december: pd.DataFrame, cities: pd.DataFrame, weight_fill: dict,
                   market: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the columns the December chart file lacks: coordinates from the city
    lookup, market_index = the national value of each date (so no row deviation)."""
    out = december[["pickup", "delivery", "distance", "equipment", "weight", "date"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["distance"] = out["distance"].astype(float)
    out["weight"] = out["weight"].astype(float)
    for end in ("pickup", "delivery"):
        out[f"{end}_lat"] = out[end].map(cities["lat"])
        out[f"{end}_lon"] = out[end].map(cities["lon"])
    out["market_index"] = out["date"].map(market["mi_day"])
    out["quote_signal"] = np.nan
    return clean_market_index(clean_weight(out, weight_fill), market)
