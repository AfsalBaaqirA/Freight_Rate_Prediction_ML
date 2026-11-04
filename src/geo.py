"""Nearest known cities, used whenever a city has no training history.

The given coordinates are noisy (shifted by up to ~5 degrees), so we only use
them for *which cities are close*, never for distance (the `distance` column is
trusted). Great-circle (haversine) distance is used instead of squared degrees:
a degree of longitude is ~20% shorter than a degree of latitude at US latitudes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

EARTH_RADIUS_MILES = 3958.8


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def nearest_known(cities: pd.DataFrame, known) -> dict:
    """For every city in the lookup: its K_NEAREST nearest *other* known cities and
    inverse-distance weights (summing to 1)."""
    known_df = cities.loc[sorted(set(known) & set(cities.index))]
    out = {}
    for city, row in cities.iterrows():
        cand = known_df.drop(index=city, errors="ignore")
        d = haversine_miles(row["lat"], row["lon"], cand["lat"].values, cand["lon"].values)
        idx = np.argsort(d)[:config.K_NEAREST]
        w = 1.0 / np.maximum(d[idx], 1.0)
        out[city] = (list(cand.index[idx]), w / w.sum())
    return out


def borrow(table: pd.Series, city: pd.Series, neighbours: dict) -> pd.Series:
    """Weighted average of `table` (city -> value) over each city's neighbours.
    Neighbours missing from the table are skipped; if none is present -> 0,
    i.e. fall back to the parent / prior level."""
    values = {}
    for c in city.unique():
        names, w = neighbours.get(c, ([], np.array([])))
        v = np.array([table.get(n, np.nan) for n in names], dtype=float)
        ok = ~np.isnan(v)
        values[c] = float(np.sum(w[ok] * v[ok]) / w[ok].sum()) if ok.any() else 0.0
    return city.map(values)
