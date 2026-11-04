"""EDA: figures (reports/figures/01-08) and the facts summary.md quotes.

Besides the usual plots this checks two things that drive the model design:
* quote_signal: is it a usable input, or does it leak the label?
* market_index vs the actual price level over time (why the index level is not a feature).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .data import flag_outliers, outlier_reference


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_eda_figures(train: pd.DataFrame, valid: pd.DataFrame, fig_dir=config.FIGURES) -> list[str]:
    rpm = train["posted_rate"] / train["distance"]
    lane = train["pickup"] + ">" + train["delivery"] + "|" + train["equipment"]
    ratio = rpm / rpm.groupby(lane).transform("median")
    outlier = flag_outliers(train, outlier_reference(train))
    saved = []

    # 1. outliers: ratio of each load's rate-per-mile to its lane median
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.log2(ratio.clip(0.1, 10)), bins=200, color="#064A56")
    ax.set_yscale("log")
    ax.set_xlabel("log2( rate per mile / lane+equipment median )")
    ax.set_ylabel("loads (log scale)")
    ax.set_title(f"posted_rate noise: {outlier.sum()} outliers ({outlier.mean():.1%}) at ~0.2-0.45x and ~2.4-5.4x")
    _save(fig, fig_dir / "01_outliers.png"); saved.append("01_outliers.png")

    # 2. market_index: national daily series (train + validation) and weekly cycle
    both = pd.concat([train.assign(split="train"), valid.assign(split="validation")])
    daily = both.groupby(["date", "split"])["mi_day"].first().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw=dict(width_ratios=[3, 1]))
    for split, color in [("train", "#064A56"), ("validation", "#D9822B")]:
        d = daily[daily["split"] == split]
        axes[0].plot(d["date"], d["mi_day"], color=color, lw=1, label=split)
    smooth = both.groupby("date")["mi_7d"].first()
    axes[0].plot(smooth.index, smooth.values, color="black", lw=1.5, label="7-day level")
    axes[0].set_title("market_index: one national value per day (seasonal + weekly cycle)")
    axes[0].legend()
    dow = both.groupby(both["date"].dt.dayofweek)["mi_week_dev"].mean()
    axes[1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.values, color="#064A56")
    axes[1].set_title("weekly deviation from 7-day level")
    _save(fig, fig_dir / "02_market_index.png"); saved.append("02_market_index.png")

    # 3. rate per mile vs distance (clean loads) by equipment
    clean = train[~outlier].assign(rpm=rpm[~outlier])
    bins = [0, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000]
    clean["band"] = pd.cut(clean["distance"], bins)
    fig, ax = plt.subplots(figsize=(8, 4))
    sample = clean.sample(min(5000, len(clean)), random_state=0)
    ax.scatter(sample["distance"], sample["rpm"], s=2, alpha=0.2, color="grey")
    for eq, color in [("Dry Van", "#064A56"), ("Reefer", "#2B7BD9"), ("Flatbed", "#D9822B")]:
        med = clean[clean["equipment"] == eq].groupby("band", observed=True).agg(d=("distance", "median"), r=("rpm", "median"))
        ax.plot(med["d"], med["r"], marker="o", color=color, label=eq)
    ax.set_xlabel("distance (miles)"); ax.set_ylabel("rate per mile ($)")
    ax.set_ylim(1.2, 4); ax.legend()
    ax.set_title("Short hauls cost more per mile; Reefer > Flatbed > Dry Van")
    _save(fig, fig_dir / "03_rate_per_mile_vs_distance.png"); saved.append("03_rate_per_mile_vs_distance.png")

    # 4. equipment premium within the same lane
    lanes = clean.groupby([clean["pickup"] + ">" + clean["delivery"], "equipment"])["rpm"].median().unstack()
    prem = pd.Series({"Reefer": (lanes["Reefer"] / lanes["Dry Van"]).median() - 1,
                      "Flatbed": (lanes["Flatbed"] / lanes["Dry Van"]).median() - 1})
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(prem.index, prem.values * 100, color=["#2B7BD9", "#D9822B"])
    for i, v in enumerate(prem.values):
        ax.text(i, v * 100 + 0.3, f"+{v:.1%}", ha="center")
    ax.set_ylabel("premium vs Dry Van on same lane (%)")
    ax.set_title("Equipment premium")
    _save(fig, fig_dir / "04_equipment_premium.png"); saved.append("04_equipment_premium.png")

    # 5. data-quality overview
    issues = pd.DataFrame({
        "train": [train["weight_missing"].sum(), train["weight_was_negative"].sum(), train["weight_capped"].sum(),
                  train["market_index_missing"].sum(), int(outlier.sum())],
        "validation": [valid["weight_missing"].sum(), valid["weight_was_negative"].sum(), valid["weight_capped"].sum(),
                       valid["market_index_missing"].sum(), 0],
    }, index=["weight missing", "weight negative", "weight capped 47.5k", "market_index missing", "rate outliers"])
    fig, ax = plt.subplots(figsize=(8, 4))
    issues.plot.barh(ax=ax, color=["#064A56", "#D9822B"])
    ax.set_title("Data-quality issues (row counts)")
    _save(fig, fig_dir / "05_data_quality.png"); saved.append("05_data_quality.png")

    # 6. unseen cities in validation
    seen = set(train["pickup"]) | set(train["delivery"])
    new = sorted((set(valid["pickup"]) | set(valid["delivery"])) - seen)
    counts = pd.Series({c: int(((valid["pickup"] == c) | (valid["delivery"] == c)).sum()) for c in new})
    if len(counts):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        counts.sort_values().plot.barh(ax=ax, color="#D9822B")
        ax.set_title(f"{len(new)} cities only in validation.csv ({(valid['pickup'].isin(new) | valid['delivery'].isin(new)).mean():.1%} of rows)")
        _save(fig, fig_dir / "06_unseen_cities.png"); saved.append("06_unseen_cities.png")
        return saved


def quote_signal_check(train: pd.DataFrame, valid: pd.DataFrame, fig_dir=config.FIGURES) -> pd.DataFrame:
    """Per month: correlation of quote_signal with the actual rate per mile (clean rows)
    and the fitted line quote = a + b * rpm. Validation has no labels, so there we use
    the lane+equipment median rpm from training as the stand-in."""
    clean = train[~flag_outliers(train, outlier_reference(train))].copy()
    clean["rpm"] = clean["posted_rate"] / clean["distance"]
    rows = {}
    for m, g in clean.groupby(clean["date"].dt.month):
        b, a = np.polyfit(g["rpm"], g["quote_signal"], 1)
        rows[f"2025-{m:02d} (train)"] = dict(corr=g[["quote_signal", "rpm"]].corr().iloc[0, 1], slope=b, intercept=a,
                                            resid_sd=float(np.std(g["quote_signal"] - (a + b * g["rpm"]))), rows=len(g))
    lane_rpm = clean.groupby(["pickup", "delivery", "equipment"])["rpm"].median().rename("lane_rpm")
    v = valid.join(lane_rpm, on=["pickup", "delivery", "equipment"]).dropna(subset=["lane_rpm", "quote_signal"])
    t = clean.join(lane_rpm, on=["pickup", "delivery", "equipment"])
    for m, g in v.groupby(v["date"].dt.month):
        rows[f"2025-{m:02d} (validation)"] = dict(corr=g[["quote_signal", "lane_rpm"]].corr().iloc[0, 1], rows=len(g))
    table = pd.DataFrame(rows).T
    table["corr_vs_lane_median_rpm"] = [t[t["date"].dt.month == int(i[5:7])][["quote_signal", "lane_rpm"]].corr().iloc[0, 1]
                                        if "train" in i else table.loc[i, "corr"] for i in table.index]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    colors = ["#064A56" if "train" in i else "#D9822B" for i in table.index]
    ax.bar([i[:7] for i in table.index], table["corr_vs_lane_median_rpm"], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("corr(quote_signal, lane rate per mile)")
    ax.set_title("quote_signal = +rpm or -rpm depending on the month (train); noise in Aug and in validation (orange)")
    _save(fig, fig_dir / "07_quote_signal_leak.png")
    return table


def price_level_vs_market(train: pd.DataFrame, fig_dir=config.FIGURES) -> pd.DataFrame:
    """Monthly price level (log rpm minus each lane+equipment's mean, clean rows)
    next to the monthly mean market_index."""
    clean = train[~flag_outliers(train, outlier_reference(train))].copy()
    y = np.log(clean["posted_rate"] / clean["distance"])
    key = clean["pickup"] + ">" + clean["delivery"] + "|" + clean["equipment"]
    clean["level"] = y - y.groupby(key).transform("mean")
    month = clean["date"].dt.month
    table = pd.DataFrame({"price_level_pct": clean.groupby(month)["level"].mean() * 100,
                          "market_index": clean.groupby(month)["mi_day"].mean()})
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(table.index, table["price_level_pct"], marker="o", color="#064A56", label="price level (% vs lane mean)")
    ax.set_ylabel("price level (%)")
    ax2 = ax.twinx()
    ax2.plot(table.index, table["market_index"], marker="s", color="#D9822B", label="market_index (monthly mean)")
    ax2.set_ylabel("market_index")
    ax.set_xlabel("month 2025")
    ax.set_title("Price level and market_index move together Jan-Jul, then split from August")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.88))
    _save(fig, fig_dir / "08_price_level_vs_market_index.png")
    return table
