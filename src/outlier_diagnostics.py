"""Are the posted_rate spikes random noise, or do they depend on something?

Before dropping flagged rows from training we test whether the outlier flag
(same rule as src/data.py: rate-per-mile < 0.6x or > 1.6x the lane+equipment
median, equipment + distance-band median for lanes with < 3 loads) or the size
of the ratio is related to any observable column.

Only train_test.csv has labels (validation.csv has no posted_rate), so all of
this runs on the training file.

Tests
1. Outlier rate by group with Wilson 95% intervals and a chi-square test per
   factor (Holm-corrected across all factors); separately for high spikes and
   low dips; Kruskal-Wallis on the spike size within flagged rows.
2. Mann-Whitney tests of every numeric column, flagged vs not flagged.
3. Classifiers predicting the flag from every column (logistic regression and a
   small LightGBM), time-based CV (train on months before M, test on M);
   pooled out-of-fold AUC with a bootstrap 95% interval. AUC ~0.5 = random.
   Same for high-vs-low among flagged rows.
4. Clustering: daily counts vs Poisson (dispersion test, lag-1 autocorrelation),
   per-lane counts vs binomial, and repeat-on-same-lane.
5. Up/down split (binomial test vs 50/50) and the multiplier distribution.

Decision rule: a factor "explains" spikes only if its Holm-adjusted p < 0.01 AND
the classifier's AUC lower bound is above 0.55. If nothing passes, the flagged
rows are treated as random label noise and dropped (the keep-all vs drop-all
comparison is in the feature study of reports/summary.md).
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

from . import config
from .data import distance_band as _distance_band, outlier_reference
from .features import region

warnings.filterwarnings("ignore", message="Unknown solver options")

# US federal holidays + the big freight dates inside / around the data period
HOLIDAYS = {
    "New Year": "2025-01-01", "MLK Day": "2025-01-20", "Presidents Day": "2025-02-17",
    "Memorial Day": "2025-05-26", "Juneteenth": "2025-06-19", "July 4": "2025-07-04",
    "Labor Day": "2025-09-01", "Columbus Day": "2025-10-13", "Veterans Day": "2025-11-11",
    "Thanksgiving": "2025-11-27", "Christmas": "2025-12-25", "New Year 2026": "2026-01-01",
}
HOLIDAY_WINDOW = 3   # days either side
ALPHA = 0.01
AUC_THRESHOLD = 0.55

CATEGORICAL = {
    "equipment": "equipment", "distance_band": "distance band", "pickup": "pickup city",
    "delivery": "delivery city", "pickup_region": "pickup region", "delivery_region": "delivery region",
    "region_pair": "region pair", "lane_size": "lane size (loads)", "reference": "reference used",
    "month": "month", "dayofweek": "day of week", "dayofmonth": "day of month",
    "holiday_window": f"within {HOLIDAY_WINDOW} days of a holiday", "holiday_name": "nearest holiday (in window)",
    "mi_q": "market_index quintile", "mi_change_q": "market_index daily-change quintile",
    "mi_week_dev_q": "market_index weekly-deviation quintile", "weight_q": "weight quintile",
    "weight_missing": "weight missing", "weight_was_negative": "weight negative",
    "weight_capped": "weight capped", "market_index_missing": "market_index missing",
    "quote_q": "quote_signal quintile", "load_id_q": "load_id order quintile",
}
NUMERIC = ["distance", "weight_clean", "market_index_clean", "mi_day", "mi_change", "mi_week_dev",
           "quote_signal", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "lane_n",
           "days_to_holiday", "dayofmonth", "load_seq", "lane_prior_flag_rate"]


# ---------------------------------------------------------------- frame
def outlier_frame(train: pd.DataFrame) -> pd.DataFrame:
    """Training rows + flag, direction, ratio to reference and every factor."""
    df = train.copy()
    ref = outlier_reference(df)
    rpm = df["posted_rate"] / df["distance"]
    lane = df["pickup"] + ">" + df["delivery"] + "|" + df["equipment"]
    lane_n = lane.map(ref["lane"]["size"]).fillna(0)
    band = df["equipment"] + "|" + _distance_band(df["distance"]).astype(str)
    use_lane = lane_n >= config.MIN_LANE_ROWS
    reference = lane.map(ref["lane"]["median"]).where(use_lane, band.map(ref["band"]))
    df["ratio"] = rpm / reference
    df["high"] = (df["ratio"] > config.OUTLIER_HIGH).astype(int)
    df["low"] = (df["ratio"] < config.OUTLIER_LOW).astype(int)
    df["flag"] = df["high"] | df["low"]
    df["direction"] = np.select([df["high"] == 1, df["low"] == 1], ["high", "low"], "normal")

    df["lane_key"] = lane
    df["lane_n"] = lane_n
    df["lane_size"] = pd.cut(lane_n, [0, 2, 9, 29, 99, 1e9], labels=["01-02", "03-09", "10-29", "30-99", "100+"]).astype(str)
    df["reference"] = np.where(use_lane, "lane median", "equipment+band median")
    df["distance_band"] = _distance_band(df["distance"]).astype(int)
    df["pickup_region"] = region(df["pickup_lon"])
    df["delivery_region"] = region(df["delivery_lon"])
    df["region_pair"] = df["pickup_region"].astype(str) + "-" + df["delivery_region"].astype(str)
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofmonth"] = df["date"].dt.day

    hol = pd.Series(pd.to_datetime(list(HOLIDAYS.values())), index=list(HOLIDAYS.keys()))
    gaps = np.abs((df["date"].values[:, None] - hol.values[None, :]).astype("timedelta64[D]").astype(int))
    df["days_to_holiday"] = gaps.min(axis=1)
    df["holiday_window"] = (df["days_to_holiday"] <= HOLIDAY_WINDOW).astype(int)
    df["holiday_name"] = np.where(df["holiday_window"] == 1, hol.index.values[gaps.argmin(axis=1)], "none")

    daily = df.groupby("date")["mi_day"].first().sort_index()
    df["mi_change"] = df["date"].map(daily.diff()).fillna(0.0)
    # how often this lane was flagged BEFORE this load (time-ordered, smoothed toward the overall rate)
    order = df.sort_values(["date", "load_id"]).index
    s = df.loc[order]
    g = s.groupby("lane_key")["flag"]
    prev_k = g.cumsum() - s["flag"]
    prev_n = g.cumcount()
    base = df["flag"].mean()
    df.loc[order, "lane_prior_flag_rate"] = ((prev_k + 10 * base) / (prev_n + 10)).values
    df.loc[order, "lane_prev_flagged"] = g.shift(1).fillna(0).values
    df["load_seq"] = df["load_id"].str.extract(r"(\d+)$")[0].astype(float)
    for col, name in [("market_index_clean", "mi_q"), ("mi_change", "mi_change_q"),
                      ("mi_week_dev", "mi_week_dev_q"), ("weight_clean", "weight_q"),
                      ("quote_signal", "quote_q"), ("load_seq", "load_id_q")]:
        df[name] = pd.qcut(df[col].rank(method="first"), 5, labels=False)
    return df


# ---------------------------------------------------------------- helpers
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def holm(pvals: pd.Series) -> pd.Series:
    order = pvals.sort_values()
    m = len(order)
    adj = np.maximum.accumulate([(m - i) * p for i, p in enumerate(order.values)])
    return pd.Series(np.minimum(adj, 1.0), index=order.index).reindex(pvals.index)


def group_rates(df, col):
    g = df.groupby(col)
    t = pd.DataFrame({"n": g.size(), "flagged": g["flag"].sum(), "high": g["high"].sum(), "low": g["low"].sum()})
    t["rate"] = t["flagged"] / t["n"]
    t["ci_low"], t["ci_high"] = wilson(t["flagged"], t["n"])
    return t


def factor_tests(df) -> pd.DataFrame:
    rows = {}
    overall = df["flag"].mean()
    flagged = df[df["flag"] == 1]
    for col, label in CATEGORICAL.items():
        t = group_rates(df, col)
        table = pd.crosstab(df[col], df["direction"])
        chi = stats.chi2_contingency(pd.crosstab(df[col], df["flag"]))
        chi3 = stats.chi2_contingency(table)
        n = len(df)
        cramer = np.sqrt(chi[0] / (n * (min(pd.crosstab(df[col], df["flag"]).shape) - 1)))
        big = t[t["n"] >= 100]
        # high-vs-low split inside flagged rows, and spike size within each direction
        hl = pd.crosstab(flagged[col], flagged["direction"])
        p_hl = stats.chi2_contingency(hl)[1] if hl.shape[0] > 1 and hl.shape[1] > 1 else np.nan
        size_p = []
        for d in ("high", "low"):
            sub = flagged[flagged["direction"] == d]
            groups = [np.log(s["ratio"]).abs().values for _, s in sub.groupby(col) if len(s) >= 5]
            size_p.append(stats.kruskal(*groups)[1] if len(groups) > 1 else np.nan)
        rows[label] = {
            "groups": len(t), "chi2_p": chi[1], "dir3_p": chi3[1], "cramers_v": cramer,
            "high_vs_low_p": p_hl, "size_high_p": size_p[0], "size_low_p": size_p[1],
            "min_rate": big["rate"].min(), "max_rate": big["rate"].max(),
            "groups_ci_excl_overall": int(((big["ci_low"] > overall) | (big["ci_high"] < overall)).sum()),
        }
    out = pd.DataFrame(rows).T
    allp = pd.concat({c: out[c] for c in ["chi2_p", "dir3_p", "high_vs_low_p", "size_high_p", "size_low_p"]}).dropna()
    adj = holm(allp.astype(float))
    for c in ["chi2_p", "dir3_p", "high_vs_low_p", "size_high_p", "size_low_p"]:
        out[c + "_holm"] = adj.get(c, pd.Series(dtype=float)).reindex(out.index)
    return out


def numeric_tests(df) -> pd.DataFrame:
    rows = {}
    f = df["flag"] == 1
    for col in NUMERIC:
        a, b = df.loc[f, col], df.loc[~f, col]
        u = stats.mannwhitneyu(a, b)
        rows[col] = {"median_flagged": a.median(), "median_normal": b.median(), "mw_p": u.pvalue,
                     "auc_single": u.statistic / (len(a) * len(b))}
        hi, lo = df.loc[df["high"] == 1, col], df.loc[df["low"] == 1, col]
        rows[col]["high_vs_low_p"] = stats.mannwhitneyu(hi, lo).pvalue
        sub = df[f]
        rows[col]["spearman_size_p"] = stats.spearmanr(sub[col], np.log(sub["ratio"]).abs()).pvalue
    out = pd.DataFrame(rows).T
    for c in ["mw_p", "high_vs_low_p", "spearman_size_p"]:
        out[c + "_holm"] = holm(out[c].astype(float))
    return out


# ---------------------------------------------------------------- classifiers
CLF_NUM = NUMERIC + ["lane_prev_flagged", "weight_missing", "weight_was_negative", "weight_capped", "market_index_missing",
                     "month", "dayofweek", "holiday_window", "distance_band"]
CLF_CAT = ["equipment", "pickup", "delivery", "region_pair"]


def _models():
    logit = make_pipeline(
        ColumnTransformer([("num", StandardScaler(), CLF_NUM),
                           ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CLF_CAT)]),
        LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
    gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             random_state=config.SEED, verbose=-1, n_jobs=config.N_JOBS)
    return {"logistic regression": logit, "LightGBM classifier": gbm}


def boot_auc(y, p, n_boot, seed):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


def classifier_auc(df, target, rows_mask=None):
    """Time-based CV: train on months < M, predict month M. Returns per-fold and pooled AUC."""
    d = df if rows_mask is None else df[rows_mask]
    d = d.copy()
    cats = {c: pd.CategoricalDtype(sorted(df[c].astype(str).unique())) for c in CLF_CAT}
    for c in CLF_CAT:
        d[c] = d[c].astype(str).astype(cats[c])
    months = sorted(d["month"].unique())[3:]
    res = {}
    for name, model in _models().items():
        oof_y, oof_p, folds = [], [], {}
        for m in months:
            tr, te = d[d["month"] < m], d[d["month"] == m]
            if tr[target].nunique() < 2 or te[target].nunique() < 2:
                continue
            Xtr, Xte = tr[CLF_NUM + CLF_CAT], te[CLF_NUM + CLF_CAT]
            if name.startswith("logistic"):
                Xtr, Xte = Xtr.astype({c: str for c in CLF_CAT}), Xte.astype({c: str for c in CLF_CAT})
            model.fit(Xtr, tr[target])
            p = model.predict_proba(Xte)[:, 1]
            folds[int(m)] = roc_auc_score(te[target], p)
            oof_y.append(te[target].values); oof_p.append(p)
        y, p = np.concatenate(oof_y), np.concatenate(oof_p)
        lo, hi = boot_auc(y, p, 1000, config.SEED)
        res[name] = {"pooled_auc": roc_auc_score(y, p), "ci_low": lo, "ci_high": hi,
                     "mean_fold_auc": np.mean(list(folds.values())), "folds": folds, "n_pos": int(y.sum()), "n": len(y)}
    return res


# ---------------------------------------------------------------- clustering
def _lane_dispersion(d) -> float:
    lanes = d.groupby("lane_key").agg(n=("flag", "size"), k=("flag", "sum"))
    big = lanes[lanes["n"] >= 10]
    if len(big) < 2:
        return float("nan")
    p = d["flag"].mean()
    e = big["n"] * p
    return float((((big["k"] - e) ** 2) / (e * (1 - p))).sum() / (len(big) - 1))


def simulated_lane_null(train, df, reps, seed=config.SEED):
    """Lane dispersion if corruption were purely random: replace flagged prices by
    clean lane-level prices, corrupt a random 1.4% of rows with multipliers drawn
    from the observed ones, re-run the same flag rule. The flag rule itself can
    create some lane-level dispersion, so this is the fair null, not 1.0."""
    rng = np.random.default_rng(seed)
    clean = df[df["flag"] == 0]
    lane_rpm = (clean["posted_rate"] / clean["distance"]).groupby(clean["lane_key"]).median()
    ref_rpm = df["posted_rate"] / df["distance"] / df["ratio"]
    base = df["lane_key"].map(lane_rpm).fillna(ref_rpm)
    sd = np.log(df.loc[df["flag"] == 0, "ratio"]).std()
    mult = df.loc[df["flag"] == 1, "ratio"].values
    rate0 = np.where(df["flag"] == 1, base * df["distance"], df["posted_rate"])
    sims = []
    for _ in range(reps):
        sim = train.copy()
        r = np.where(df["flag"] == 1, rate0 * np.exp(rng.normal(0, sd, len(df))), rate0)
        pick = rng.random(len(df)) < df["flag"].mean()
        sim["posted_rate"] = np.where(pick, r * rng.choice(mult, len(df)), r)
        sims.append(_lane_dispersion(outlier_frame(sim)))
    return np.array(sims)


def clustering(df, train=None, reps=0) -> dict:
    out = {}
    daily = df.groupby("date").agg(n=("flag", "size"), k=("flag", "sum"))
    p = df["flag"].mean()
    expected = daily["n"] * p
    chi = (((daily["k"] - expected) ** 2) / (expected * (1 - p))).sum()
    dof = len(daily) - 1
    out["daily_dispersion"] = chi / dof
    out["daily_dispersion_p"] = stats.chi2.sf(chi, dof)
    rate = daily["k"] / daily["n"]
    out["daily_lag1_autocorr"] = rate.autocorr(1)
    out["daily_lag7_autocorr"] = rate.autocorr(7)
    out["max_daily_count"] = int(daily["k"].max())
    out["days"] = len(daily)
    out["poisson_p_max_day"] = float(stats.binom.sf(daily["k"].max() - 1, int(daily["n"].loc[daily["k"].idxmax()]), p) * len(daily))

    lanes = df.groupby("lane_key").agg(n=("flag", "size"), k=("flag", "sum"))
    big = lanes[lanes["n"] >= 10]
    e = big["n"] * p
    chi_l = (((big["k"] - e) ** 2) / (e * (1 - p))).sum()
    out["lanes_tested"] = len(big)
    dof_l = max(len(big) - 1, 1)
    out["lane_dispersion"] = chi_l / dof_l if len(big) > 1 else float("nan")
    out["lane_dispersion_p"] = stats.chi2.sf(chi_l, dof_l) if len(big) > 1 else float("nan")
    big = big.assign(p=stats.binom.sf(big["k"] - 1, big["n"], p))
    big["p_holm"] = holm(big["p"])
    out["top_lanes"] = big.sort_values("p").head(5)
    out["lanes_significant"] = int((big["p_holm"] < ALPHA).sum())
    out["lane_sim_reps"] = reps
    if reps and len(big) > 1:
        sims = simulated_lane_null(train, df, reps)
        sims = sims[~np.isnan(sims)]
        out["lane_sim_p95"] = float(np.percentile(sims, 95))
        out["lane_sim_median"] = float(np.median(sims))
        out["lane_sim_p"] = float((1 + (sims >= out["lane_dispersion"]).sum()) / (len(sims) + 1))
    se = 1 / np.sqrt(len(daily))
    out["lag1_p"] = float(2 * stats.norm.sf(abs(out["daily_lag1_autocorr"]) / se))

    s = df.sort_values(["lane_key", "date", "load_id"])
    prev = s.groupby("lane_key")["flag"].shift(1)
    m = prev.notna()
    ct = pd.crosstab(prev[m].astype(int), s.loc[m, "flag"])
    out["p_flag_after_flag"] = ct.loc[1, 1] / ct.loc[1].sum() if 1 in ct.index else np.nan
    out["p_flag_after_normal"] = ct.loc[0, 1] / ct.loc[0].sum()
    out["repeat_p"] = stats.fisher_exact(ct.values)[1] if ct.shape == (2, 2) else np.nan
    out["daily"] = daily
    return out


# ---------------------------------------------------------------- figures
def figures(df, tests, clf, clus, figs):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    lr = np.log(df["ratio"])
    axes[0].hist(lr, bins=200, color="#064A56")
    axes[0].set_yscale("log")
    for v in (config.OUTLIER_LOW, config.OUTLIER_HIGH):
        axes[0].axvline(np.log(v), color="#D9822B", ls="--")
    axes[0].set_title("log(rate per mile / reference), all rows (log count)")
    flagged = df[df["flag"] == 1]
    axes[1].hist([flagged.loc[flagged.high == 1, "ratio"], 1 / flagged.loc[flagged.low == 1, "ratio"]],
                 bins=30, label=["high spikes: ratio", "low dips: 1 / ratio"], color=["#064A56", "#D9822B"])
    axes[1].legend(); axes[1].set_title("multiplier size, high vs low")
    d = clus["daily"]
    r = (d["k"] / d["n"]).rolling(7, center=True).mean()
    axes[2].plot(d.index, d["k"] / d["n"], color="#BBBBBB", lw=0.7, label="daily")
    axes[2].plot(d.index, r, color="#064A56", label="7-day mean")
    axes[2].axhline(df["flag"].mean(), color="#D9822B", ls="--", label="overall")
    axes[2].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    axes[2].legend(); axes[2].set_title("share of flagged loads over time")
    fig.tight_layout(); fig.savefig(figs / "09_outlier_distribution.png", dpi=130); plt.close(fig)

    show = ["equipment", "distance_band", "lane_size", "month", "dayofweek", "holiday_window",
            "mi_q", "mi_change_q", "weight_q", "weight_was_negative", "quote_q", "pickup_region"]
    fig, axes = plt.subplots(3, 5, figsize=(20, 11))
    overall = df["flag"].mean()
    for ax, col in zip(axes.flat, show):
        t = group_rates(df, col)
        x = np.arange(len(t))
        ax.errorbar(x, t["rate"] * 100, yerr=[((t["rate"] - t["ci_low"]) * 100).clip(lower=0), ((t["ci_high"] - t["rate"]) * 100).clip(lower=0)],
                    fmt="o", color="#064A56", capsize=3)
        ax.axhline(overall * 100, color="#D9822B", ls="--")
        ax.set_xticks(x); ax.set_xticklabels(t.index.astype(str), rotation=45, fontsize=8)
        ax.set_title(f"{CATEGORICAL[col]}\nHolm p = {tests.loc[CATEGORICAL[col], 'chi2_p_holm']:.2g}", fontsize=9)
        ax.set_ylabel("% flagged")
    for ax, (name, r) in zip(axes.flat[len(show):], clf.items()):
        ax.bar([str(k) for k in r["folds"]], r["folds"].values(), color="#064A56")
        ax.axhline(0.5, color="#D9822B", ls="--")
        ax.set_ylim(0.3, 0.8); ax.set_xlabel("test month")
        ax.set_title(f"{name}: AUC per time fold\npooled {r['pooled_auc']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]", fontsize=9)
    for ax in axes.flat[len(show) + len(clf):]:
        ax.axis("off")
    fig.suptitle("Outlier rate by factor (95% Wilson intervals; dashed = overall rate)")
    fig.tight_layout(); fig.savefig(figs / "10_outlier_rate_by_factor.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- report
def _md(df, fmts):
    cols = list(df.columns)
    lines = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join(fmts.get(c, "{}").format(row[c]) if row[c] == row[c] else "-"
                                                for c in cols) + " |")
    return "\n".join(lines)


def run(train: pd.DataFrame) -> dict:
    df = outlier_frame(train)
    tests = factor_tests(df)
    num = numeric_tests(df)
    clf = classifier_auc(df, "flag")
    clf_dir = classifier_auc(df, "high", rows_mask=df["flag"] == 1)
    clus = clustering(df, train, reps=200)

    sig_factors = list(tests.index[tests["chi2_p_holm"] < ALPHA]) + list(num.index[num["mw_p_holm"] < ALPHA])
    if clus.get("lane_sim_p", 1) < ALPHA:
        sig_factors.append("lane clustering")
    auc_ok = any(r["ci_low"] > AUC_THRESHOLD for r in clf.values())
    explained = bool(sig_factors) and auc_ok
    verdict = {"explained": explained, "significant_factors": sig_factors, "classifier_predicts": auc_ok}

    figures(df, tests, clf, clus, config.FIGURES)
    art = config.ARTIFACTS
    tests.to_csv(art / "outlier_factor_tests.csv"); num.to_csv(art / "outlier_numeric_tests.csv")
    df[df["flag"] == 1][["load_id", "date", "pickup", "delivery", "equipment", "distance", "posted_rate",
                         "ratio", "direction", "reference"]].to_csv(art / "flagged_rows.csv", index=False)

    n, k = len(df), int(df["flag"].sum())
    hi, lo = int(df["high"].sum()), int(df["low"].sum())
    fl = df[df["flag"] == 1]
    lr = np.log(df["ratio"])
    gap_lo = int(((df["ratio"] >= 0.5) & (df["ratio"] < 0.8)).sum())
    gap_hi = int(((df["ratio"] > 1.25) & (df["ratio"] <= 2.0)).sum())
    q = [0, .1, .25, .5, .75, .9, 1]
    mult = pd.DataFrame({"high spikes (ratio)": fl.loc[fl.high == 1, "ratio"].quantile(q).values,
                         "low dips (ratio)": fl.loc[fl.low == 1, "ratio"].quantile(q).values,
                         "low dips (1 / ratio)": (1 / fl.loc[fl.low == 1, "ratio"]).quantile(q).values},
                        index=["min", "p10", "p25", "median", "p75", "p90", "max"])
    fmt_p = "{:.3g}"
    tt = tests[["groups", "min_rate", "max_rate", "groups_ci_excl_overall", "cramers_v", "chi2_p", "chi2_p_holm",
                "dir3_p_holm", "high_vs_low_p_holm", "size_high_p_holm", "size_low_p_holm"]]
    nt = num[["median_flagged", "median_normal", "auc_single", "mw_p", "mw_p_holm", "high_vs_low_p_holm",
              "spearman_size_p_holm"]]
    cl_rows = pd.DataFrame({k2: {"pooled AUC": v["pooled_auc"], "95% CI low": v["ci_low"], "95% CI high": v["ci_high"],
                                 "mean fold AUC": v["mean_fold_auc"], "positives": v["n_pos"], "rows": v["n"]}
                            for k2, v in clf.items()}).T
    cd_rows = pd.DataFrame({k2: {"pooled AUC": v["pooled_auc"], "95% CI low": v["ci_low"], "95% CI high": v["ci_high"],
                                 "mean fold AUC": v["mean_fold_auc"], "high spikes": v["n_pos"], "flagged rows": v["n"]}
                            for k2, v in clf_dir.items()}).T

    if explained:
        conclusion = (f"**Something explains the spikes**: {', '.join(sig_factors)} (Holm p < {ALPHA}) and a "
                      f"classifier beats AUC {AUC_THRESHOLD}. These rows should NOT be blindly dropped - add the "
                      "factor as a feature and compare drop-all / drop-unexplained / keep-all.")
    else:
        conclusion = ("**No factor explains the spikes - they behave like random label noise.** "
                      + (f"Factors significant after correction: {', '.join(sig_factors)}, but no classifier "
                         f"reaches AUC {AUC_THRESHOLD}, so they cannot be predicted from the inputs. "
                         if sig_factors else
                         f"No factor is significant after Holm correction (p < {ALPHA}) and no classifier's "
                         f"AUC interval clears {AUC_THRESHOLD}. ")
                      + "Flagged rows stay excluded from training; the feature study in `reports/summary.md` shows the "
                        "keep-all vs drop-all comparison.")
    text = [
        "# Outlier diagnostics: are posted_rate spikes random?" + (""),
        "", "Only `train_test.csv` has labels (`validation.csv` has no `posted_rate`), so this uses the training file.",
        f"Rule (as in `src/data.py`): rate per mile < {config.OUTLIER_LOW}x or > {config.OUTLIER_HIGH}x the "
        f"lane+equipment median (equipment + distance-band median when the lane has < {config.MIN_LANE_ROWS} loads).",
        "", "## Verdict", "", conclusion, "",
        "## 1. How many, which way, how big", "",
        f"- {k:,} of {n:,} loads flagged ({k / n:.2%}): {hi} high spikes, {lo} low dips "
        f"(binomial test vs 50/50: p = {stats.binomtest(hi, k).pvalue:.3g}).",
        f"- The ratio distribution has empty gaps: {gap_lo} loads between 0.5x and 0.8x, {gap_hi} between 1.25x and 2.0x; "
        f"{int(((df['ratio'] >= 0.8) & (df['ratio'] <= 1.25)).sum()):,} loads sit within 0.8x-1.25x. Real market moves "
        "would fill those gaps continuously; a separate cluster points to multiplied (corrupted) prices.",
        f"- Std of log ratio for unflagged loads: {lr[df['flag'] == 0].std():.3f}.", "",
        _md(mult, {c: "{:.3f}" for c in mult.columns}), "",
        "## 2. Outlier rate by factor", "",
        "`min_rate` / `max_rate` = lowest / highest flagged share among groups with >= 100 loads; "
        "`groups_ci_excl_overall` = groups whose 95% Wilson interval excludes the overall rate "
        f"({k / n:.2%}). p-values: chi-square of flag vs group; `dir3` = normal/high/low vs group; "
        "`high_vs_low` = direction within flagged rows; `size_*` = Kruskal-Wallis of |log ratio| within "
        "high / low rows. `_holm` = Holm-corrected over every test in this table.", "",
        _md(tt, {"groups": "{:.0f}", "min_rate": "{:.2%}", "max_rate": "{:.2%}", "groups_ci_excl_overall": "{:.0f}",
                 "cramers_v": "{:.3f}", **{c: fmt_p for c in tt.columns if c.endswith("_p") or "holm" in c}}), "",
        "## 3. Numeric columns, flagged vs normal", "",
        "`auc_single` = Mann-Whitney AUC of that one column (0.5 = no separation).", "",
        _md(nt, {"median_flagged": "{:.4g}", "median_normal": "{:.4g}", "auc_single": "{:.3f}",
                 **{c: fmt_p for c in nt.columns if "_p" in c}}), "",
        "## 4. Can a model predict the flag? (time-based CV: train on months < M, test on M)", "",
        f"Features: {', '.join(CLF_NUM + CLF_CAT)}. AUC 0.5 = random; 95% interval from 1,000 bootstrap resamples "
        "of the pooled out-of-fold predictions.", "",
        _md(cl_rows, {"positives": "{:.0f}", "rows": "{:.0f}", **{c: "{:.3f}" for c in ["pooled AUC", "95% CI low", "95% CI high", "mean fold AUC"]}}),
        "", "Per test month: " + "; ".join(f"{nm}: " + ", ".join(f"{m}={a:.3f}" for m, a in r["folds"].items())
                                           for nm, r in clf.items()), "",
        "High spike vs low dip, among flagged rows only:", "",
        _md(cd_rows, {"high spikes": "{:.0f}", "flagged rows": "{:.0f}", **{c: "{:.3f}" for c in ["pooled AUC", "95% CI low", "95% CI high", "mean fold AUC"]}}),
        "", "## 5. Do flagged rows cluster?", "",
        f"- **In time:** dispersion of daily flagged counts vs a constant rate = {clus['daily_dispersion']:.2f} "
        f"(1 = random; p = {clus['daily_dispersion_p']:.3g}, {clus['days']} days). Lag-1 autocorrelation of the "
        f"daily share = {clus['daily_lag1_autocorr']:.3f} (p = {clus['lag1_p']:.2g}), lag-7 = {clus['daily_lag7_autocorr']:.3f}. "
        f"Busiest day: {clus['max_daily_count']} flags.",
        f"- **On lanes:** {clus['lanes_tested']} lane+equipment groups with >= 10 loads; dispersion = "
        f"{clus['lane_dispersion']:.2f} (vs a binomial: p = {clus['lane_dispersion_p']:.3g}). The flag rule itself adds "
        f"some lane dispersion, so the fair null is a simulation ({clus['lane_sim_reps']} runs, random corruption at the "
        f"same rate and multipliers, same rule): median {clus.get('lane_sim_median', float('nan')):.2f}, "
        f"95th percentile {clus.get('lane_sim_p95', float('nan')):.2f}, simulation p = {clus.get('lane_sim_p', float('nan')):.3f}. Lanes with significantly more flags "
        f"than chance after Holm correction: {clus['lanes_significant']}.",
        f"- **Repeats:** P(flag | previous load on the same lane was flagged) = {clus['p_flag_after_flag']:.2%} vs "
        f"{clus['p_flag_after_normal']:.2%} after a normal load (Fisher p = {clus['repeat_p']:.3g}).", "",
        "Most extreme lanes:", "",
        _md(clus["top_lanes"].rename(columns={"n": "loads", "k": "flagged"}),
            {"loads": "{:.0f}", "flagged": "{:.0f}", "p": fmt_p, "p_holm": fmt_p}), "",
        "`lane_prior_flag_rate` (share of earlier loads on the same lane that were flagged) and "
        "`lane_prev_flagged` are in the classifiers above, so lane history is also tested out of time.", "",
        "## Decision rule", "",
        f"A factor (or lane clustering vs the simulation) counts as explaining spikes only if its Holm-adjusted p < {ALPHA} **and** a classifier's AUC "
        f"95% interval lies above {AUC_THRESHOLD}. Figures: `figures/09_outlier_distribution.png`, "
        "`figures/10_outlier_rate_by_factor.png`. Tables: `artifacts/outlier_*_tests.csv`, `artifacts/flagged_rows.csv`.", "",
    ]
    (config.REPORTS / "outlier_diagnostics.md").write_text("\n".join(text), encoding="utf-8")
    return verdict
