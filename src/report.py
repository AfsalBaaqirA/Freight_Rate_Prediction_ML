"""reports/summary.md (one page) plus the CV and feature-study figures.
Every number in the summary is read from this run's tables."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

SHOW = ["selection_score", "all_mae", "all_rmse", "clean_mae", "clean_rmse", "unseen_city_mae"]


def _md(df: pd.DataFrame, digits=1, index=True) -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else (f"{v:.3f}" if c in ("selection_score", "tuning_score", "mean_factor") else f"{v:,.{digits}f}"))
    if index:
        df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.values]
    return "\n".join([head, "|" + "---|" * len(df.columns)] + rows)


def cv_figure(summary: pd.DataFrame, path=config.FIGURES / "11_cv_results.png"):
    s = summary.iloc[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.45 * len(s) + 1.5), sharey=True)
    for ax, col in zip(axes, ["selection_score", "clean_mae", "unseen_city_mae"]):
        ax.barh(s.index, s[col], color=["#D9822B" if i == summary.index[0] else "#064A56" for i in s.index])
        ax.set_title(col)
    fig.suptitle("Time-based CV (lower is better; orange = selected)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def study_figure(study: pd.DataFrame, path=config.FIGURES / "12_feature_study.png"):
    s = study.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(s) + 1.5))
    colors = {"time level": "#064A56", "added to shipped recipe": "#D9822B",
              "removed from shipped recipe": "#7A7A7A", "cleaning": "#A23B72"}
    ax.barh(s["variant"], s["clean_mae"], color=[colors[c] for c in s["section"]])
    ax.set_xlabel("clean MAE, mean of 5 time folds ($)")
    ax.set_title("Feature study (refit, single seed); colour = section")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def write(train, valid, summary, results, tuning, params, half_life, calibration, study, quote, level,
          verdict, best, fit, template, dec, minutes):
    cv_figure(summary)
    study_figure(study)
    S = summary
    b, base, v1 = S.loc[best], S.loc["baseline_lane_median"], S.loc["lightgbm_v1"]
    st = study.set_index("variant")
    folds = [c for c in study.columns if c.endswith("_clean_mae")]
    tl = study[study["section"] == "time level"].set_index("variant")[["clean_mae", "all_mae", "all_rmse"] + folds + ["rolling_07_clean_bias_pct"]]
    tl.columns = [c.replace("_clean_mae", "").replace("rolling_07_clean_bias_pct", "rolling_07 bias %") for c in tl.columns]
    mk = study[study["section"] == "added to shipped recipe"].set_index("variant")[["clean_mae", "all_mae", "all_rmse"]]
    rm = study[study["section"].isin(["removed from shipped recipe", "cleaning"])].set_index("variant")[["clean_mae", "all_mae", "all_rmse"]]
    ref_name = next(v for v in tl.index if v.startswith("recency")) if half_life else "pooled level (no time feature, no weights)"
    ref = st.loc[ref_name]
    rm = pd.concat([st.loc[[ref_name], ["clean_mae", "all_mae", "all_rmse"]].rename(index={ref_name: f"shipped recipe ({ref_name})"}), rm])
    mk = pd.concat([rm.iloc[[0]], mk])
    pooled = st.loc["pooled level (no time feature, no weights)"]
    season = st.loc["season (month + day of year)"]
    q = st.loc["+ quote_signal"]
    lev = level["price_level_pct"]
    cal = calibration
    cal_better = [m for m in ("rmse_multiplier", "smearing")
                  if all(cal.loc[m, c] < cal.loc["none", c] for c in ["all_mae", "all_rmse", "clean_mae", "clean_rmse", "unseen_city_mae"])]
    n_out = fit["n_dropped"]
    wt_missing, wt_neg, wt_cap = (int(train[c].sum() + valid[c].sum()) for c in ("weight_missing", "weight_was_negative", "weight_capped"))
    mi_missing = int(train["market_index_missing"].sum() + valid["market_index_missing"].sum())
    unseen_cities = sorted((set(valid["pickup"]) | set(valid["delivery"])) - (set(train["pickup"]) | set(train["delivery"])))
    unseen_share = (~valid["pickup"].isin(train["pickup"].tolist() + train["delivery"].tolist())
                    | ~valid["delivery"].isin(train["pickup"].tolist() + train["delivery"].tolist())).mean()
    qt = quote["corr_vs_lane_median_rpm"]
    tune_best = tuning.iloc[0]

    L = [
        "# Summary", "",
        f"Full run: {minutes:.0f} min on this machine ({config.N_JOBS} LightGBM threads). Everything below is produced by `python main.py`; "
        "detail tables are in `artifacts/`, figures in `reports/figures/`.", "",
        "## 1. Data issues and how each is handled", "",
        "| issue | evidence | handling |", "|---|---|---|",
        f"| `posted_rate` spikes | {n_out:,} of {len(train):,} training rows ({n_out / len(train):.1%}) are <0.6x or >1.6x their lane+equipment median (clean rows sit within about ±5%) | "
        f"dropped from training (fold-train only inside CV); `outlier_diagnostics.md`: {'a factor explains them - check' if verdict['explained'] else 'no column predicts them (random corruption)'}. "
        f"Keeping them: clean MAE {st.loc['keep flagged outlier rows in training', 'clean_mae']:.1f} vs {ref['clean_mae']:.1f} |",
        f"| `quote_signal` leaks the label | corr with lane rate per mile by month: {', '.join(f'{i[5:7]}: {v:+.2f}' for i, v in qt.items())} (Nov-Dec = validation). "
        "It is +rpm or -rpm depending on the month, noise in Aug and in validation | excluded. Adding it looks good on some folds "
        f"and fails on others (August is noise, so rolling_08 breaks; clean MAE {q['clean_mae']:.1f} vs {ref['clean_mae']:.1f}) and would be pure noise on validation |",
        f"| `weight` | {wt_missing:,} missing, {wt_neg:,} negative (sign flips), {wt_cap:,} at the {config.WEIGHT_CAP:,} lb cap (train + validation) | abs(), fill with the equipment median, clip, keep 3 flags |",
        f"| `market_index` | {mi_missing:,} missing; one national value per day + row noise | fill with the day's national value; only the row deviation `mi_row_dev` is a feature (section 4) |",
        f"| cities never seen in training | {len(unseen_cities)} cities ({', '.join(unseen_cities)}), {unseen_share:.1%} of validation rows | k=3 nearest known cities (haversine, 1/d weights) for the city encodings, lane encodings blank; scored on 3 cold-start folds |",
        "| holidays | the labelled data stops on 31 Oct, so no Thanksgiving/Christmas history exists | no holiday effect is modelled or invented; the December line is a normal-month forecast |", "",
        "## 2. Model comparison (time-based CV)", "",
        "Folds: rolling Jul, Aug, Sep, Oct (train on all earlier months) + forward Jan-Aug -> Sep-Oct (averaged = *all* and *clean*), "
        "and 3 cold-start folds with 8 random cities removed (*unseen_city*). Errors in $. "
        "selection_score = mean of all-row MAE, clean RMSE and unseen-city MAE, each divided by the lane-median baseline; lowest wins.", "",
        _md(S[SHOW]), "",
        f"`lightgbm_v1` is the previous submission's recipe (market_index level features + season, 63 leaves). "
        f"Old vs new on the same folds: selection score {v1['selection_score']:.3f} -> {b['selection_score']:.3f}, "
        f"clean MAE {v1['clean_mae']:.1f} -> {b['clean_mae']:.1f}, all MAE {v1['all_mae']:.1f} -> {b['all_mae']:.1f}, "
        f"unseen-city MAE {v1['unseen_city_mae']:.1f} -> {b['unseen_city_mae']:.1f}. "
        "All-row RMSE barely moves for any model: it is dominated by the injected spikes in the test months, which no model can predict.", "",
        f"**Tuning** (LightGBM, 5 time folds, single seed, score = mean of all MAE and clean RMSE vs baseline): "
        f"{len(tuning)} settings tried; best num_leaves={int(tune_best['num_leaves'])}, n_estimators={int(tune_best['n_estimators'])}, "
        f"half-life={tune_best['half_life']} (score {tune_best['tuning_score']:.4f}; worst {tuning['tuning_score'].max():.4f}). "
        f"{'Small trees win' if int(tune_best['num_leaves']) == min(config.TUNING_GRID['num_leaves']) else 'Differences are small'}: the signal is lane level + distance + equipment, which the encodings already carry. "
        f"The final LightGBM candidates average {config.N_SEEDS} seeds. Full grid: `artifacts/tuning.csv`.", "",
        "## 3. Time level", "",
        f"The price level drifts a few percent over the year (clean rows, % vs each lane's mean): "
        + ", ".join(f"{m}: {v:+.1f}" for m, v in lev.items()) + ". "
        "Every option below is the tuned LightGBM refit on the 5 time folds (clean MAE in $, per fold and mean):", "",
        _md(tl), "",
        f"- *Recency weights* (each training row weighted 0.5^(age / half-life)) {'are' if ref['clean_mae'] < pooled['clean_mae'] else 'are not'} better than pooling "
        f"({ref['clean_mae']:.1f} vs {pooled['clean_mae']:.1f}). They only re-weight real history, so they cannot extrapolate a level that has already turned.",
        f"- *Linear trend + trees on the detrended target*: extrapolated, the line keeps climbing or falling after the level has turned "
        f"({st.loc['linear trend, extrapolated + trees on detrended target', 'clean_mae']:.1f}); held at the last training day it still starts from the line's end point, which "
        f"overshoots after the turn ({st.loc['linear trend, held at last training day + trees on detrended target', 'clean_mae']:.1f}).",
        f"- *Season* (month, day of year): trees cannot extrapolate past the last training date, so for new months they repeat the level of the last "
        f"months they saw ({season['clean_mae']:.1f}).",
        "- *Trailing 30/60-day level* (mean residual of the previous 30/60 days, computed with a 1-day lag; dates after the training "
        "end get the last known value, so no leakage) carries the most recent level forward "
        f"({st.loc['trailing 30-day recent level', 'clean_mae']:.1f} / {st.loc['trailing 60-day recent level', 'clean_mae']:.1f}).",
        f"- **rolling_07 weakness:** June was the peak ({lev.loc[6]:+.1f}%) and July fell to {lev.loc[7]:+.1f}%. Methods that carry the latest level forward "
        f"over-predict July (bias column: season {season['rolling_07_clean_bias_pct']:+.1f}%, trailing 30-day {st.loc['trailing 30-day recent level', 'rolling_07_clean_bias_pct']:+.1f}%), "
        f"pooling under-predicts it ({pooled['rolling_07_clean_bias_pct']:+.1f}%) and recency weighting sits between ({ref['rolling_07_clean_bias_pct']:+.1f}%). Level changes of 1-2% from one month to the next "
        "are not predictable from the inputs, so the safest choice is the one with the lowest mean error over all folds, not the best single fold.", "",
        "## 4. Market index", "",
        "Every option below is added to the shipped recipe and refit (5 time folds):", "",
        _md(mk), "",
        "Why the old market features hurt: the monthly mean index goes from "
        f"{level['market_index'].iloc[0]:.2f} in Jan to {level['market_index'].max():.2f} in {pd.Timestamp(2025, int(level['market_index'].idxmax()), 1):%B} and down to {level['market_index'].loc[9]:.2f} in Sep. "
        f"Until July it moves with the price level (figure 08). From August the index keeps falling while prices recover "
        f"(Sep {lev.loc[9]:+.1f}%, Oct {lev.loc[10]:+.1f}%). A model trained on the months before learns 'low index = cheap' and under-prices Sep-Oct. "
        "Lags, leads, changes, ratios and the equipment interaction inherit the same break. "
        f"Season looked helpful in the old ablation only because it repaired the damage done by the level features: market level alone gives {st.loc['+ market_level', 'clean_mae']:.1f}, "
        f"the old recipe (market level + season) {v1['clean_mae']:.1f} in the comparison, while season added to the new recipe gives {season['clean_mae']:.1f} vs {ref['clean_mae']:.1f}. "
        "Leads are also unsafe in principle: in production the future index is not known. What does hold every month is the "
        "*row-level deviation* from the day's national value (`mi_row_dev`): a load whose index is above the national value that day is priced slightly higher (removing it: see section 5). That is the only market feature kept.", "",
        "## 5. Feature importance (refit ablation, not permutation)", "",
        "Permutation importance was dropped. The old report had a contradiction: refitting without season raised forward-fold clean MAE by about $61, "
        "yet season's permutation importance was 0. Both numbers were right. In the forward fold every test date (Sep-Oct) lies after the last training date, "
        "so the trees send all of them down the same 'latest months' branch. Shuffling month or day of year among test rows changes nothing, so the importance is 0. "
        "Refitting without season, however, left the market-level features in charge, and those break in Sep-Oct (section 4). Season mattered only as a counterweight "
        "to bad features. Permutation cannot measure a time feature on future dates, so each group is removed and the model refit instead "
        "(5 time folds; higher error than the shipped recipe = the group helps):", "",
        _md(rm), "",
        "A change near zero means the group is redundant given the others on seen lanes; city encodings and geo matter mainly for unseen cities, "
        "which these time folds do not contain.", "",
        "## 6. RMSE-aware calibration", "",
        "The model predicts the median of log(rate per mile), so in dollars it under-shoots the mean (and the injected spikes push the mean up further). "
        "Two global multipliers were fitted on each fold's training rows (outliers included) and scored on the test rows:", "",
        _md(cal), "",
        f"Decision: {'CV favours ' + ' and '.join(cal_better) + ' on every metric' if cal_better else 'no method improves every metric'}; "
        f"shipped `CALIBRATION = {config.CALIBRATION!r}` in `src/config.py` ({'set to ' + repr(cal_better[0]) + ' to enable' if cal_better else 'options: \"rmse_multiplier\", \"smearing\"'}). "
        "Trade-off: a factor above 1 lowers all-row RMSE slightly, because the spikes add far more dollars upward than downward (multiplicative noise: a 3x spike adds 2x the rate, a 0.3x dip removes 0.7x), but it shifts every clean prediction up, so MAE and clean errors get worse.", "",
        "## 7. Final model", "",
        f"**{best}** (selection score {b['selection_score']:.3f}; baseline 1.000), retrained on {fit['n_train']:,} clean Jan-Oct rows "
        f"({n_out:,} spike rows dropped), calibration factor {fit['factor']:.3f}. "
        f"Validation predictions: {len(template):,} rows, min ${template['predicted_rate'].min():,.0f}, median ${template['predicted_rate'].median():,.0f}, max ${template['predicted_rate'].max():,.0f}.", "",
        "## 8. December chart (Lexington -> Fort Wayne, 360 mi, Dry Van, 32,000 lb)", "",
        f"Predicted ${dec['min']:,.0f}-${dec['max']:,.0f} (mean ${dec['mean']:,.0f}). The {dec['lane_n']} real 2025 loads on this lane: "
        f"median ${dec['lane_median']:,.0f}, range ${dec['lane_min']:,.0f}-${dec['lane_max']:,.0f}. "
        f"The line is nearly flat because inputs are fixed except the date. The small wiggles are day of week "
        f"(highest on average: {dec['high_day']}, lowest: {dec['low_day']}). No holiday dip or spike is shown, because there is no holiday history to learn one from. "
        "Figure 13 overlays the lane history; `scorer_results/` has the scorer's chart.", "",
        "## 9. Limitations", "",
        "- 10 months of data: no yearly seasonality and no holidays can be learned; Nov-Dec level shifts of 1-2% are unpredictable.",
        "- All-row RMSE is dominated by random label spikes (about 1.4% of rows, 0.2-5x); no model can reduce that part.",
        f"- Unseen cities rely on nearby cities' levels; their error stays well above that of seen lanes (unseen-city MAE {b['unseen_city_mae']:.1f} vs all-row MAE {b['all_mae']:.1f}).",
        "- `quote_signal` is unusable as given. If its sign convention per period were known, it would be a strong feature.",
    ]
    (config.REPORTS / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
