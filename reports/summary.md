# Summary

Full run: 16 min on this machine (2 LightGBM threads). Everything below is produced by `python main.py`; detail tables are in `artifacts/`, figures in `reports/figures/`.

## 1. Data issues and how each is handled

| issue | evidence | handling |
|---|---|---|
| `posted_rate` spikes | 677 of 48,000 training rows (1.4%) are <0.6x or >1.6x their lane+equipment median (clean rows sit within about ±5%) | dropped from training (fold-train only inside CV); `outlier_diagnostics.md`: no column predicts them (random corruption). Keeping them: clean MAE 44.5 vs 44.0 |
| `quote_signal` leaks the label | corr with lane rate per mile by month: 01: +0.95, 02: +0.96, 03: +0.96, 04: -0.96, 05: -0.96, 06: +0.96, 07: -0.96, 08: -0.03, 09: +0.97, 10: -0.96, 11: +0.00, 12: -0.02 (Nov-Dec = validation). It is +rpm or -rpm depending on the month, noise in Aug and in validation | excluded. Adding it looks good on some folds and fails on others (August is noise, so rolling_08 breaks; clean MAE 48.0 vs 44.0) and would be pure noise on validation |
| `weight` | 465 missing, 437 negative (sign flips), 1,505 at the 47,500 lb cap (train + validation) | abs(), fill with the equipment median, clip, keep 3 flags |
| `market_index` | 623 missing; one national value per day + row noise | fill with the day's national value; only the row deviation `mi_row_dev` is a feature (section 4) |
| cities never seen in training | 8 cities (Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego), 12.1% of validation rows | k=3 nearest known cities (haversine, 1/d weights) for the city encodings, lane encodings blank; scored on 3 cold-start folds |
| holidays | the labelled data stops on 31 Oct, so no Thanksgiving/Christmas history exists | no holiday effect is modelled or invented; the December line is a normal-month forecast |

## 2. Model comparison (time-based CV)

Folds: rolling Jul, Aug, Sep, Oct (train on all earlier months) + forward Jan-Aug -> Sep-Oct (averaged = *all* and *clean*), and 3 cold-start folds with 8 random cities removed (*unseen_city*). Errors in $. selection_score = mean of all-row MAE, clean RMSE and unseen-city MAE, each divided by the lane-median baseline; lowest wins.

| model | selection_score | all_mae | all_rmse | clean_mae | clean_rmse | unseen_city_mae |
|---|---|---|---|---|---|---|
| blend_lgbm_hierarchical | 0.648 | 97.9 | 626.4 | 43.5 | 63.2 | 114.3 |
| lightgbm+recency | 0.651 | 98.3 | 626.1 | 43.9 | 63.4 | 115.2 |
| blend_lgbm_catboost | 0.651 | 98.2 | 626.0 | 43.8 | 63.4 | 115.3 |
| lightgbm | 0.659 | 99.4 | 627.2 | 45.0 | 65.1 | 115.9 |
| catboost+recency | 0.661 | 99.1 | 626.0 | 44.7 | 64.7 | 117.5 |
| hierarchical_shrinkage | 0.665 | 99.9 | 627.1 | 45.6 | 66.3 | 116.5 |
| linear_huber_gam | 0.669 | 100.2 | 627.0 | 45.8 | 66.4 | 117.9 |
| lightgbm_v1 | 0.764 | 110.9 | 629.4 | 56.6 | 82.6 | 129.7 |
| lightgbm+season | 0.776 | 115.0 | 628.8 | 60.9 | 87.1 | 125.1 |
| baseline_lane_median | 1.000 | 138.8 | 637.2 | 84.9 | 123.2 | 157.9 |

`lightgbm_v1` is the previous submission's recipe (market_index level features + season, 63 leaves). Old vs new on the same folds: selection score 0.764 -> 0.648, clean MAE 56.6 -> 43.5, all MAE 110.9 -> 97.9, unseen-city MAE 129.7 -> 114.3. All-row RMSE barely moves for any model: it is dominated by the injected spikes in the test months, which no model can predict.

**Tuning** (LightGBM, 5 time folds, single seed, score = mean of all MAE and clean RMSE vs baseline): 24 settings tried; best num_leaves=7, n_estimators=600, half-life=180 (score 0.6122; worst 0.6411). Small trees win: the signal is lane level + distance + equipment, which the encodings already carry. The final LightGBM candidates average 5 seeds. Full grid: `artifacts/tuning.csv`.

## 3. Time level

The price level drifts a few percent over the year (clean rows, % vs each lane's mean): 1: -4.1, 2: -3.0, 3: -0.4, 4: +0.1, 5: +1.9, 6: +3.6, 7: +1.5, 8: -0.8, 9: +0.4, 10: +0.7. Every option below is the tuned LightGBM refit on the 5 time folds (clean MAE in $, per fold and mean):

| variant | clean_mae | all_mae | all_rmse | forward | rolling_07 | rolling_08 | rolling_09 | rolling_10 | rolling_07 bias % |
|---|---|---|---|---|---|---|---|---|---|
| pooled level (no time feature, no weights) | 45.1 | 99.5 | 627.3 | 41.0 | 64.6 | 38.7 | 43.3 | 37.9 | -2.1 |
| recency weights, half-life 180 days | 44.0 | 98.3 | 626.1 | 38.8 | 55.4 | 47.7 | 42.4 | 35.4 | -1.5 |
| linear trend, extrapolated + trees on detrended target | 111.9 | 165.2 | 632.5 | 98.5 | 123.9 | 175.1 | 91.6 | 70.3 | 5.3 |
| linear trend, held at last training day + trees on detrended target | 94.7 | 148.2 | 629.4 | 77.0 | 96.8 | 154.8 | 82.5 | 62.3 | 4.1 |
| trailing 30-day recent level | 64.2 | 118.2 | 628.7 | 44.7 | 120.0 | 76.9 | 44.7 | 34.5 | 4.9 |
| trailing 60-day recent level | 60.9 | 115.0 | 627.0 | 44.3 | 119.4 | 54.8 | 50.8 | 35.4 | 4.9 |
| season (month + day of year) | 61.2 | 115.3 | 629.0 | 50.3 | 119.0 | 34.0 | 50.6 | 52.0 | 4.9 |

- *Recency weights* (each training row weighted 0.5^(age / half-life)) are better than pooling (44.0 vs 45.1). They only re-weight real history, so they cannot extrapolate a level that has already turned.
- *Linear trend + trees on the detrended target*: extrapolated, the line keeps climbing or falling after the level has turned (111.9); held at the last training day it still starts from the line's end point, which overshoots after the turn (94.7).
- *Season* (month, day of year): trees cannot extrapolate past the last training date, so for new months they repeat the level of the last months they saw (61.2).
- *Trailing 30/60-day level* (mean residual of the previous 30/60 days, computed with a 1-day lag; dates after the training end get the last known value, so no leakage) carries the most recent level forward (64.2 / 60.9).
- **rolling_07 weakness:** June was the peak (+3.6%) and July fell to +1.5%. Methods that carry the latest level forward over-predict July (bias column: season +4.9%, trailing 30-day +4.9%), pooling under-predicts it (-2.1%) and recency weighting sits between (-1.5%). Level changes of 1-2% from one month to the next are not predictable from the inputs, so the safest choice is the one with the lowest mean error over all folds, not the best single fold.

## 4. Market index

Every option below is added to the shipped recipe and refit (5 time folds):

| variant | clean_mae | all_mae | all_rmse |
|---|---|---|---|
| shipped recipe (recency weights, half-life 180 days) | 44.0 | 98.3 | 626.1 |
| + market_level | 91.4 | 145.1 | 640.7 |
| + mi_lag_1 | 91.0 | 144.7 | 640.4 |
| + mi_lag_7 | 84.5 | 138.3 | 638.5 |
| + mi_lag_14 | 81.7 | 135.5 | 637.7 |
| + mi_lead_1 | 92.1 | 145.9 | 641.0 |
| + mi_lead_7 | 99.1 | 152.7 | 642.4 |
| + mi_lead_14 | 85.4 | 139.2 | 638.5 |
| + mi_change_7d | 68.2 | 122.2 | 629.6 |
| + mi_ratio_28d | 61.2 | 115.4 | 629.7 |
| + mi_x_equipment | 92.6 | 146.3 | 641.5 |
| + quote_signal | 48.0 | 102.2 | 626.6 |

Why the old market features hurt: the monthly mean index goes from 0.93 in Jan to 1.30 in May and down to 0.89 in Sep. Until July it moves with the price level (figure 08). From August the index keeps falling while prices recover (Sep +0.4%, Oct +0.7%). A model trained on the months before learns 'low index = cheap' and under-prices Sep-Oct. Lags, leads, changes, ratios and the equipment interaction inherit the same break. Season looked helpful in the old ablation only because it repaired the damage done by the level features: market level alone gives 91.4, the old recipe (market level + season) 56.6 in the comparison, while season added to the new recipe gives 61.2 vs 44.0. Leads are also unsafe in principle: in production the future index is not known. What does hold every month is the *row-level deviation* from the day's national value (`mi_row_dev`): a load whose index is above the national value that day is priced slightly higher (removing it: see section 5). That is the only market feature kept.

## 5. Feature importance (refit ablation, not permutation)

Permutation importance was dropped. The old report had a contradiction: refitting without season raised forward-fold clean MAE by about $61, yet season's permutation importance was 0. Both numbers were right. In the forward fold every test date (Sep-Oct) lies after the last training date, so the trees send all of them down the same 'latest months' branch. Shuffling month or day of year among test rows changes nothing, so the importance is 0. Refitting without season, however, left the market-level features in charge, and those break in Sep-Oct (section 4). Season mattered only as a counterweight to bad features. Permutation cannot measure a time feature on future dates, so each group is removed and the model refit instead (5 time folds; higher error than the shipped recipe = the group helps):

| variant | clean_mae | all_mae | all_rmse |
|---|---|---|---|
| shipped recipe (recency weights, half-life 180 days) | 44.0 | 98.3 | 626.1 |
| - distance | 45.5 | 99.8 | 626.1 |
| - equipment | 106.9 | 160.1 | 639.1 |
| - weight | 65.5 | 119.7 | 630.2 |
| - mi_row_dev | 44.5 | 98.8 | 626.2 |
| - dayofweek | 46.7 | 101.1 | 627.0 |
| - geo | 44.3 | 98.7 | 626.2 |
| - city_te | 44.0 | 98.4 | 626.0 |
| - lane_te | 44.6 | 99.0 | 626.3 |
| keep flagged outlier rows in training | 44.5 | 98.9 | 626.3 |

A change near zero means the group is redundant given the others on seen lanes; city encodings and geo matter mainly for unseen cities, which these time folds do not contain.

## 6. RMSE-aware calibration

The model predicts the median of log(rate per mile), so in dollars it under-shoots the mean (and the injected spikes push the mean up further). Two global multipliers were fitted on each fold's training rows (outliers included) and scored on the test rows:

| calibration | all_mae | all_rmse | clean_mae | clean_rmse | unseen_city_mae | unseen_city_rmse | mean_factor |
|---|---|---|---|---|---|---|---|
| none | 97.9 | 626.4 | 43.5 | 63.2 | 114.3 | 726.0 | 1.000 |
| rmse_multiplier | 99.0 | 625.1 | 44.6 | 64.0 | 116.4 | 724.4 | 1.009 |
| smearing | 99.8 | 625.0 | 45.5 | 65.0 | 117.4 | 724.2 | 1.011 |

Decision: no method improves every metric; shipped `CALIBRATION = None` in `src/config.py` (options: "rmse_multiplier", "smearing"). Trade-off: a factor above 1 lowers all-row RMSE slightly, because the spikes add far more dollars upward than downward (multiplicative noise: a 3x spike adds 2x the rate, a 0.3x dip removes 0.7x), but it shifts every clean prediction up, so MAE and clean errors get worse.

## 7. Final model

**blend_lgbm_hierarchical** (selection score 0.648; baseline 1.000), retrained on 47,323 clean Jan-Oct rows (677 spike rows dropped), calibration factor 1.000. Validation predictions: 12,000 rows, min $195, median $2,041, max $6,888.

## 8. December chart (Lexington -> Fort Wayne, 360 mi, Dry Van, 32,000 lb)

Predicted $813-$832 (mean $823). The 21 real 2025 loads on this lane: median $808, range $758-$934. The line is nearly flat because inputs are fixed except the date. The small wiggles are day of week (highest on average: Thursday, lowest: Sunday). No holiday dip or spike is shown, because there is no holiday history to learn one from. Figure 13 overlays the lane history; `scorer_results/` has the scorer's chart.

## 9. Limitations

- 10 months of data: no yearly seasonality and no holidays can be learned; Nov-Dec level shifts of 1-2% are unpredictable.
- All-row RMSE is dominated by random label spikes (about 1.4% of rows, 0.2-5x); no model can reduce that part.
- Unseen cities rely on nearby cities' levels; their error stays well above that of seen lanes (unseen-city MAE 114.3 vs all-row MAE 97.9).
- `quote_signal` is unusable as given. If its sign convention per period were known, it would be a strong feature.
