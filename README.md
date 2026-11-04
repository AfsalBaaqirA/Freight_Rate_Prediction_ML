# Freight Rate Prediction ML


This repo predicts `posted_rate` (USD) for truckload loads. One command runs the whole pipeline:

```
ETL / cleaning -> EDA (incl. outlier diagnostics) -> model comparison with time-based CV
-> auto-selection -> final fit -> predictions + December chart -> score.py -> reports/summary.md
```

The run writes `reports/summary.md` (the full write-up) and `reports/outlier_diagnostics.md`.
This README gives a short version of both.

## Setup and run

Python 3.10-3.13.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Or `bash run.sh` (macOS/Linux), which creates `.venv`, installs the requirements and runs `main.py`.

**Runtime:** about 9.5 minutes end to end (including the venv install) when pinned to 4 cores on my test box (`taskset -c 0-3`). A slower 4-core laptop should still finish in well under 45 minutes.
LightGBM/CatBoost use half the logical CPUs by default; override with `N_JOBS=8 python main.py`.
Everything is seeded (`SEED` in `src/config.py`) and package versions are pinned.

## Outputs

| output | content |
|---|---|
| `validation_predictions.csv` | `load_id,predicted_rate` for the 12,000 rows of `data/validation.csv` |
| `data/december_chart_inputs.csv` | filled in place (as the task asks): `predicted_rate` for Dec 1-31. The pipeline reads only the 6 input columns, so re-running is safe |
| `scorer_results/` | output of the unmodified `score.py` on both files |
| `reports/summary.md` | the full write-up (every number comes from the run) |
| `reports/outlier_diagnostics.md` | tests of whether the `posted_rate` spikes depend on any column |
| `reports/figures/` | 01-06 EDA, 07 quote_signal leak, 08 price level vs market_index, 09-10 outliers, 11 CV results, 12 feature study, 13 December vs lane history |
| `artifacts/` | CSV tables behind the report and `best_model.json` |

## Repo layout

```
main.py                     the pipeline, in order (read this first)
run.sh                      venv + install + python main.py
score.py                    Spotter's validator, unchanged
requirements.txt            pinned versions
data/                       train_test.csv, validation.csv, validation_predictions_template.csv, december_chart_inputs.csv
src/config.py               paths, cleaning thresholds, feature groups, CV folds, model settings, tuning grid
src/data.py                 ETL: load, clean weight, national daily market_index, outlier flag, December inputs
src/eda.py                  EDA figures, quote_signal leak check, price level vs market_index
src/outlier_diagnostics.py  are the rate spikes random? factor tests, classifiers, clustering
src/features.py             target, row features, out-of-fold target encodings
src/geo.py                  haversine distance; unseen cities borrow from their 3 nearest known cities
src/hierarchy.py            hierarchical empirical-Bayes effects (region pair -> city -> lane -> lane+equipment)
src/models.py               candidate models and blends
src/cv.py                   folds, metrics, tuning, model comparison + selection, calibration, feature study
src/predict.py              final fit, submission files, December figure, runs score.py
src/report.py               reports/summary.md and figures 11-12
```

## 1. Outliers: which rate spikes I did not train on

**Rule** (`src/data.py`): a training load is flagged when its rate per mile is **< 0.6x or > 1.6x the median rate per mile of its lane + equipment**. When the lane has fewer than 3 loads, I use the equipment + distance-band median instead.

| | count |
|---|---|
| flagged loads | **677 of 48,000 (1.41%)** |
| high spikes (> 1.6x) | 340, ratio 2.08x-5.44x (median 3.34x) |
| low dips (< 0.6x) | 337, ratio 0.17x-0.47x (median 0.28x) |
| loads used for the final fit | 47,323 |

- The flagged rows are **dropped from training only** (inside CV, only from each fold's training part). Validation rows are never altered, removed or re-labelled.
- All 18 training loads priced below $150 are injected dips; the cheapest clean load is $184.55.
- As a backstop for anything the rule misses, both LightGBM and CatBoost use a **Huber loss** on log(rate per mile).
- Keeping the flagged rows in training makes clean MAE slightly worse (44.5 vs 44.0 over the 5 time folds).

## 2. Outlier diagnostics (summary of `reports/outlier_diagnostics.md`)

I tested whether the spikes are real market surges or random corruption.

- **Empty gaps:** no load sits between 0.5x and 0.8x of its reference, and only 1 between 1.25x and 2.0x; 47,322 loads sit within 0.8x-1.25x (std of log ratio for normal loads: 0.042). Real market moves would fill these gaps.
- **Direction is a coin flip:** 340 high vs 337 low (binomial p = 0.939).
- **No factor explains them:** after Holm correction, no factor reaches p < 0.01 (the smallest Holm-adjusted p-value in the report is 0.127). Factors tested: equipment, distance band, pickup/delivery city and region, region pair, lane size, month, day of week, day of month, holiday window and nearest holiday, market_index level/change/weekly deviation, weight (quintile, missing, negative, capped), quote_signal and load_id order. Flagged rates per group stay close to the overall 1.41% (for example equipment 1.35%-1.51%, month 1.26%-1.65%).
- **Not predictable:** classifiers trained out of time (train on months before M, test on M) reach pooled AUC 0.508 (logistic regression, 95% CI 0.482-0.534) and 0.513 (LightGBM, 95% CI 0.489-0.537). 0.5 means random guessing. Telling high spikes from low dips: AUC 0.535 / 0.515.
- **No clustering:**
  - In time, the dispersion of daily counts is 0.98 (1 = random, p = 0.569).
  - On lanes, the dispersion is 1.12, inside a simulation of random corruption (95th percentile 1.14, p = 0.075). After Holm correction, 0 lanes have more flags than chance.
  - A flag after a flagged load on the same lane: 2.50% vs 1.40% (Fisher p = 0.057, not significant).

**Conclusion:** the spikes behave like randomly injected multiplicative noise (prices multiplied by about 2-5.4x or 0.17-0.47x), not real surges. That is why I drop them from training rather than try to model them.

## 3. Data cleaning

| column | finding | handling |
|---|---|---|
| `weight` | 292 train / 145 validation values are negative (sign flips) | `abs()`, plus a `weight_was_negative` flag |
| `weight` | 300 train / 165 validation values are blank | filled with the training median of the same equipment type, plus a `weight_missing` flag |
| `weight` | the source data is already capped at 47,500 lb: 1,204 train rows (2.5%) and 301 validation rows sit exactly at the cap, none above, none below 5,000 lb | `weight_capped` flag. The clip to [5,000, 47,500] is only a safety guard and changes no rows |
| `market_index` | 374 train / 249 validation blanks; the index is one national value per day plus small row noise | blank -> that day's national value, plus a `market_index_missing` flag |


## 4. Features used

The target is `log(posted_rate / distance)`; predictions are converted back to dollars.

| feature | what it is | why |
|---|---|---|
| `distance`, `log_distance` | trip length | rate per mile falls with distance; removing distance raises clean MAE 44.0 -> 45.5 |
| `equipment_code` | Dry Van / Reefer / Flatbed | equipment premium; the strongest group (removing it: 106.9) |
| `weight_clean` + `weight_missing`, `weight_was_negative`, `weight_capped` | cleaned weight and 3 cleaning flags | heavier loads cost more; removing the group: 65.5 |
| `mi_row_dev`, `market_index_missing` | the load's `market_index` minus that day's national value | the only market signal whose relation to price holds every month (removing it: 44.5) |
| `dayofweek` | day of week | weekly price pattern (removing it: 46.7) |
| pickup/delivery lat-lon, `d_lat`, `d_lon`, pickup/delivery region | coarse location (4 longitude bands) | coordinates are noisy, so only coarse location is used; it helps mostly for unseen cities |
| `te_pickup`, `te_delivery`, `pickup_seen`, `delivery_seen` | smoothed out-of-fold target encodings of each city (residual of log rpm ~ log distance + equipment, shrunk with a pseudo-count of 20) | city price level. **Unseen cities** borrow the inverse-distance-weighted (1/d, haversine) encoding of their **k = 3 nearest known cities**; the `*_seen` flag tells the model it was borrowed |
| `te_lane`, `te_lane_equipment` | the same encoding for the lane and lane + equipment | lane level; left blank for a new lane (removing them: 44.6) |
| recency weights (sample weights, not a column) | each training row weighted 0.5^(age / 180 days) | handles the drifting price level better than pooling (44.0 vs 45.1); half-life 180 was tuned in CV |

The ablation numbers are clean MAE ($) over the 5 time folds, each group removed and the model refit. The shipped recipe scores 44.0. `hierarchical_shrinkage` (`src/hierarchy.py`) is a separate model, not a feature: partial pooling region pair -> city -> lane -> lane + equipment, blended with LightGBM.

## 5. Features left out

| feature | why I left it out | evidence (clean MAE, 5 time folds; shipped = 44.0) |
|---|---|---|
| `market_index` daily level (`market_level`) | I tested it, and its relation to price breaks after August. The monthly mean index goes 0.93 (Jan) -> 1.30 (May) -> 0.89 (Sep) and tracks prices until July. From August prices recover (Sep +0.4%, Oct +0.7%) while the index stays low, so a model learns "low index = cheap" and under-prices Sep-Oct | + market_level: 91.4 |
| index lags / leads / changes / ratios / equipment interaction | the same break; leads also use future values that are not known in production | lag_1 91.0, lag_7 84.5, lag_14 81.7, lead_1 92.1, lead_7 99.1, lead_14 85.4, change_7d 68.2, ratio_28d 61.2, x_equipment 92.6 |
| `quote_signal` | leaks the label in training with a month-dependent sign: its correlation with lane rate per mile is about +0.96 in Jan-Mar, Jun and Sep and about -0.96 in Apr, May, Jul and Oct. It is noise in Aug (-0.03) and in Nov-Dec (+0.00 / -0.02), the validation months | 48.0; it breaks the August fold and would be pure noise on validation |
| `month`, `dayofyear` (season) | trees cannot extrapolate past the last training date. In the old recipe season mostly compensated for the broken market-level features; recency weighting replaced it | 61.2 |
| linear time trend | keeps climbing or falling after the level has turned | 111.9 extrapolated, 94.7 held at the last day |
| trailing 30 / 60-day level | carries the latest level forward and over-predicts July | 64.2 / 60.9 |

So I did consider `market_index`. I keep its daily deviation (`mi_row_dev`) and drop its level.

## 6. Validation and results

- **Folds** (`src/cv.py`): rolling monthly folds for Jul, Aug, Sep and Oct (each trained on all earlier months) and a forward split (Jan-Aug -> Sep-Oct).
- **Cold start:** 3 more folds each hold out 8 random cities, because 8 validation cities (12.1% of validation rows) never appear in training.
- **selection_score** = mean of (all-row MAE, clean-row RMSE, unseen-city MAE), each divided by the lane-median baseline. The lowest score wins and is retrained on all clean Jan-Oct rows.

| model | selection_score | all MAE | all RMSE | clean MAE | clean RMSE | unseen-city MAE |
|---|---|---|---|---|---|---|
| **blend_lgbm_hierarchical** (selected) | **0.648** | 97.9 | 626.4 | 43.5 | 63.2 | 114.3 |
| lightgbm+recency | 0.651 | 98.3 | 626.1 | 43.9 | 63.4 | 115.2 |
| lightgbm_v1 (previous recipe) | 0.764 | 110.9 | 629.4 | 56.6 | 82.6 | 129.7 |
| baseline_lane_median | 1.000 | 138.8 | 637.2 | 84.9 | 123.2 | 157.9 |

Errors are in $. All 10 candidates are in `reports/summary.md`.

- All-row RMSE barely moves for any model: the injected spikes in the test months dominate it, and no model can predict them.
- Tuned LightGBM: `num_leaves=7`, `n_estimators=600`, recency half-life 180 days, averaged over 5 seeds.
- Calibration: I evaluated a global RMSE multiplier and a smearing correction in CV. Neither improves every metric, so `CALIBRATION = None`.
- Final validation predictions: 12,000 rows, min $195, median $2,041, max $6,888.

## 7. December forecast (Lexington -> Fort Wayne, 360 mi, Dry Van, 32,000 lb)

My forecast is **$813-$832 per load (mean $823)**. The 21 real 2025 loads on this lane have a median of $808 (range $758-$934).
The line is nearly flat because every input except the date is fixed; the small wiggles come from day of week.
The labelled data ends on 31 Oct, so I do not model or invent any holiday effect.

## Limitations

- With 10 months of data, the model cannot learn yearly seasonality or holidays. Nov-Dec level shifts of 1-2% are not predictable from the inputs.
- About 1.4% random label spikes dominate all-row RMSE.
- Error on unseen cities stays higher than on seen lanes (114.3 vs 97.9 MAE).
- `quote_signal` is unusable as given. If its sign convention per period were known, it would be a strong feature.
