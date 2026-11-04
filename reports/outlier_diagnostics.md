# Outlier diagnostics: are posted_rate spikes random?

Only `train_test.csv` has labels (`validation.csv` has no `posted_rate`), so this uses the training file.
Rule (as in `src/data.py`): rate per mile < 0.6x or > 1.6x the lane+equipment median (equipment + distance-band median when the lane has < 3 loads).

## Verdict

**No factor explains the spikes - they behave like random label noise.** No factor is significant after Holm correction (p < 0.01) and no classifier's AUC interval clears 0.55. Flagged rows stay excluded from training; the feature study in `reports/summary.md` shows the keep-all vs drop-all comparison.

## 1. How many, which way, how big

- 677 of 48,000 loads flagged (1.41%): 340 high spikes, 337 low dips (binomial test vs 50/50: p = 0.939).
- The ratio distribution has empty gaps: 0 loads between 0.5x and 0.8x, 1 between 1.25x and 2.0x; 47,322 loads sit within 0.8x-1.25x. Real market moves would fill those gaps continuously; a separate cluster points to multiplied (corrupted) prices.
- Std of log ratio for unflagged loads: 0.042.

| | high spikes (ratio) | low dips (ratio) | low dips (1 / ratio) |
|---|---|---|---|
| min | 2.077 | 0.168 | 2.120 |
| p10 | 2.443 | 0.198 | 2.526 |
| p25 | 2.864 | 0.222 | 2.841 |
| median | 3.344 | 0.283 | 3.529 |
| p75 | 4.226 | 0.352 | 4.507 |
| p90 | 4.675 | 0.396 | 5.060 |
| max | 5.441 | 0.472 | 5.954 |

## 2. Outlier rate by factor

`min_rate` / `max_rate` = lowest / highest flagged share among groups with >= 100 loads; `groups_ci_excl_overall` = groups whose 95% Wilson interval excludes the overall rate (1.41%). p-values: chi-square of flag vs group; `dir3` = normal/high/low vs group; `high_vs_low` = direction within flagged rows; `size_*` = Kruskal-Wallis of |log ratio| within high / low rows. `_holm` = Holm-corrected over every test in this table.

| | groups | min_rate | max_rate | groups_ci_excl_overall | cramers_v | chi2_p | chi2_p_holm | dir3_p_holm | high_vs_low_p_holm | size_high_p_holm | size_low_p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| equipment | 3 | 1.35% | 1.51% | 0 | 0.006 | 0.443 | 1 | 1 | 1 | 1 | 1 |
| distance band | 7 | 1.21% | 2.01% | 0 | 0.014 | 0.163 | 1 | 1 | 1 | 1 | 1 |
| pickup city | 64 | 0.00% | 3.26% | 4 | 0.034 | 0.76 | 1 | 1 | 1 | 1 | 1 |
| delivery city | 64 | 0.43% | 2.66% | 4 | 0.041 | 0.0507 | 1 | 1 | 1 | 1 | 1 |
| pickup region | 4 | 1.08% | 1.46% | 0 | 0.010 | 0.205 | 1 | 1 | 1 | 1 | 1 |
| delivery region | 4 | 1.12% | 1.49% | 0 | 0.010 | 0.168 | 1 | 1 | 1 | 1 | 1 |
| region pair | 16 | 0.76% | 1.63% | 0 | 0.016 | 0.626 | 1 | 1 | 1 | 1 | 1 |
| lane size (loads) | 3 | 1.34% | 1.45% | 0 | 0.004 | 0.714 | 1 | 1 | 1 | 1 | 1 |
| reference used | 2 | 1.34% | 1.42% | 0 | 0.002 | 0.643 | 1 | 1 | 1 | 1 | 1 |
| month | 10 | 1.26% | 1.65% | 0 | 0.011 | 0.768 | 1 | 1 | 1 | 1 | 1 |
| day of week | 7 | 1.23% | 1.59% | 0 | 0.010 | 0.6 | 1 | 1 | 1 | 1 | 1 |
| day of month | 31 | 0.99% | 2.35% | 1 | 0.023 | 0.724 | 1 | 1 | 1 | 1 | 1 |
| within 3 days of a holiday | 2 | 1.41% | 1.41% | 0 | 0.000 | 1 | 1 | 1 | 1 | 1 | 1 |
| nearest holiday (in window) | 9 | 0.95% | 1.82% | 0 | 0.010 | 0.776 | 1 | 1 | 1 | 1 | 1 |
| market_index quintile | 5 | 1.24% | 1.65% | 0 | 0.011 | 0.186 | 1 | 1 | 1 | 1 | 1 |
| market_index daily-change quintile | 5 | 1.27% | 1.65% | 0 | 0.011 | 0.199 | 1 | 1 | 1 | 1 | 1 |
| market_index weekly-deviation quintile | 5 | 1.30% | 1.50% | 0 | 0.006 | 0.774 | 1 | 1 | 1 | 0.892 | 1 |
| weight quintile | 5 | 1.25% | 1.55% | 0 | 0.011 | 0.255 | 1 | 1 | 1 | 1 | 0.559 |
| weight missing | 2 | 1.00% | 1.41% | 0 | 0.002 | 0.719 | 1 | 1 | 1 | - | - |
| weight negative | 2 | 1.40% | 3.08% | 1 | 0.010 | 0.0292 | 1 | 1 | 1 | 1 | - |
| weight capped | 2 | 1.33% | 1.41% | 0 | 0.001 | 0.905 | 1 | 1 | 1 | 1 | 1 |
| market_index missing | 2 | 1.41% | 1.60% | 0 | 0.000 | 0.921 | 1 | 1 | 1 | - | - |
| quote_signal quintile | 5 | 1.30% | 1.58% | 0 | 0.008 | 0.489 | 1 | 1 | 1 | 1 | 1 |
| load_id order quintile | 5 | 1.29% | 1.51% | 0 | 0.006 | 0.738 | 1 | 1 | 1 | 1 | 1 |

## 3. Numeric columns, flagged vs normal

`auc_single` = Mann-Whitney AUC of that one column (0.5 = no separation).

| | median_flagged | median_normal | auc_single | mw_p | mw_p_holm | high_vs_low_p_holm | spearman_size_p_holm |
|---|---|---|---|---|---|---|---|
| distance | 874.4 | 954.1 | 0.474 | 0.0193 | 0.309 | 1 | 1 |
| weight_clean | 3.097e+04 | 3.149e+04 | 0.484 | 0.163 | 1 | 1 | 1 |
| market_index_clean | 1.048 | 1.056 | 0.494 | 0.57 | 1 | 1 | 0.127 |
| mi_day | 1.043 | 1.052 | 0.492 | 0.468 | 1 | 1 | 0.153 |
| mi_change | 0.01482 | 0.01176 | 0.517 | 0.124 | 1 | 0.433 | 1 |
| mi_week_dev | -0.008818 | -0.009323 | 0.515 | 0.193 | 1 | 1 | 1 |
| quote_signal | 2.04 | 2.056 | 0.493 | 0.557 | 1 | 1 | 1 |
| pickup_lat | 35.3 | 35.29 | 0.509 | 0.408 | 1 | 1 | 1 |
| pickup_lon | -87.53 | -88.09 | 0.509 | 0.406 | 1 | 1 | 1 |
| delivery_lat | 35.29 | 35.29 | 0.500 | 0.989 | 1 | 1 | 1 |
| delivery_lon | -87.53 | -87.53 | 0.512 | 0.276 | 1 | 1 | 1 |
| lane_n | 6 | 6 | 0.500 | 0.967 | 1 | 1 | 1 |
| days_to_holiday | 10 | 11 | 0.484 | 0.144 | 1 | 1 | 1 |
| dayofmonth | 16 | 16 | 0.510 | 0.389 | 1 | 1 | 1 |
| load_seq | 2.41e+04 | 2.4e+04 | 0.502 | 0.845 | 1 | 1 | 1 |
| lane_prior_flag_rate | 0.01175 | 0.01175 | 0.505 | 0.662 | 1 | 1 | 1 |

## 4. Can a model predict the flag? (time-based CV: train on months < M, test on M)

Features: distance, weight_clean, market_index_clean, mi_day, mi_change, mi_week_dev, quote_signal, pickup_lat, pickup_lon, delivery_lat, delivery_lon, lane_n, days_to_holiday, dayofmonth, load_seq, lane_prior_flag_rate, lane_prev_flagged, weight_missing, weight_was_negative, weight_capped, market_index_missing, month, dayofweek, holiday_window, distance_band, equipment, pickup, delivery, region_pair. AUC 0.5 = random; 95% interval from 1,000 bootstrap resamples of the pooled out-of-fold predictions.

| | pooled AUC | 95% CI low | 95% CI high | mean fold AUC | positives | rows |
|---|---|---|---|---|---|---|
| logistic regression | 0.508 | 0.482 | 0.534 | 0.506 | 468 | 33709 |
| LightGBM classifier | 0.513 | 0.489 | 0.537 | 0.505 | 468 | 33709 |

Per test month: logistic regression: 4=0.483, 5=0.569, 6=0.451, 7=0.480, 8=0.552, 9=0.457, 10=0.547; LightGBM classifier: 4=0.554, 5=0.559, 6=0.482, 7=0.492, 8=0.524, 9=0.464, 10=0.464

High spike vs low dip, among flagged rows only:

| | pooled AUC | 95% CI low | 95% CI high | mean fold AUC | high spikes | flagged rows |
|---|---|---|---|---|---|---|
| logistic regression | 0.535 | 0.482 | 0.584 | 0.543 | 236 | 468 |
| LightGBM classifier | 0.515 | 0.462 | 0.569 | 0.519 | 236 | 468 |

## 5. Do flagged rows cluster?

- **In time:** dispersion of daily flagged counts vs a constant rate = 0.98 (1 = random; p = 0.569, 304 days). Lag-1 autocorrelation of the daily share = 0.092 (p = 0.11), lag-7 = -0.043. Busiest day: 7 flags.
- **On lanes:** 1033 lane+equipment groups with >= 10 loads; dispersion = 1.12 (vs a binomial: p = 0.00406). The flag rule itself adds some lane dispersion, so the fair null is a simulation (200 runs, random corruption at the same rate and multipliers, same rule): median 1.00, 95th percentile 1.14, simulation p = 0.075. Lanes with significantly more flags than chance after Holm correction: 0.
- **Repeats:** P(flag | previous load on the same lane was flagged) = 2.50% vs 1.40% after a normal load (Fisher p = 0.0572).

Most extreme lanes:

| | loads | flagged | p | p_holm |
|---|---|---|---|---|
| Greensboro>Baton Rouge|Dry Van | 11 | 3 | 0.000425 | 0.439 |
| Los Angeles>Mobile|Dry Van | 13 | 3 | 0.000722 | 0.745 |
| Tucson>Phoenix|Dry Van | 13 | 3 | 0.000722 | 0.745 |
| Oklahoma City>Mobile|Dry Van | 22 | 3 | 0.00353 | 1 |
| Chattanooga>Grand Rapids|Dry Van | 10 | 2 | 0.0083 | 1 |

`lane_prior_flag_rate` (share of earlier loads on the same lane that were flagged) and `lane_prev_flagged` are in the classifiers above, so lane history is also tested out of time.

## Decision rule

A factor (or lane clustering vs the simulation) counts as explaining spikes only if its Holm-adjusted p < 0.01 **and** a classifier's AUC 95% interval lies above 0.55. Figures: `figures/09_outlier_distribution.png`, `figures/10_outlier_rate_by_factor.png`. Tables: `artifacts/outlier_*_tests.csv`, `artifacts/flagged_rows.csv`.
