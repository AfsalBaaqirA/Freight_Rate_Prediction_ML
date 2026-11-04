"""Central configuration: paths, seeds, thresholds and model settings.

Everything tunable lives here so the rest of the code stays readable.
"""
import os
from pathlib import Path

SEED = 42

# ---- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train_test.csv"
VALID_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ARTIFACTS = ROOT / "artifacts"
SCORER_DIR = ROOT / "scorer_results"

# ---- cleaning ------------------------------------------------------------------
WEIGHT_CAP = 47_500        # cap weights above this to avoid extreme RMSE penalties
WEIGHT_FLOOR = 5_000       # floor weights below this to avoid extreme RMSE penalties
OUTLIER_LOW = 0.6
OUTLIER_HIGH = 1.6
MIN_LANE_ROWS = 3          # below this, fall back to an equipment + distance-band reference
MIN_PREDICTED_RATE = 150.0 # cap rates below this to avoid extreme RMSE penalties

# ---- encodings -------------------------------------------------------------------
TE_SMOOTHING = 20          # pseudo-count that shrinks small groups toward the prior
TE_FOLDS = 5               # out-of-fold encoding inside every training set
K_NEAREST = 3              # unseen cities borrow from the 3 nearest known cities (haversine, 1/d weights)
HIER_SMOOTHING = {"region_pair": 50, "pickup": 20, "delivery": 20, "lane": 10, "lane_equipment": 10}
HIER_PASSES = 3            # backfitting passes over the hierarchy levels
HIER_OUTER_ITERS = 3       # hierarchical_shrinkage: alternate base fit <-> effects

# ---- features -----------------------------------------------------------------------
# market_index is used only through its *row-level* deviation from the national
# daily value (mi_row_dev). The daily level itself is NOT a feature: its Jan-Aug
# relation to price breaks in Sep-Oct (see reports/summary.md).
FEATURE_GROUPS = {
    "distance": ["distance", "log_distance"],
    "equipment": ["equipment_code"],
    "weight": ["weight_clean", "weight_missing", "weight_was_negative", "weight_capped"],
    "mi_row_dev": ["mi_row_dev", "market_index_missing"],
    "dayofweek": ["dayofweek"],
    "geo": ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "d_lat", "d_lon",
            "pickup_region", "delivery_region"],
    "city_te": ["te_pickup", "te_delivery", "pickup_seen", "delivery_seen"],
    "lane_te": ["te_lane", "te_lane_equipment"],
}
BASE_FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]
# Optional columns tested in the feature study (and the "previous" reference model).
EXTRA_FEATURES = {
    "season": ["month", "dayofyear"],
    "market_level": ["market_index_clean", "mi_day", "mi_7d", "mi_week_dev"],   # the v1 market features
    "recent_level_30": ["level_30d"],
    "recent_level_60": ["level_60d"],
    **{f"mi_lag_{k}": [f"mi_lag_{k}"] for k in (1, 7, 14)},
    **{f"mi_lead_{k}": [f"mi_lead_{k}"] for k in (1, 7, 14)},
    "mi_change_7d": ["mi_change_7d"],
    "mi_ratio_28d": ["mi_ratio_28d"],
    "mi_x_equipment": ["mi_7d", "mi_7d_x_equipment"],
    "quote_signal": ["quote_signal"],          # leaks the label in training months only (summary.md)
}

# ---- cross-validation -----------------------------------------------------------------
ROLLING_MONTHS = [7, 8, 9, 10]   # rolling_M: train on months < M, test on M
COLD_START_REPEATS = 3           # forward split with 8 random cities held out, 3 city sets

# ---- models -----------------------------------------------------------------------------
# LightGBM recommends the number of *physical* cores (about half the logical CPUs).
# Override with e.g.  N_JOBS=8 python main.py
N_JOBS = int(os.environ.get("N_JOBS", max(1, (os.cpu_count() or 2) // 2)))
N_SEEDS = 5                      # LightGBM candidates average 5 seeds (CV and final fit)
LGBM_PARAMS = dict(
    objective="huber", alpha=0.1,          # Huber in log space: robust to leftover noise
    n_estimators=1000, learning_rate=0.05, num_leaves=15, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=SEED, verbose=-1, n_jobs=N_JOBS,
)
# Small deterministic grid, searched on the 5 time folds (single seed).
# HALF_LIFE = recency weighting: weight = 0.5 ** (days before the last training day / half-life).
TUNING_GRID = dict(num_leaves=[7, 15, 31], n_estimators=[600, 1200])
HALF_LIVES = [None, 90, 180, 365]
# The model selected in v1 (market-level features + season, old settings); kept as a reference.
V1_LGBM_PARAMS = {**LGBM_PARAMS, "n_estimators": 1500, "learning_rate": 0.03, "num_leaves": 63,
                  "min_child_samples": 30}
CATBOOST_PARAMS = dict(
    loss_function="Huber:delta=0.1", iterations=1500, learning_rate=0.05, depth=6,
    l2_leaf_reg=3.0, random_seed=SEED, verbose=False, thread_count=N_JOBS,
    allow_writing_files=False,
)

# ---- RMSE-aware calibration ---------------------------------------------------------------
# The model predicts a typical (median-like) rate. A global multiplier can lower RMSE on noisy
# rows at the cost of MAE on normal rows. Both options are evaluated in CV (summary.md);
# set to "rmse_multiplier" or "smearing" to apply one to the final predictions.
CALIBRATION = None
