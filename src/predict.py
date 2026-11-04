"""Final fit on all labelled data, predictions for validation.csv and the December
chart inputs, the December figure, and the provided score.py."""
from __future__ import annotations

import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .cv import calibration_factor
from .data import december_frame, flag_outliers, outlier_reference
from .features import FeatureBuilder, target
from .models import blends, build_candidates, to_rate


def final_fit(best, train, valid, december, weight_fill, market, cities, params, half_life):
    """Retrain the selected recipe (single model or blend) on all Jan-Oct rows
    (outliers dropped) and predict validation + December. Returns a dict of results."""
    clean_train = train[~flag_outliers(train, outlier_reference(train))]
    fb = FeatureBuilder(cities).fit(clean_train)
    X_tr, y_tr = fb.transform_train(), target(clean_train)
    dec_in = december_frame(december, cities, weight_fill, market)
    frames = {"valid": fb.transform(valid), "december": fb.transform(dec_in), "train_all": fb.transform(train)}
    cands = build_candidates(cities, params, half_life)
    members = blends(cands).get(best, [best])
    models = [cands[name]().fit(X_tr, y_tr) for name in members]
    log_pred = {k: np.mean([m.predict(X) for m in models], axis=0) for k, X in frames.items()}
    factor = calibration_factor(config.CALIBRATION, log_pred["train_all"], frames["train_all"])
    rate = {k: factor * to_rate(log_pred[k], frames[k]["distance"]) for k in ("valid", "december")}
    return dict(rate=rate, n_train=len(clean_train), n_dropped=len(train) - len(clean_train), factor=factor,
                dec_in=dec_in)


def write_outputs(fit, valid, december) -> tuple[pd.DataFrame, pd.DataFrame]:
    template = pd.read_csv(config.TEMPLATE_PATH)
    pred = pd.Series(fit["rate"]["valid"], index=valid["load_id"])
    template["predicted_rate"] = template["load_id"].map(pred).round(2)
    assert template["predicted_rate"].notna().all() and len(template) == 12_000
    template[["load_id", "predicted_rate"]].to_csv(config.PREDICTIONS_PATH, index=False)
    dec_out = december.copy()           # same 7 columns and row order, predicted_rate filled
    dec_out["predicted_rate"] = np.round(fit["rate"]["december"], 2)
    dec_out.to_csv(config.DECEMBER_PATH, index=False)
    return template, dec_out


def december_figure(dec_out, train, path=config.FIGURES / "13_december_vs_history.png") -> dict:
    """The December line next to the same lane's actual 2025 loads; returns the facts for summary.md."""
    d = dec_out.assign(date=pd.to_datetime(dec_out["date"]))
    lane = train[(train["pickup"] == "Lexington") & (train["delivery"] == "Fort Wayne") & (train["equipment"] == "Dry Van")]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(d["date"], d["predicted_rate"], marker="o", color="#064A56", label="model, Dec 2025 (fixed inputs)")
    ax.scatter(lane["date"], lane["posted_rate"], color="#D9822B", s=18, label="actual Lexington->Fort Wayne Dry Van, 2025")
    ax.set_ylabel("$"); ax.legend(); ax.set_title("December chart vs lane history (no holiday effect modelled)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    by_dow = d.groupby(d["date"].dt.day_name())["predicted_rate"].mean()
    return dict(min=d["predicted_rate"].min(), max=d["predicted_rate"].max(), mean=d["predicted_rate"].mean(),
                high_day=by_dow.idxmax(), low_day=by_dow.idxmin(), by_dow=by_dow,
                lane_n=len(lane), lane_median=lane["posted_rate"].median(),
                lane_min=lane["posted_rate"].min(), lane_max=lane["posted_rate"].max(),
                lane_recent=lane.sort_values("date").tail(5)[["date", "posted_rate"]])


def run_scorer():
    """Run the provided scorer exactly as documented in the task README."""
    cmd = [sys.executable, "score.py", "--predictions", config.PREDICTIONS_PATH.name,
           "--december-predictions", str(config.DECEMBER_PATH.relative_to(config.ROOT)),
           "--output-dir", config.SCORER_DIR.name]
    print("    running:", " ".join(cmd), flush=True)
    if subprocess.run(cmd, cwd=config.ROOT).returncode != 0:
        raise SystemExit("score.py failed")
