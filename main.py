"""Spotter freight-rate model, end to end:   python main.py   (no options)

1. ETL       load + clean train / validation / December inputs            (src/data.py)
2. EDA       figures, quote_signal + market_index checks, outlier tests   (src/eda.py, src/outlier_diagnostics.py)
3. Models    time-based CV: tune LightGBM, compare all candidates, pick    (src/cv.py, src/models.py)
             the lowest selection score, feature study, calibration check
4. Final     retrain the winner on all Jan-Oct rows, write
             validation_predictions.csv + data/december_chart_inputs.csv,
             run score.py                                                  (src/predict.py)
5. Report    reports/summary.md + figures                                  (src/report.py)
"""
import json
import time

from src import config, cv, eda, outlier_diagnostics, predict, report
from src.data import city_lookup, load_all


def step(title):
    print(f"[{(time.time() - START) / 60:5.1f} min] {title}", flush=True)


if __name__ == "__main__":
    START = time.time()
    for d in (config.FIGURES, config.ARTIFACTS):
        d.mkdir(parents=True, exist_ok=True)

    step("1/5 load and clean data")
    train, valid, december, weight_fill, market = load_all()
    cities = city_lookup(train, valid)

    step("2/5 EDA and outlier diagnostics")
    eda.make_eda_figures(train, valid)
    quote = eda.quote_signal_check(train, valid)
    level = eda.price_level_vs_market(train)
    verdict = outlier_diagnostics.run(train)
    if verdict["explained"]:
        print("    WARNING: outlier diagnostics found a factor that explains the spikes - see reports/outlier_diagnostics.md")

    step("3/5 time-based CV")
    folds, eval_clean = cv.prepare_all(train, cities)
    tuning, params, half_life = cv.tune(folds)
    print(f"    tuned: num_leaves={params['num_leaves']} n_estimators={params['n_estimators']} half_life={half_life}")
    results, preds = cv.compare(folds, cities, params, half_life)
    summary = cv.summarise(results)
    best = summary.index[0]
    print(f"    selected: {best} (selection score {summary.loc[best, 'selection_score']:.3f})")
    calibration = cv.calibration_study(folds, preds, best)
    study = cv.feature_study(folds, train, cities, eval_clean, params, half_life)

    step("4/5 final fit, submission files, score.py")
    fit = predict.final_fit(best, train, valid, december, weight_fill, market, cities, params, half_life)
    template, dec_out = predict.write_outputs(fit, valid, december)
    dec_facts = predict.december_figure(dec_out, train)
    predict.run_scorer()

    step("5/5 report")
    for name, table in [("quote_signal_check", quote), ("price_level_by_month", level), ("tuning", tuning),
                        ("cv_results_long", results), ("cv_summary", summary), ("calibration", calibration),
                        ("feature_study", study)]:
        table.to_csv(config.ARTIFACTS / f"{name}.csv")
    (config.ARTIFACTS / "best_model.json").write_text(json.dumps(
        dict(model=best, lgbm_params={k: params[k] for k in config.TUNING_GRID}, half_life=half_life,
             calibration=config.CALIBRATION, calibration_factor=fit["factor"]), indent=2))
    report.write(train, valid, summary, results, tuning, params, half_life, calibration, study, quote, level,
                 verdict, best, fit, template, dec_facts, minutes=(time.time() - START) / 60)
    step("done")
