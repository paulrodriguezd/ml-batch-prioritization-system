"""Train and evaluate the models, log them to MLflow and write reports/.

Models compared on the same splits:
- naive:       score = customer's historical conversion rate (no ML);
- no_history:  GBT on request attributes only;
- full:        GBT on request attributes + leakage-free history;
- leaky_demo:  GBT with the original (leaky) history logic, to measure
               how much it inflates the metrics. Not a candidate model.

Usage:
    python -m batch_prioritization.train
"""

import json
from pathlib import Path

import mlflow
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from batch_prioritization import config
from batch_prioritization.data import load_customers, load_requests
from batch_prioritization.evaluation import deciles, summary, with_score
from batch_prioritization.features import build_features
from batch_prioritization.labels import build_labels
from batch_prioritization.leaky import leaky_history_features
from batch_prioritization.model import CATEGORICAL, FULL_NUMERIC, NO_HISTORY_NUMERIC, gbt_pipeline, prepare
from batch_prioritization.spark import get_spark
from batch_prioritization.split import time_split

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
LEAKY_COLUMNS = [
    "prior_request_count",
    "prior_known_conversion_count",
    "previous_request_converted_known",
    "historical_conversion_rate",
    "days_since_last_known_conversion",
]


def build_dataset(spark) -> DataFrame:
    requests, customers = load_requests(spark), load_customers(spark)
    features = prepare(build_features(requests, customers))
    return features.join(build_labels(requests), on="request_id", how="inner").cache()


def build_leaky_dataset(spark, honest: DataFrame) -> DataFrame:
    leaky = leaky_history_features(load_requests(spark))
    return prepare(honest.drop(*LEAKY_COLUMNS).join(leaky, on="request_id", how="left")).cache()


def naive_scores(train: DataFrame, df: DataFrame) -> DataFrame:
    """Historical conversion rate; first-time customers get the training base rate."""
    base_rate = train.agg(F.avg("label")).first()[0]
    return df.withColumn(
        "score",
        F.when(F.col("has_prior_request") == 1, F.col("historical_conversion_rate")).otherwise(F.lit(base_rate)),
    )


def log_run(name: str, params: dict, results: dict, model=None) -> None:
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        for split, metrics in results["metrics"].items():
            mlflow.log_metrics({f"{split}_{k}": v for k, v in metrics.items()})
        mlflow.log_dict(results["test_deciles"], "test_deciles.json")
        if model is not None:
            mlflow.spark.log_model(model, artifact_path="model")


def evaluate(name: str, scored: dict) -> dict:
    metrics = {split: summary(df) for split, df in scored.items()}
    return {"metrics": metrics, "test_deciles": deciles(scored["test"])}


def main() -> None:
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")
    # Local MLflow: runs in a SQLite file, artifacts (models, tables) in mlruns/.
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    if mlflow.get_experiment_by_name("batch-prioritization") is None:
        mlflow.create_experiment("batch-prioritization", artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment("batch-prioritization")

    data = build_dataset(spark)
    train, valid, test = time_split(data)
    splits = {"valid": valid, "test": test}
    common = {
        "label_days": config.LABEL_DAYS,
        "valid_start": config.VALID_START.date().isoformat(),
        "test_start": config.TEST_START.date().isoformat(),
        "as_of": config.AS_OF.date().isoformat(),
        "n_train": train.count(),
    }
    results = {}

    # Naive baseline: no model.
    results["naive"] = evaluate("naive", {s: naive_scores(train, df) for s, df in splits.items()})
    log_run("naive", {**common, "model": "historical_conversion_rate"}, results["naive"])

    # GBT models.
    for name, numeric in [("no_history", NO_HISTORY_NUMERIC), ("full", FULL_NUMERIC)]:
        model = gbt_pipeline(numeric, CATEGORICAL).fit(train)
        results[name] = evaluate(name, {s: with_score(model.transform(df)) for s, df in splits.items()})
        log_run(name, {**common, "model": "gbt", "features": ",".join(numeric + CATEGORICAL)}, results[name], model)

    # Leakage demonstration: same model, original history logic.
    leaky_train, leaky_valid, leaky_test = time_split(build_leaky_dataset(spark, data))
    leaky_model = gbt_pipeline(FULL_NUMERIC, CATEGORICAL).fit(leaky_train)
    results["leaky_demo"] = evaluate(
        "leaky_demo", {s: with_score(leaky_model.transform(df)) for s, df in {"valid": leaky_valid, "test": leaky_test}.items()}
    )
    log_run("leaky_demo", {**common, "model": "gbt", "note": "leaky history, demo only"}, results["leaky_demo"])

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps({"params": common, "results": results}, indent=2))

    print(f"train={common['n_train']}  valid={valid.count()}  test={test.count()}")
    print(f"{'model':<12}{'split':<7}{'n':>7}{'base':>7}{'ROC':>7}{'PR':>7}{'prec@10':>9}{'capt@10':>9}{'capt@20':>9}")
    for name, r in results.items():
        for split, m in r["metrics"].items():
            print(
                f"{name:<12}{split:<7}{m['n']:>7}{m['base_rate']:>7.3f}{m['roc_auc']:>7.3f}{m['pr_auc']:>7.3f}"
                f"{m['precision_top10']:>9.3f}{m['capture_top10']:>9.3f}{m['capture_top20']:>9.3f}"
            )
    spark.stop()


if __name__ == "__main__":
    main()
