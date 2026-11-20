"""Tests for the label window, the embargoed time split and the ranking metrics."""

from datetime import datetime

import pytest
from pyspark.sql import functions as F

from batch_prioritization.data import REQUESTS_SCHEMA
from batch_prioritization.evaluation import capture_at, deciles
from batch_prioritization.labels import build_labels
from batch_prioritization.split import time_split


def ts(text):
    return datetime.fromisoformat(text) if text else None


def requests_df(spark, rows):
    """rows: (request_id, requested_at, converted_at or None)."""
    data = [(rid, "C1", ts(req), "Standard", "web", 1000.0, 0, ts(conv)) for rid, req, conv in rows]
    return spark.createDataFrame(data, schema=REQUESTS_SCHEMA)


def test_label_window_and_censoring(spark):
    as_of = datetime(2026, 1, 1)
    df = requests_df(
        spark,
        [
            ("in_window", "2025-12-01 10:00:00", "2025-12-06 10:00:00"),  # exactly 5 days -> 1
            ("too_late", "2025-12-01 10:00:00", "2025-12-06 10:00:01"),  # 5 days + 1 s -> 0
            ("before_request", "2025-12-01 10:00:00", "2025-11-30 10:00:00"),  # invalid -> 0
            ("never", "2025-12-01 10:00:00", None),  # -> 0
            ("closes_on_as_of", "2025-12-27 00:00:00", None),  # window closes exactly at as_of -> labelled
            ("still_open", "2025-12-27 00:00:01", None),  # outcome unknown -> no label
        ],
    )
    labels = {r["request_id"]: r["label"] for r in build_labels(df, as_of=as_of, label_days=5).collect()}
    assert labels == {"in_window": 1.0, "too_late": 0.0, "before_request": 0.0, "never": 0.0, "closes_on_as_of": 0.0}


def test_time_split_leaves_an_embargo_before_each_boundary(spark):
    df = requests_df(
        spark,
        [
            ("train", "2025-06-25 23:59:59", None),
            ("embargo_train", "2025-06-26 00:00:00", None),  # label would land in validation
            ("valid", "2025-07-01 00:00:00", None),
            ("embargo_valid", "2025-09-26 00:00:00", None),
            ("test", "2025-10-01 00:00:00", None),
        ],
    )
    train, valid, test = time_split(
        df, valid_start=datetime(2025, 7, 1), test_start=datetime(2025, 10, 1), label_days=5
    )
    ids = lambda d: {r["request_id"] for r in d.collect()}  # noqa: E731
    assert ids(train) == {"train"}
    assert ids(valid) == {"valid"}
    assert ids(test) == {"test"}


def scored(spark, pairs):
    """pairs: (score, label) -> DataFrame with request_id, score, label."""
    return spark.createDataFrame(
        [(f"r{i:02d}", float(s), float(lab)) for i, (s, lab) in enumerate(pairs)], ["request_id", "score", "label"]
    )


def test_capture_is_share_of_all_conversions_not_precision(spark):
    # 10 requests, 4 conversions; the top 10% (1 request) is a conversion.
    df = scored(spark, [(0.9, 1), (0.8, 0), (0.7, 1), (0.6, 0), (0.5, 1), (0.4, 0), (0.3, 0), (0.2, 1), (0.1, 0), (0.0, 0)])
    top = capture_at(df, 0.10)
    assert top["precision"] == 1.0  # 1 of 1 in the top converts
    assert top["capture"] == 0.25  # it is 1 of the 4 conversions


def test_deciles_lift_on_a_perfect_ranking(spark):
    # 20 requests, the 2 with the highest score are the only conversions.
    df = scored(spark, [(1 - i / 20, 1 if i < 2 else 0) for i in range(20)])
    d = deciles(df)
    assert [row["decile"] for row in d] == list(range(1, 11))
    assert d[0]["conversion_rate"] == 1.0
    assert d[0]["lift"] == pytest.approx(10.0)  # 100% vs 10% overall
    assert sum(row["n"] for row in d) == 20


def test_cut_offs_do_not_depend_on_the_os_time_zone(spark):
    """The split boundary is 2025-07-01 00:00 UTC even if the laptop is in Madrid."""
    import os
    import time

    df = spark.sql(
        "SELECT 'late_june' AS request_id, to_timestamp('2025-06-30 23:30:00') AS requested_at "
        "UNION ALL SELECT 'july', to_timestamp('2025-07-01 00:30:00')"
    )
    os.environ["TZ"] = "Europe/Madrid"
    time.tzset()
    try:
        _, valid, _ = time_split(df, valid_start=datetime(2025, 7, 1), test_start=datetime(2025, 10, 1), label_days=5)
        assert {r["request_id"] for r in valid.collect()} == {"july"}
    finally:
        os.environ["TZ"] = "UTC"
        time.tzset()


def test_naive_baseline_uses_train_base_rate_for_new_customers(spark):
    from batch_prioritization.train import naive_scores

    train = spark.createDataFrame([(1.0,), (0.0,), (0.0,), (0.0,)], ["label"])  # base rate 0.25
    df = spark.createDataFrame(
        [("new", 0.0, None), ("returning", 1.0, 0.8)],
        "request_id string, has_prior_request double, historical_conversion_rate double",
    )
    scores = {r["request_id"]: r["score"] for r in naive_scores(train, df).collect()}
    assert scores == {"new": 0.25, "returning": 0.8}


def test_ranking_metrics_are_reproducible_with_ties(spark):
    # All scores tied: the result must be the same on every call.
    df = scored(spark, [(0.5, i % 2) for i in range(40)])
    results = {tuple(capture_at(df, 0.10).values()) for _ in range(3)}
    assert len(results) == 1


def test_capture_with_no_conversions_is_nan(spark):
    import math

    df = scored(spark, [(0.9, 0), (0.1, 0)])
    assert math.isnan(capture_at(df, 0.5)["capture"])


def test_leaky_logic_counts_future_conversions_and_honest_does_not(spark):
    from batch_prioritization.features import build_history_features
    from batch_prioritization.leaky import leaky_history_features

    df = requests_df(
        spark,
        [
            ("r1", "2025-01-10 09:00:00", "2025-01-20 09:00:00"),  # converts after r2 is made
            ("r2", "2025-01-15 09:00:00", None),
        ],
    )
    honest = {r["request_id"]: r for r in build_history_features(df).collect()}["r2"]
    leaky = {r["request_id"]: r for r in leaky_history_features(df).collect()}["r2"]
    assert honest["prior_known_conversion_count"] == 0
    assert leaky["prior_known_conversion_count"] == 1
    assert leaky["days_since_last_known_conversion"] < 0  # a "last conversion" in the future
