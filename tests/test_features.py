"""Leakage tests for the history features.

Each test builds a tiny, hand-made history for one customer and checks
what the features say at the moment of the last request.
"""

from datetime import datetime

import pytest

from batch_prioritization.data import REQUESTS_SCHEMA
from batch_prioritization.features import build_history_features


def ts(text):
    return datetime.fromisoformat(text)


def history_for(spark, rows):
    """rows: (request_id, requested_at, converted_at or None[, customer_id])."""
    data = [
        (r[0], r[3] if len(r) > 3 else "C1", ts(r[1]), "Standard", "web", 1000.0, 0, ts(r[2]) if r[2] else None)
        for r in rows
    ]
    df = spark.createDataFrame(data, schema=REQUESTS_SCHEMA)
    return {r["request_id"]: r.asDict() for r in build_history_features(df).collect()}


def test_first_request_has_no_history(spark):
    h = history_for(spark, [("r1", "2025-01-10 09:00:00", None)])["r1"]
    assert h["prior_request_count"] == 0
    assert h["prior_known_conversion_count"] == 0
    assert h["historical_conversion_rate"] is None
    assert h["previous_request_converted_known"] is None
    assert h["days_since_last_request"] is None
    assert h["days_since_last_known_conversion"] is None


def test_conversion_after_current_request_is_not_counted(spark):
    # r1 converts on 20 Jan, but r2 is made on 15 Jan: at that moment
    # nobody could know r1 would convert.
    h = history_for(
        spark,
        [
            ("r1", "2025-01-10 09:00:00", "2025-01-20 09:00:00"),
            ("r2", "2025-01-15 09:00:00", None),
        ],
    )["r2"]
    assert h["prior_request_count"] == 1
    assert h["prior_known_conversion_count"] == 0
    assert h["historical_conversion_rate"] == 0.0
    assert h["previous_request_converted_known"] == 0
    assert h["days_since_last_known_conversion"] is None


def test_conversion_before_current_request_is_counted(spark):
    h = history_for(
        spark,
        [
            ("r1", "2025-01-10 09:00:00", "2025-01-12 09:00:00"),
            ("r2", "2025-01-15 09:00:00", None),
        ],
    )["r2"]
    assert h["prior_known_conversion_count"] == 1
    assert h["historical_conversion_rate"] == 1.0
    assert h["previous_request_converted_known"] == 1
    assert h["days_since_last_known_conversion"] == pytest.approx(3.0)
    assert h["days_since_last_request"] == pytest.approx(5.0)


def test_conversion_at_the_same_second_is_not_counted(spark):
    # Strictly earlier: a conversion recorded at the very same moment
    # as the new request is not treated as known.
    h = history_for(
        spark,
        [
            ("r1", "2025-01-10 09:00:00", "2025-01-15 09:00:00"),
            ("r2", "2025-01-15 09:00:00", None),
        ],
    )["r2"]
    assert h["prior_known_conversion_count"] == 0


def test_previous_request_flag_uses_the_most_recent_request(spark):
    # r1 converted (known), r2 did not: the previous request (r2) did not convert.
    h = history_for(
        spark,
        [
            ("r1", "2025-01-01 09:00:00", "2025-01-02 09:00:00"),
            ("r2", "2025-01-05 09:00:00", None),
            ("r3", "2025-01-10 09:00:00", None),
        ],
    )["r3"]
    assert h["prior_request_count"] == 2
    assert h["prior_known_conversion_count"] == 1
    assert h["historical_conversion_rate"] == 0.5
    assert h["previous_request_converted_known"] == 0


def test_other_customers_history_is_ignored(spark):
    # C2 converted before C1's request: it must not count for C1.
    h = history_for(
        spark,
        [
            ("x1", "2025-01-01 09:00:00", "2025-01-02 09:00:00", "C2"),
            ("r1", "2025-01-10 09:00:00", None, "C1"),
        ],
    )["r1"]
    assert h["prior_request_count"] == 0
    assert h["prior_known_conversion_count"] == 0


def test_previous_request_flag_is_deterministic_on_ties(spark):
    # r1 and r2 share the latest timestamp; r1's conversion is known.
    # The rule: on a tie, a known conversion wins, so the flag is 1.
    rows = [
        ("r1", "2025-01-05 09:00:00", "2025-01-06 09:00:00"),
        ("r2", "2025-01-05 09:00:00", None),
        ("r3", "2025-01-10 09:00:00", None),
    ]
    flags = {history_for(spark, rows)["r3"]["previous_request_converted_known"] for _ in range(3)}
    assert flags == {1}


def test_no_negative_recency_on_generated_data(spark):
    """On the full synthetic dataset, no recency can be negative (that would
    mean using an event from the future)."""
    from pyspark.sql import functions as F

    from batch_prioritization.data import RAW_DIR, load_requests

    if not (RAW_DIR / "quote_requests.csv").exists():
        pytest.skip("run scripts/generate_data.py first")
    h = build_history_features(load_requests(spark))
    negatives = h.filter(
        (F.col("days_since_last_request") < 0) | (F.col("days_since_last_known_conversion") < 0)
    ).count()
    assert negatives == 0
    assert h.filter(F.col("prior_known_conversion_count") > F.col("prior_request_count")).count() == 0


def test_matches_plain_python_on_a_sample(spark):
    """Recompute the counts in plain Python for 50 random requests and
    compare: an independent check of the leakage rule on real-looking data."""
    import csv
    import random
    from datetime import datetime

    from batch_prioritization.data import RAW_DIR, load_requests

    path = RAW_DIR / "quote_requests.csv"
    if not path.exists():
        pytest.skip("run scripts/generate_data.py first")

    rows = list(csv.DictReader(path.open()))
    by_customer = {}
    for r in rows:
        by_customer.setdefault(r["customer_id"], []).append(r)
    sample = random.Random(0).sample(rows, 50)

    h = {r["request_id"]: r for r in build_history_features(load_requests(spark)).collect()}
    for r in sample:
        t = datetime.fromisoformat(r["requested_at"])
        earlier = [p for p in by_customer[r["customer_id"]] if datetime.fromisoformat(p["requested_at"]) < t]
        known = [p for p in earlier if p["converted_at"] and datetime.fromisoformat(p["converted_at"]) < t]
        assert h[r["request_id"]]["prior_request_count"] == len(earlier)
        assert h[r["request_id"]]["prior_known_conversion_count"] == len(known)
