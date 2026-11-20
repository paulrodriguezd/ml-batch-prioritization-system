"""Features for each quote request, using only what was known at request time.

The rule that prevents leakage: for a request made at time t, a previous
request only contributes its conversion if that conversion happened
strictly before t. A previous request that converted *after* t looks, at
time t, like a request that has not converted (yet).

History counts any earlier purchase, however long it took; the label only
counts purchases within LABEL_DAYS. They are related but different things
on purpose: "has this customer bought before" vs "will this request
convert on its own quickly".
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

SECONDS_PER_DAY = 86_400

HISTORY_FEATURES = [
    "prior_request_count",
    "prior_known_conversion_count",
    "historical_conversion_rate",
    "previous_request_converted_known",
    "days_since_last_request",
    "days_since_last_known_conversion",
]

REQUEST_NUMERIC_FEATURES = ["log_quote_amount", "discount_pct", "customer_tenure_days"]
REQUEST_CATEGORICAL_FEATURES = ["product_line", "channel", "segment", "region", "industry"]


def _days_between(later, earlier):
    return (F.unix_timestamp(later) - F.unix_timestamp(earlier)) / SECONDS_PER_DAY


def build_history_features(requests: DataFrame) -> DataFrame:
    """One row per request with the customer's history up to that moment.

    Implemented as a self-join (each request with the same customer's earlier
    requests) rather than window functions, because the leakage rule depends
    on comparing the earlier conversion time with the current request time,
    which a row-based window cannot express.
    """
    cur = requests.select("request_id", "customer_id", "requested_at").alias("cur")
    prev = requests.select("customer_id", "requested_at", "converted_at").alias("prev")

    # Strictly earlier requests only: a request never sees itself.
    pairs = cur.join(
        prev,
        (F.col("cur.customer_id") == F.col("prev.customer_id"))
        & (F.col("prev.requested_at") < F.col("cur.requested_at")),
        how="left",
    )

    known = F.col("prev.converted_at") < F.col("cur.requested_at")

    history = pairs.groupBy("cur.request_id", "cur.requested_at").agg(
        F.count("prev.requested_at").alias("prior_request_count"),
        F.sum(F.when(known, 1).otherwise(0)).alias("prior_known_conversion_count"),
        F.max("prev.requested_at").alias("last_prior_request_at"),
        F.max(F.when(known, F.col("prev.converted_at"))).alias("last_known_conversion_at"),
        # Outcome of the most recent earlier request, as known at time t.
        # max over (requested_at, flag) is deterministic: if two earlier
        # requests share the latest timestamp, a known conversion wins.
        F.max(
            F.struct(
                F.col("prev.requested_at").alias("at"),
                F.when(known, 1).otherwise(0).alias("flag"),
            )
        ).alias("_last_prev"),
    )

    return history.select(
        "request_id",
        "prior_request_count",
        "prior_known_conversion_count",
        F.when(
            F.col("prior_request_count") > 0,
            F.col("prior_known_conversion_count") / F.col("prior_request_count"),
        ).alias("historical_conversion_rate"),
        # No earlier request -> null, not 0.
        F.when(F.col("prior_request_count") > 0, F.col("_last_prev.flag")).alias(
            "previous_request_converted_known"
        ),
        _days_between(F.col("requested_at"), F.col("last_prior_request_at")).alias("days_since_last_request"),
        _days_between(F.col("requested_at"), F.col("last_known_conversion_at")).alias(
            "days_since_last_known_conversion"
        ),
    )


def build_features(requests: DataFrame, customers: DataFrame) -> DataFrame:
    """Request attributes + customer attributes + history features.

    Nulls are kept on purpose (e.g. no history yet); imputation belongs to
    the modelling step, where it can be fitted on the training period only.
    The outcome (converted_at) is deliberately not included: the label is
    built in a separate step, so it cannot slip into the features.
    """
    request_level = requests.join(customers, on="customer_id", how="left").select(
        "request_id",
        "customer_id",
        "requested_at",
        "product_line",
        "channel",
        "segment",
        "region",
        "industry",
        F.log10(F.col("quote_amount")).alias("log_quote_amount"),
        "discount_pct",
        F.datediff(F.to_date("requested_at"), F.col("signup_date")).alias("customer_tenure_days"),
    )
    return request_level.join(build_history_features(requests), on="request_id", how="left")
