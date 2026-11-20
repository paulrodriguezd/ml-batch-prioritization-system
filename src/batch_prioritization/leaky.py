"""FOR DEMONSTRATION ONLY: the history logic of the original notebook.

It counts every earlier request's conversion, even when that conversion
happened after the request being scored, and takes the "last conversion"
from those too. It is used once in training to show how much this
inflates the metrics. Never use it in a real model.
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

SECONDS_PER_DAY = 86_400


def leaky_history_features(requests: DataFrame) -> DataFrame:
    order = Window.partitionBy("customer_id").orderBy("requested_at", "request_id")
    earlier = order.rowsBetween(Window.unboundedPreceding, -1)
    converted = F.col("converted_at").isNotNull().cast("int")
    return requests.select(
        "request_id",
        "requested_at",
        F.count("*").over(earlier).alias("prior_request_count"),
        F.coalesce(F.sum(converted).over(earlier), F.lit(0)).alias("prior_known_conversion_count"),
        F.lag(converted, 1).over(order).alias("previous_request_converted_known"),
        F.max("converted_at").over(earlier).alias("_last_conversion_at"),
    ).select(
        "request_id",
        "prior_request_count",
        "prior_known_conversion_count",
        "previous_request_converted_known",
        F.when(
            F.col("prior_request_count") > 0,
            F.col("prior_known_conversion_count") / F.col("prior_request_count"),
        ).alias("historical_conversion_rate"),
        # Can be negative: the "last conversion" may be in the future.
        ((F.unix_timestamp("requested_at") - F.unix_timestamp("_last_conversion_at")) / SECONDS_PER_DAY).alias(
            "days_since_last_known_conversion"
        ),
    )
