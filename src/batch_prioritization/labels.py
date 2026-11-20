"""Target: did the request convert within LABEL_DAYS of being made?

Only requests whose LABEL_DAYS window had fully closed by AS_OF get a
label. A recent request with no conversion yet is "unknown", not 0.
"""

from datetime import datetime, timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from batch_prioritization.config import AS_OF, LABEL_DAYS
from batch_prioritization.spark import ts_lit


def build_labels(requests: DataFrame, as_of: datetime = AS_OF, label_days: int = LABEL_DAYS) -> DataFrame:
    """One row per labelled request: request_id, label (0/1).

    A conversion recorded before the request itself is treated as invalid
    data and does not count as a conversion.
    """
    window = F.expr(f"INTERVAL {label_days} DAYS")
    closed_by = ts_lit(as_of - timedelta(days=label_days))

    converted_in_window = (
        F.col("converted_at").isNotNull()
        & (F.col("converted_at") >= F.col("requested_at"))
        & (F.col("converted_at") <= F.col("requested_at") + window)
    )
    return requests.filter(F.col("requested_at") <= closed_by).select(
        "request_id",
        F.when(converted_in_window, 1.0).otherwise(0.0).alias("label"),
    )
