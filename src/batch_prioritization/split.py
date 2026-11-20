"""Time-based train / validation / test split with an embargo.

A training request made just before VALID_START would only get its label
LABEL_DAYS later, inside the validation period. To keep every split's
labels fully known before the next split starts, requests in the last
LABEL_DAYS before each boundary are dropped.
"""

from datetime import datetime, timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from batch_prioritization.config import LABEL_DAYS, TEST_START, VALID_START
from batch_prioritization.spark import ts_lit


def time_split(
    df: DataFrame,
    valid_start: datetime = VALID_START,
    test_start: datetime = TEST_START,
    label_days: int = LABEL_DAYS,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    gap = timedelta(days=label_days)
    t = F.col("requested_at")
    train = df.filter(t < ts_lit(valid_start - gap))
    valid = df.filter((t >= ts_lit(valid_start)) & (t < ts_lit(test_start - gap)))
    test = df.filter(t >= ts_lit(test_start))
    return train, valid, test
