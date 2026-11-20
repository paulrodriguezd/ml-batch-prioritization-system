import os
import sys
from datetime import datetime

from pyspark.sql import Column, SparkSession
from pyspark.sql import functions as F


def ts_lit(dt: datetime) -> Column:
    """A timestamp literal read in the session time zone (UTC).

    F.lit(datetime) would convert using the operating system's time zone,
    shifting every cut-off by an hour or two on a laptop outside UTC.
    """
    return F.to_timestamp(F.lit(dt.strftime("%Y-%m-%d %H:%M:%S")))


def get_spark(app_name: str = "batch-prioritization") -> SparkSession:
    """Local Spark session. All timestamps are handled in UTC."""
    # Spark workers must use the same Python as this process.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
