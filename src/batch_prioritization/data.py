"""Load the raw CSVs with explicit schemas (no type inference)."""

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

CUSTOMERS_SCHEMA = StructType(
    [
        StructField("customer_id", StringType(), False),
        StructField("segment", StringType(), True),
        StructField("region", StringType(), True),
        StructField("industry", StringType(), True),
        StructField("signup_date", DateType(), True),
    ]
)

REQUESTS_SCHEMA = StructType(
    [
        StructField("request_id", StringType(), False),
        StructField("customer_id", StringType(), False),
        StructField("requested_at", TimestampType(), False),
        StructField("product_line", StringType(), True),
        StructField("channel", StringType(), True),
        StructField("quote_amount", DoubleType(), True),
        StructField("discount_pct", IntegerType(), True),
        # Empty when there was no conversion before the extraction date.
        StructField("converted_at", TimestampType(), True),
    ]
)


def load_customers(spark: SparkSession, raw_dir: Path = RAW_DIR) -> DataFrame:
    return spark.read.csv(str(raw_dir / "customers.csv"), header=True, schema=CUSTOMERS_SCHEMA)


def load_requests(spark: SparkSession, raw_dir: Path = RAW_DIR) -> DataFrame:
    return spark.read.csv(str(raw_dir / "quote_requests.csv"), header=True, schema=REQUESTS_SCHEMA)
