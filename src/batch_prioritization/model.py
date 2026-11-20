"""Model pipelines: a gradient-boosted trees classifier in Spark ML.

Missing values are filled with fixed, documented constants instead of
statistics, so nothing is learned from the validation or test periods.
"""

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from batch_prioritization.config import SEED
from batch_prioritization.features import (
    HISTORY_FEATURES,
    REQUEST_CATEGORICAL_FEATURES,
    REQUEST_NUMERIC_FEATURES,
)

# "Never happened" is encoded as a very long time ago; trees split on it cleanly.
NEVER_DAYS = 9999.0

FILL_VALUES = {
    "prior_request_count": 0,
    "prior_known_conversion_count": 0,
    "historical_conversion_rate": 0.0,
    "previous_request_converted_known": 0,
    "days_since_last_request": NEVER_DAYS,
    "days_since_last_known_conversion": NEVER_DAYS,
    "customer_tenure_days": 0,
}


def prepare(df: DataFrame) -> DataFrame:
    """Add explicit "has history" flags, then fill nulls with constants."""
    return df.withColumn(
        "has_prior_request", (F.col("prior_request_count") > 0).cast("double")
    ).fillna(FILL_VALUES)


def gbt_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical
    ]
    assembler = VectorAssembler(inputCols=numeric + [f"{c}_idx" for c in categorical], outputCol="features")
    gbt = GBTClassifier(
        labelCol="label",
        featuresCol="features",
        maxDepth=5,
        maxIter=80,
        stepSize=0.05,
        subsamplingRate=0.8,
        seed=SEED,
    )
    return Pipeline(stages=indexers + [assembler, gbt])


# Request attributes only: what a rule-free model knows without history.
NO_HISTORY_NUMERIC = REQUEST_NUMERIC_FEATURES
FULL_NUMERIC = REQUEST_NUMERIC_FEATURES + HISTORY_FEATURES + ["has_prior_request"]
CATEGORICAL = REQUEST_CATEGORICAL_FEATURES
