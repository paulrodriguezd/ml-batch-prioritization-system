"""Ranking metrics for a score column on a labelled DataFrame.

The top of the ranking holds the requests most likely to convert on their
own: the sales team skips them and spends its time on the rest. So the
metrics that matter are how many of the natural conversions the top of the
ranking identifies (precision) and what share of all of them it contains
(capture).

Ties in the score (frequent for the naive rule) are broken by a hash of
request_id: arbitrary but reproducible, and not related to time or customer.
"""

import math

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def with_score(predictions: DataFrame) -> DataFrame:
    """Spark ML output -> a plain 'score' column (probability of class 1)."""
    return predictions.withColumn("score", vector_to_array("probability")[1])


def _tie_breaker():
    return F.xxhash64("request_id")


def auc(df: DataFrame, metric: str = "areaUnderROC") -> float:
    return BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="score", metricName=metric).evaluate(df)


def ranked(df: DataFrame) -> DataFrame:
    w = Window.orderBy(F.col("score").desc(), _tie_breaker())
    return df.withColumn("rank", F.row_number().over(w))


def capture_at(df: DataFrame, share: float) -> dict:
    """For the top `share` of the ranking:
    - precision: conversion rate inside the top;
    - capture: share of ALL conversions that the top contains.
    """
    r = ranked(df)
    n = r.count()
    k = math.ceil(share * n)
    total_pos = r.agg(F.sum("label")).first()[0]
    top = r.filter(F.col("rank") <= k).agg(F.count("*").alias("n"), F.sum("label").alias("pos")).first()
    return {
        "precision": top["pos"] / top["n"],
        "capture": top["pos"] / total_pos if total_pos else float("nan"),
    }


def deciles(df: DataFrame) -> list[dict]:
    """Decile 1 = top 10% by score. Lift = decile conversion rate / overall rate."""
    w = Window.orderBy(F.col("score").desc(), _tie_breaker())
    base_rate = df.agg(F.avg("label")).first()[0]
    rows = (
        df.withColumn("decile", F.ntile(10).over(w))
        .groupBy("decile")
        .agg(F.count("*").alias("n"), F.avg("label").alias("conversion_rate"), F.avg("score").alias("avg_score"))
        .orderBy("decile")
        .collect()
    )
    return [
        {
            "decile": r["decile"],
            "n": r["n"],
            "conversion_rate": r["conversion_rate"],
            "avg_score": r["avg_score"],
            "lift": r["conversion_rate"] / base_rate,
        }
        for r in rows
    ]


def summary(df: DataFrame) -> dict:
    top10, top20 = capture_at(df, 0.10), capture_at(df, 0.20)
    return {
        "n": df.count(),
        "base_rate": df.agg(F.avg("label")).first()[0],
        "roc_auc": auc(df, "areaUnderROC"),
        "pr_auc": auc(df, "areaUnderPR"),
        "precision_top10": top10["precision"],
        "capture_top10": top10["capture"],
        "capture_top20": top20["capture"],
    }
