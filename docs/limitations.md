# Limitations and next steps

## Limitations

- **Synthetic data.** Effects are moderate by design. The results show how the method behaves, not how large the
  gains would be on a real business.
- **The model does not beat the naive rule here.** On this data the request attributes add little to the
  customer's history. On real data that may change; the comparison has to be repeated before choosing.
- **The leak barely changes the metrics here.** That is a property of this data. The size of the effect on real
  data depends on how often outcomes arrive after the customer's next request.
- **One time split.** Results come from a single validation and test period. Differences of a few thousandths of
  AUC are within the variation of another period or seed.
- **No intervention data.** The label is "converted on its own". If the sales team contacts some requests, their
  outcome is no longer "on their own"; a real deployment needs to record who was contacted.
- **Local Spark.** Runs on a laptop in `local[*]` mode. The same code runs on a cluster, but it has not been
  tested at scale here.

## Next steps

- Rolling evaluation over several periods to measure how stable the ranking is.
- Calibration of the probabilities if they are going to be read as probabilities, not just as a ranking.
- A daily batch job that scores the unlabelled requests and writes the ranked list.
- Monitoring of the input distribution and of the conversion rate of the skipped requests.
