# Changelog

## 1.0.0 — 2026-11-20

First public version: a full rebuild on synthetic data, developed in three steps.

### Data and leakage-free features

- New generic domain: a fictional B2B company where customers request quotes.
- Deterministic synthetic data generator (seed 42). It writes only what would be known on the extraction
  date, so later conversions are empty, as in a real extract.
- History features per request. An earlier request's conversion counts only if it happened strictly before
  the request being scored. The earlier logic counted every earlier conversion, which on this data leaks
  future information into 15.2% of requests.
- Tests: the leakage rule, other customers' history, timestamp ties, and an independent recomputation in
  plain Python. Deliberately breaking the rule makes them fail.

### Label, split, models and evaluation

- Label: converted within 5 days. Requests whose window had not closed by the extraction date get no label.
- Time split with a 5-day embargo before each boundary. Cut-offs are UTC and do not depend on the machine's
  time zone.
- Models compared on the same splits: the naive historical rate, a GBT without history, a GBT with
  leakage-free history, and a demo of the leaky logic.
- Ranking metrics with the correct definition of capture (share of all conversions) and reproducible
  tie-breaking. Runs are logged to a local MLflow store.
- Results reported as they are: the GBT ties the naive rule, and the leak barely changes the metrics on
  this data.
- The notebook is generated from code and executed with its outputs.

### Documentation and CI

- README with the problem definition, results, chart and how to run it.
- `docs/method.md` and `docs/limitations.md`.
- GitHub Actions: Java 17 + Python 3.12, a data reproducibility check and the test suite.
- MIT license.
