# Method

## 1. The question

Every day the company receives quote requests. Some customers buy on their own within a few days, without any
sales follow-up. The model estimates, for each request, the probability of that happening within **5 days**.
The sales team skips the requests at the top of the ranking and spends its limited time on the rest.

## 2. The label and why some requests have none

`label = 1` if the request converted within 5 days of being made, else `0`.

A request made on 30 December has had only two days to convert by the extraction date (1 January). Calling it
`0` would mix "did not convert" with "we do not know yet". So only requests whose 5-day window had closed by the
extraction date are labelled; the rest are left out of training and evaluation. A conversion dated before its
own request is treated as invalid data, not as a conversion.

## 3. History features without information from the future

For a request made at time *t*, the features describe the customer's past: how many requests they made, how many
of those converted, the outcome of the previous one and how long ago things happened.

The rule: **an earlier request's conversion counts only if it happened strictly before *t*.** An earlier request
that converts after *t* looks, at time *t*, like one that has not converted yet. The original version of this
project counted every earlier conversion regardless of when it happened; on the synthetic data that touches about
15% of requests.

Implementation: a self-join of each request with the same customer's earlier requests
(`features.py`). Window functions cannot express this rule, because it compares the earlier *conversion* time
with the current *request* time.

History counts any earlier purchase, however long it took; the label counts only purchases within 5 days.
"Has this customer bought before" and "will this request convert on its own quickly" are different questions
on purpose.

## 4. Time split with an embargo

| Split | Requests |
|---|---|
| Train | until 2025-06-25 |
| Validation | 2025-07-01 to 2025-09-25 |
| Test | from 2025-10-01 |

A training request from 29 June gets its label on 4 July, inside the validation period. To keep each split's
labels fully known before the next split starts, the last 5 days before each boundary are dropped (the
*embargo*). Cut-offs are UTC timestamps built in Spark, so they do not shift with the machine's time zone.

## 5. Models

All models are trained on the same training split and compared on the same validation and test splits.

- **Naive rule**: the customer's historical conversion rate; first-time customers get the training base rate.
  No model, and the bar every model must clear.
- **GBT, request attributes only**: product line, channel, amount, discount, customer segment, region, industry
  and tenure.
- **GBT, full**: the above plus the leakage-free history.
- **Leaky demo**: the full GBT with the original history logic, to measure what the leak does. Not a candidate.

Missing history (first request) is encoded with fixed constants plus a `has_prior_request` flag, so nothing is
learned from validation or test data. Categorical encoders are fitted on the training split inside the pipeline.

## 6. Metrics

- **ROC-AUC, PR-AUC**: overall ranking quality.
- **Precision in the top k%**: share of natural conversions among the requests the team would skip.
- **Capture in the top k%**: share of *all* natural conversions found in that top; it tells how much work the
  ranking saves.
- **Lift by decile**: conversion rate of each decile divided by the overall rate.

Ties in the score are frequent for the naive rule (many customers share the same rate). They are broken by a hash
of the request id, which is reproducible and unrelated to time or customer.
