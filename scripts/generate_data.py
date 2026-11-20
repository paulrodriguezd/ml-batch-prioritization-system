"""Generate the synthetic raw data for the project.

Fictional B2B company: customers request quotes, and some of them buy
(convert) on their own a few days later, without any sales follow-up.
All data is invented. A fixed seed makes every run identical.

The data is written as it would look on the extraction date (AS_OF):
conversions that would happen after AS_OF are not known yet, so they
are left empty, exactly like in a real extract.

Signal planted on purpose (so the model has something to learn):
- each customer has a hidden propensity to buy;
- customers who bought before tend to buy again (only purchases that had
  already happened count, so honest history features can learn it);
- phone requests convert more than web ones; bigger quotes convert less;
  discounts help.

Requests run until 2025-12-31 and some conversions take up to 60 days,
so the outcome of the most recent requests is not fully known on AS_OF.
The labelling step only uses requests whose follow-up window has closed.

Usage:
    python scripts/generate_data.py            # writes data/raw/*.csv
"""

import csv
import math
import random
from datetime import date, datetime, timedelta
from pathlib import Path

SEED = 42
N_CUSTOMERS = 3000
REQUESTS_START = date(2024, 1, 1)
REQUESTS_END = date(2025, 12, 31)
AS_OF = datetime(2026, 1, 1)

SEGMENTS = [("SMB", 0.6), ("Mid-Market", 0.3), ("Enterprise", 0.1)]
REGIONS = ["North", "South", "East", "West"]
INDUSTRIES = ["Manufacturing", "Retail", "Software", "Logistics", "Services"]
PRODUCT_LINES = ["Standard", "Premium", "Custom"]
CHANNELS = ["web", "email", "phone"]

# Effects on the log-odds of converting at all.
CHANNEL_EFFECT = {"web": -0.4, "email": 0.0, "phone": 0.5}
SEGMENT_AMOUNT = {"SMB": 2_000, "Mid-Market": 10_000, "Enterprise": 50_000}
PRODUCT_AMOUNT = {"Standard": 1.0, "Premium": 1.8, "Custom": 3.0}

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def random_datetime(rng, start, end):
    seconds = int((end - start).total_seconds())
    return start + timedelta(seconds=rng.randint(0, seconds))


def build_customers(rng):
    customers = []
    for i in range(1, N_CUSTOMERS + 1):
        segment = rng.choices([s for s, _ in SEGMENTS], weights=[w for _, w in SEGMENTS])[0]
        customers.append(
            {
                "customer_id": f"C{i:05d}",
                "segment": segment,
                "region": rng.choice(REGIONS),
                "industry": rng.choice(INDUSTRIES),
                "signup_date": (date(2022, 1, 1) + timedelta(days=rng.randint(0, 1400))).isoformat(),
                # Hidden traits, used only to simulate; not written to the CSV.
                "_propensity": rng.gauss(-0.8, 1.0),
                "_speed": rng.uniform(0.3, 0.8),
            }
        )
    return customers


def build_requests(rng, customers):
    rows = []
    n = 0
    for c in customers:
        start = max(datetime.fromisoformat(c["signup_date"]), datetime.combine(REQUESTS_START, datetime.min.time()))
        end = datetime.combine(REQUESTS_END, datetime.max.time().replace(microsecond=0))
        if start >= end:
            continue
        # About 10 requests per year of activity, at least 1.
        years_active = (end - start).days / 365
        n_requests = max(1, round(rng.uniform(0.5, 1.5) * 10 * years_active))
        times = sorted(random_datetime(rng, start, end) for _ in range(n_requests))

        conversion_times = []  # when this customer's earlier conversions happened
        for requested_at in times:
            # "Bought before -> buys again" only through purchases that had
            # already happened at this moment, like the real behaviour.
            past_conversions = sum(1 for t in conversion_times if t < requested_at)
            n += 1
            product = rng.choice(PRODUCT_LINES)
            channel = rng.choices(CHANNELS, weights=[0.5, 0.3, 0.2])[0]
            amount = round(SEGMENT_AMOUNT[c["segment"]] * PRODUCT_AMOUNT[product] * rng.lognormvariate(0, 0.5), 2)
            discount = rng.choice([0, 0, 5, 10, 15])

            logit = (
                c["_propensity"]
                + CHANNEL_EFFECT[channel]
                - 0.35 * math.log10(amount / 1000)
                + 0.04 * discount
                + 0.25 * min(past_conversions, 4)
            )
            converted_at = ""
            if rng.random() < sigmoid(logit):
                # Fast buyers convert within days; the rest take weeks.
                if rng.random() < c["_speed"]:
                    delay = timedelta(days=rng.uniform(0, 5))
                else:
                    delay = timedelta(days=rng.uniform(5, 60))
                when = requested_at + delay
                conversion_times.append(when)
                # Not observable yet on the extraction date.
                if when < AS_OF:
                    converted_at = when.isoformat(sep=" ", timespec="seconds")

            rows.append(
                {
                    "request_id": f"R{n:06d}",
                    "customer_id": c["customer_id"],
                    "requested_at": requested_at.isoformat(sep=" ", timespec="seconds"),
                    "product_line": product,
                    "channel": channel,
                    "quote_amount": amount,
                    "discount_pct": discount,
                    "converted_at": converted_at,
                }
            )
    return rows


def write_csv(name, rows, columns):
    path = OUT_DIR / f"{name}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path.relative_to(OUT_DIR.parent.parent)}: {len(rows)} rows")


def main():
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    customers = build_customers(rng)
    requests = build_requests(rng, customers)
    write_csv("customers", customers, ["customer_id", "segment", "region", "industry", "signup_date"])
    write_csv(
        "quote_requests",
        requests,
        ["request_id", "customer_id", "requested_at", "product_line", "channel", "quote_amount", "discount_pct", "converted_at"],
    )


if __name__ == "__main__":
    main()
