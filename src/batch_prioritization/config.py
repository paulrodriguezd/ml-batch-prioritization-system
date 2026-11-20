"""Dates and windows that define the problem. Changing them changes the metric."""

from datetime import datetime

# Extraction date: nothing after this moment is known.
AS_OF = datetime(2026, 1, 1)

# Target: the request converts on its own within this many days.
LABEL_DAYS = 5

# Time split. Each period's labels must be fully known before the next
# period starts, so a gap of LABEL_DAYS is left before each boundary.
VALID_START = datetime(2025, 7, 1)
TEST_START = datetime(2025, 10, 1)

SEED = 42
