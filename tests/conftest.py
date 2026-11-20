import os
import time

import pytest

# Python turns naive datetimes into timestamps with the OS time zone.
# Pin it to UTC so test dates mean the same thing everywhere (and across
# daylight-saving changes), matching the Spark session time zone.
os.environ["TZ"] = "UTC"
time.tzset()

from batch_prioritization.spark import get_spark  # noqa: E402


@pytest.fixture(scope="session")
def spark():
    session = get_spark("tests")
    yield session
    session.stop()
