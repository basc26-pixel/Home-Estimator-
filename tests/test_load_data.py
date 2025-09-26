"""Tests for the load_data helper in house_price_predictor."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from house_price_predictor import load_data


class _DummyPandas:
    """Stub pandas module that simulates missing openpyxl dependency."""

    @staticmethod
    def read_excel(*args, **kwargs):
        raise ImportError("No module named 'openpyxl'")


def test_load_data_missing_openpyxl():
    """load_data should exit with a helpful message when openpyxl is unavailable."""

    with pytest.raises(SystemExit) as exc:
        load_data(_DummyPandas, "fake.xlsx", "Sheet1")

    message = str(exc.value)
    assert "pip install openpyxl" in message
    assert "openpyxl" in message
