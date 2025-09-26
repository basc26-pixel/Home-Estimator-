"""Unit tests for the baked-in linear regression estimator."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from house_price_predictor import (
    DEFAULT_FEATURES,
    FEATURE_DEFAULTS,
    LINEAR_COEFFICIENTS,
    LINEAR_INTERCEPT,
    TRAINING_MAE,
    TRAINING_R2,
    feature_contributions,
    gather_flag_values,
    predict_price,
)


def test_predict_price_matches_known_row() -> None:
    """The predictor should be close to the original spreadsheet value."""

    sample = {
        "inStoreys": 1.0,
        "inBedrooms": 4.0,
        "inBathrooms": 2.0,
        "inCarSpaces": 2.0,
        "dcTotalAreaM2": 251.2,
        "dcHouseLength": 22.91,
        "dcHouseWidth": 12.59,
        "dcGroundFloorArea": 196.0,
        "dcAlfrescoArea": 12.4,
        "dcPorchArea": 6.4,
        "dcGarageArea": 36.4,
    }
    predicted = predict_price(sample)
    assert abs(predicted - 301_100.0) < 2_500  # within training MAE margin


def test_feature_contributions_align_with_coefficients() -> None:
    """Each contribution should equal coefficient × feature value."""

    contributions = feature_contributions(FEATURE_DEFAULTS)
    for feature, coefficient in LINEAR_COEFFICIENTS.items():
        expected = coefficient * FEATURE_DEFAULTS[feature]
        assert contributions[feature] == expected


def test_gather_flag_values_filters_none() -> None:
    """Only populated CLI flags should surface in the returned dict."""

    class Args:
        inStoreys = 1.0
        inBedrooms = None
        inBathrooms = 2.0
        inCarSpaces = None
        dcTotalAreaM2 = 200.0
        dcHouseLength = None
        dcHouseWidth = None
        dcGroundFloorArea = 150.0
        dcAlfrescoArea = None
        dcPorchArea = None
        dcGarageArea = 30.0

    result = gather_flag_values(Args())
    assert result == {
        "inStoreys": 1.0,
        "inBathrooms": 2.0,
        "dcTotalAreaM2": 200.0,
        "dcGroundFloorArea": 150.0,
        "dcGarageArea": 30.0,
    }


def test_training_metrics_are_positive() -> None:
    """Embedded training metrics should reflect sensible values."""

    assert TRAINING_MAE > 0
    assert 0 <= TRAINING_R2 <= 1
    assert LINEAR_INTERCEPT != 0
    assert len(DEFAULT_FEATURES) == len(LINEAR_COEFFICIENTS)
