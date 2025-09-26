"""Streamlit UI for the pre-trained linear regression house price estimator."""
from __future__ import annotations

from typing import Mapping

import streamlit as st

from house_price_predictor import (
    DEFAULT_FEATURES,
    FEATURE_DEFAULTS,
    LINEAR_INTERCEPT,
    TRAINING_MAE,
    TRAINING_R2,
    feature_contributions,
    predict_price,
)


def _format_feature_name(name: str) -> str:
    """Create a friendlier label for Streamlit widgets."""

    readable = name.replace("dc", "").replace("in", "").replace("sg", "")
    readable = readable.replace("M2", " m²")
    parts = []
    buffer = ""
    for char in readable:
        if char.isupper() and buffer:
            parts.append(buffer)
            buffer = char
        else:
            buffer += char
    if buffer:
        parts.append(buffer)
    return " ".join(part.capitalize() for part in parts)


def gather_feature_inputs() -> Mapping[str, float]:
    """Render numeric inputs for each feature and return their values."""

    values = {}
    for feature in DEFAULT_FEATURES:
        label = _format_feature_name(feature)
        default = float(FEATURE_DEFAULTS.get(feature, 0.0))
        values[feature] = st.number_input(
            label,
            value=default,
            min_value=0.0,
            format="%.2f",
        )
    return values


def show_prediction(feature_values: Mapping[str, float]) -> None:
    """Display the estimated price along with contribution details."""

    estimated_price = predict_price(feature_values)
    st.success(f"Estimated price: ${estimated_price:,.2f}")
    st.caption(
        f"Expected error ±${TRAINING_MAE:,.2f} (training MAE). "
        f"Training R²: {TRAINING_R2:.3f}."
    )

    contributions = feature_contributions(feature_values)
    st.write("### Feature contribution breakdown")
    st.dataframe(
        {
            "Feature": list(contributions.keys()),
            "Contribution ($)": [f"{value:,.2f}" for value in contributions.values()],
        },
        use_container_width=True,
    )

    st.write("### Contribution chart")
    st.bar_chart({feature: value for feature, value in contributions.items()})

    st.caption(
        "Intercept contributes "
        f"${LINEAR_INTERCEPT:,.2f} before considering the feature adjustments."
    )


def main() -> None:
    st.set_page_config(page_title="House Price Estimator", page_icon="🏠", layout="wide")
    st.title("🏠 House Price Estimator")
    st.write(
        "This tool uses a linear regression model trained on the original Excel "
        "dataset. The coefficients are baked into the app, so you no longer need "
        "to upload or provide the spreadsheet."
    )

    with st.form("prediction_form"):
        feature_values = gather_feature_inputs()
        submitted = st.form_submit_button("Estimate price")

    if submitted:
        show_prediction(feature_values)


if __name__ == "__main__":  # pragma: no cover - Streamlit entry point
    main()
