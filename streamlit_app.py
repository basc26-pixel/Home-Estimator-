"""Streamlit app for the Enhanced House Price Predictor."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import streamlit as st

from house_price_predictor import (
    DEFAULT_FEATURES,
    Dependencies,
    ModelResults,
    clean_numeric_columns,
    identify_price_column,
    load_data,
    load_dependencies,
    optimize_weights,
    prepare_features,
    predict_final_price,
    train_models,
)


@st.cache_resource
def get_dependencies() -> Dependencies:
    """Load third-party dependencies once per Streamlit session."""

    return load_dependencies()


def load_dataset(data_source: Any, sheet_name: str) -> Any:
    """Load the dataset from disk or an uploaded file."""

    deps = get_dependencies()
    return load_data(deps.pd, data_source, sheet_name)


def train_pipeline(
    data_source: Any,
    sheet_name: str,
    test_size: float,
    random_state: int,
) -> Tuple[
    Dependencies,
    ModelResults,
    Sequence[str],
    Mapping[str, float],
    float,
    float,
    float,
    float,
    Any,
    str,
]:
    """Train the ensemble pipeline and return the key artefacts."""

    deps = get_dependencies()
    df = load_dataset(data_source, sheet_name)
    price_column = identify_price_column(df.columns)
    numeric_columns = list(DEFAULT_FEATURES) + [price_column]
    df_clean = clean_numeric_columns(deps.pd, df, numeric_columns)
    df_clean = df_clean.dropna(subset=[price_column])

    if len(df_clean) < 3:
        raise ValueError("Not enough valid rows after cleaning the dataset.")

    X, y, available_features = prepare_features(df_clean, price_column)
    if not available_features:
        raise ValueError(
            "No usable feature columns were found in the uploaded dataset."
        )

    feature_defaults = {
        feature: float(df_clean[feature].median()) for feature in available_features
    }
    results = train_models(deps, X, y, test_size, random_state)
    (
        optimal_rf_weight,
        optimal_lr_weight,
        best_mae,
        combined_r2,
    ) = optimize_weights(deps, results)
    return (
        deps,
        results,
        available_features,
        feature_defaults,
        optimal_rf_weight,
        optimal_lr_weight,
        best_mae,
        combined_r2,
        df_clean,
        price_column,
    )


def show_model_metrics(
    results: ModelResults,
    optimal_rf_weight: float,
    optimal_lr_weight: float,
    best_mae: float,
    combined_r2: float,
) -> None:
    """Display a comparison of the trained models."""

    st.subheader("Model performance")
    metric_rows = [
        {
            "Model": "Random Forest",
            "MAE ($)": f"{results.rf_mae:,.2f}",
            "R²": f"{results.rf_r2:.3f}",
            "Weight": "100%",
        },
        {
            "Model": "Linear Regression",
            "MAE ($)": f"{results.lr_mae:,.2f}",
            "R²": f"{results.lr_r2:.3f}",
            "Weight": "100%",
        },
        {
            "Model": "Combined",
            "MAE ($)": f"{best_mae:,.2f}",
            "R²": f"{combined_r2:.3f}",
            "Weight": f"RF {optimal_rf_weight:.0%} / LR {optimal_lr_weight:.0%}",
        },
    ]
    st.dataframe(metric_rows, use_container_width=True)

    st.metric(
        "Recommended model",
        "Combined ensemble",
        help="Uses the optimized blend of Random Forest and Linear Regression",
    )


def render_prediction_form(
    deps: Dependencies,
    results: ModelResults,
    available_features: Sequence[str],
    feature_defaults: Mapping[str, float],
    optimal_rf_weight: float,
    optimal_lr_weight: float,
    best_mae: float,
    combined_r2: float,
) -> None:
    """Render the interactive prediction form."""

    st.subheader("Predict a house price")
    with st.form("prediction_form"):
        feature_values: Dict[str, float] = {}
        for feature in available_features:
            feature_values[feature] = st.number_input(
                feature,
                value=float(feature_defaults.get(feature, 0.0)),
                format="%.2f",
            )
        submitted = st.form_submit_button("Estimate price")

    if not submitted:
        return

    final_price, rf_price, lr_price = predict_final_price(
        feature_values,
        available_features,
        results.rf_model,
        results.lr_model,
        optimal_rf_weight,
        optimal_lr_weight,
    )

    st.success(f"Estimated price: ${final_price:,.2f}")
    st.write(
        f"Random Forest: ${rf_price:,.2f} | "
        f"Linear Regression: ${lr_price:,.2f} | "
        f"Weights → RF: {optimal_rf_weight:.0%}, LR: {optimal_lr_weight:.0%}"
    )

    lower_bound = final_price - best_mae
    upper_bound = final_price + best_mae
    st.caption(
        f"Expected error ±${best_mae:,.2f}. Estimated range: ${lower_bound:,.2f} – ${upper_bound:,.2f}. "
        f"Combined R²: {combined_r2:.3f}."
    )

    if hasattr(results.rf_model, "feature_importances_"):
        st.write("### Feature importance (Random Forest)")
        importance_df = deps.pd.DataFrame(
            {
                "feature": available_features,
                "importance": results.rf_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        st.bar_chart(importance_df.set_index("feature"))


def main() -> None:
    st.set_page_config(page_title="House Price Estimator", page_icon="🏠", layout="wide")
    st.title("🏠 Enhanced House Price Predictor")
    st.write(
        "Upload your dataset and explore predictions using a blended Random Forest + "
        "Linear Regression ensemble."
    )

    st.sidebar.header("Configuration")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel data",
        type=["xlsx"],
        help="Provide a workbook that matches the training schema.",
    )
    data_source = st.sidebar.text_input(
        "Or use a data file path",
        value="your_house_data.xlsx",
        help="Fallback path to an Excel workbook on the server.",
    )
    sheet_name = st.sidebar.text_input("Sheet name", value="Table1")
    test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

    data_handle: Optional[Any] = uploaded_file if uploaded_file is not None else data_source
    if not data_handle:
        st.info("Upload a dataset or provide a valid file path to begin training.")
        return

    try:
        (
            deps,
            results,
            available_features,
            feature_defaults,
            optimal_rf_weight,
            optimal_lr_weight,
            best_mae,
            combined_r2,
            cleaned_data,
            price_column,
        ) = train_pipeline(data_handle, sheet_name, test_size, random_state)
    except FileNotFoundError:
        st.error(
            "Data file not found. Upload your dataset to the project directory or adjust the path in the sidebar."
        )
        return
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # pragma: no cover - streamlit runtime errors
        st.error(f"Failed to train models: {exc}")
        return

    st.success("Models trained successfully! Adjust the sidebar to retrain with different parameters.")

    st.subheader("Training data preview")
    preview_columns = list(available_features) + [price_column]
    st.dataframe(cleaned_data[preview_columns].head(), use_container_width=True)
    st.caption(
        f"Rows used for training: {len(cleaned_data)} | Feature columns: {len(available_features)}"
    )

    show_model_metrics(results, optimal_rf_weight, optimal_lr_weight, best_mae, combined_r2)
    render_prediction_form(
        deps,
        results,
        available_features,
        feature_defaults,
        optimal_rf_weight,
        optimal_lr_weight,
        best_mae,
        combined_r2,
    )


if __name__ == "__main__":
    main()
