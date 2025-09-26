"""Command-line utility for training and using a house price ensemble model."""
from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_FEATURES: Sequence[str] = (
    "inStoreys",
    "inBedrooms",
    "inBathrooms",
    "inCarSpaces",
    "dcTotalAreaM2",
    "dcHouseLength",
    "dcHouseWidth",
    "dcGroundFloorArea",
    "dcAlfrescoArea",
    "dcPorchArea",
    "dcGarageArea",
)

PROMPTS: Dict[str, str] = {
    "inStoreys": "Number of storeys (floors): ",
    "inBedrooms": "Number of bedrooms: ",
    "inBathrooms": "Number of bathrooms: ",
    "inCarSpaces": "Number of car spaces: ",
    "dcTotalAreaM2": "Total area in square meters: ",
    "dcHouseLength": "House length (meters): ",
    "dcHouseWidth": "House width (meters): ",
    "dcGroundFloorArea": "Ground floor area (m²): ",
    "dcAlfrescoArea": "Alfresco area (m²): ",
    "dcPorchArea": "Porch area (m²): ",
    "dcGarageArea": "Garage area (m²): ",
}


@dataclass
class Dependencies:
    """Container holding third-party dependencies used by the script."""

    np: Any
    pd: Any
    RandomForestRegressor: Any
    LinearRegression: Any
    mean_absolute_error: Any
    r2_score: Any
    train_test_split: Any


@dataclass
class ModelResults:
    """Holds trained models and metrics for downstream usage."""

    rf_model: Any
    lr_model: Any
    rf_predictions: Any
    lr_predictions: Any
    y_test: Any
    rf_mae: float
    lr_mae: float
    rf_r2: float
    lr_r2: float


def ensure_dependency(module: str, install_hint: str) -> Any:
    """Import a dependency, raising a friendly error if it is missing."""

    try:
        return importlib.import_module(module)
    except ModuleNotFoundError:
        raise SystemExit(
            "❌ Missing required dependency '"
            + module
            + f"'. Install it with `pip install {install_hint}`."
        ) from None


def load_dependencies() -> Dependencies:
    """Load and bundle all third-party dependencies used in the script."""

    np = ensure_dependency("numpy", "numpy")
    pd = ensure_dependency("pandas", "pandas")
    # Ensure the default Excel engine for .xlsx files is available before reading data.
    ensure_dependency("openpyxl", "openpyxl")
    ensemble = ensure_dependency("sklearn.ensemble", "scikit-learn")
    linear_model = ensure_dependency("sklearn.linear_model", "scikit-learn")
    metrics = ensure_dependency("sklearn.metrics", "scikit-learn")
    model_selection = ensure_dependency("sklearn.model_selection", "scikit-learn")
    return Dependencies(
        np=np,
        pd=pd,
        RandomForestRegressor=ensemble.RandomForestRegressor,
        LinearRegression=linear_model.LinearRegression,
        mean_absolute_error=metrics.mean_absolute_error,
        r2_score=metrics.r2_score,
        train_test_split=model_selection.train_test_split,
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Enhanced house price predictor")
    parser.add_argument(
        "--data",
        default="your_house_data.xlsx",
        help="Path to the Excel file containing house data.",
    )
    parser.add_argument(
        "--sheet",
        default="Table1",
        help="Sheet name inside the Excel file.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for testing (between 0 and 1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip the interactive prediction loop.",
    )
    return parser.parse_args(argv)


def load_data(pd: Any, data_path: str, sheet_name: str) -> Any:
    """Load raw data from the Excel file."""

    try:
        df = pd.read_excel(data_path, sheet_name=sheet_name)
    except ImportError as exc:
        raise SystemExit(
            "❌ Missing Excel dependency 'openpyxl'. Install it with `pip install openpyxl`."
        ) from None
    except FileNotFoundError as exc:  # pragma: no cover - interactive feedback
        print(f"❌ File '{data_path}' not found!", file=sys.stderr)
        raise exc

    print(f"✅ Loaded {len(df)} rows from {data_path}")
    return df


def identify_price_column(columns: Iterable[str]) -> str:
    """Identify the price column from a list of available columns."""

    potential = {" Price", "Price"}
    for column in columns:
        if column in potential:
            print(f"Using price column: '{column}'")
            return column
    raise KeyError("❌ Price column not found!")


def clean_numeric_columns(pd: Any, df: Any, columns: Sequence[str]) -> Any:
    """Convert selected columns to numeric values and fill NaNs with medians."""

    df_clean = df.copy()
    for column in columns:
        if column not in df_clean.columns:
            continue
        df_clean[column] = pd.to_numeric(df_clean[column], errors="coerce")
        df_clean[column] = df_clean[column].fillna(df_clean[column].median())
    return df_clean


def prepare_features(df: Any, price_column: str) -> Tuple[Any, Any, List[str]]:
    """Prepare the feature matrix and target vector from the dataframe."""

    available_features = [col for col in DEFAULT_FEATURES if col in df.columns]
    print(f"✅ Using {len(available_features)} features")
    X = df[available_features]
    y = df[price_column]
    return X, y, available_features


def train_models(
    deps: Dependencies, X: Any, y: Any, test_size: float, random_state: int
) -> ModelResults:
    """Train both the random forest and linear regression models."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    X_train, X_test, y_train, y_test = deps.train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    print(
        f"📊 Training with {len(X_train)} houses, testing with {len(X_test)} houses"
    )

    print("\n" + "=" * 50)
    print("TRAINING BOTH MODELS")
    print("=" * 50)

    print("🌳 Training Random Forest...")
    rf_model = deps.RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_mae = deps.mean_absolute_error(y_test, rf_predictions)
    rf_r2 = deps.r2_score(y_test, rf_predictions)

    print("📈 Training Linear Regression...")
    lr_model = deps.LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    lr_mae = deps.mean_absolute_error(y_test, lr_predictions)
    lr_r2 = deps.r2_score(y_test, lr_predictions)

    return ModelResults(
        rf_model=rf_model,
        lr_model=lr_model,
        rf_predictions=rf_predictions,
        lr_predictions=lr_predictions,
        y_test=y_test,
        rf_mae=rf_mae,
        lr_mae=lr_mae,
        rf_r2=rf_r2,
        lr_r2=lr_r2,
    )


def optimize_weights(deps: Dependencies, results: ModelResults) -> Tuple[float, float, float, float]:
    """Optimize the ensemble weights using a simple grid search."""

    print("\n🔧 Optimizing model weights...")
    best_mae = float("inf")
    best_weights = (0.5, 0.5)

    for rf_weight in deps.np.arange(0.1, 1.0, 0.1):
        lr_weight = 1.0 - rf_weight
        current_predictions = (
            results.rf_predictions * rf_weight
            + results.lr_predictions * lr_weight
        )
        current_mae = deps.mean_absolute_error(results.y_test, current_predictions)
        if current_mae < best_mae:
            best_mae = current_mae
            best_weights = (rf_weight, lr_weight)

    optimal_rf_weight, optimal_lr_weight = best_weights
    combined_predictions = (
        results.rf_predictions * optimal_rf_weight
        + results.lr_predictions * optimal_lr_weight
    )
    combined_r2 = deps.r2_score(results.y_test, combined_predictions)
    return optimal_rf_weight, optimal_lr_weight, best_mae, combined_r2


def display_model_performance(
    results: ModelResults,
    best_mae: float,
    optimal_rf_weight: float,
    optimal_lr_weight: float,
    combined_r2: float,
) -> str:
    """Pretty print the model comparison and return the name of the best model."""

    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 50)

    print("🌳 RANDOM FOREST ONLY:")
    print(f"   - Mean Absolute Error: ${results.rf_mae:,.2f}")
    print(f"   - R² Score: {results.rf_r2:.4f}")

    print("📈 LINEAR REGRESSION ONLY:")
    print(f"   - Mean Absolute Error: ${results.lr_mae:,.2f}")
    print(f"   - R² Score: {results.lr_r2:.4f}")

    print("🤖 COMBINED MODEL (Optimized):")
    print(f"   - Mean Absolute Error: ${best_mae:,.2f}")
    print(f"   - R² Score: {combined_r2:.4f}")
    print(
        "   - Optimal Weights: Random Forest "
        f"{optimal_rf_weight:.0%}, Linear Regression {optimal_lr_weight:.0%}"
    )

    models = {
        "Random Forest": results.rf_mae,
        "Linear Regression": results.lr_mae,
        "Combined": best_mae,
    }
    best_method = min(models, key=models.get)
    print(f"\n🎯 RECOMMENDED: {best_method} (Lowest Error)")
    return best_method


def predict_final_price(
    house_features: Dict[str, float],
    available_features: Sequence[str],
    rf_model: Any,
    lr_model: Any,
    rf_weight: float,
    lr_weight: float,
) -> Tuple[float, float, float]:
    """Predict a price using the combined model and return component predictions."""

    features_array = [house_features[feature] for feature in available_features]
    rf_price = rf_model.predict([features_array])[0]
    lr_price = lr_model.predict([features_array])[0]
    final_price = (rf_price * rf_weight) + (lr_price * lr_weight)
    return final_price, rf_price, lr_price


def interactive_loop(
    deps: Dependencies,
    available_features: Sequence[str],
    results: ModelResults,
    optimal_rf_weight: float,
    optimal_lr_weight: float,
    best_mae: float,
    combined_r2: float,
) -> None:  # pragma: no cover - interactive loop
    """Run an interactive CLI for predicting additional house prices."""

    while True:
        print("\n" + "=" * 60)
        print("🏠 HOUSE PRICE PREDICTION")
        print("=" * 60)
        print("📝 Enter the features for your house (or type 'quit' to exit):")

        house_features: Dict[str, float] = {}
        for feature in available_features:
            while True:
                prompt = PROMPTS.get(feature, f"Enter {feature}: ")
                user_input = input(prompt)
                if user_input.lower() == "quit":
                    print("👋 Thank you for using the Enhanced House Price Predictor!")
                    return
                try:
                    value = float(user_input)
                except ValueError:
                    print("❌ Please enter a valid number!")
                    continue
                house_features[feature] = value
                break

        final_price, rf_price, lr_price = predict_final_price(
            house_features,
            available_features,
            results.rf_model,
            results.lr_model,
            optimal_rf_weight,
            optimal_lr_weight,
        )

        print("\n" + "💰" * 50)
        print("🎯 FINAL PREDICTED PRICE (Combined Model):")
        print(f"   ${final_price:,.2f}")
        print("💰" * 50)

        print("\n📊 Individual Model Predictions:")
        print(f"   🌳 Random Forest: ${rf_price:,.2f}")
        print(f"   📈 Linear Regression: ${lr_price:,.2f}")
        print(
            "   ⚖️  Weighted Average: "
            f"{optimal_rf_weight:.0%} RF + {optimal_lr_weight:.0%} LR"
        )

        print("\n📈 Accuracy Information:")
        print(f"   Expected Error Range: ±${best_mae:,.2f}")
        print(f"   Confidence: {max(0, combined_r2) * 100:.1f}%")

        lower_bound = final_price - best_mae
        upper_bound = final_price + best_mae
        print(
            "   📏 Estimated Price Range: "
            f"${lower_bound:,.2f} - ${upper_bound:,.2f}"
        )

        if hasattr(results.rf_model, "feature_importances_"):
            print("\n🔍 Most Important Features:")
            rf_importance = deps.pd.DataFrame(
                {
                    "feature": available_features,
                    "importance": results.rf_model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            for _, row in rf_importance.head(3).iterrows():
                print(f"   • {row['feature']}: {row['importance']:.1%} impact")

        print("\n" + "-" * 50)
        another = input("Would you like to predict another house? (y/n): ").lower()
        if another != "y":
            print("👋 Thank you for using the Enhanced House Price Predictor!")
            return


def main(argv: Sequence[str]) -> int:
    """Entry point for the command-line interface."""

    args = parse_args(argv)
    deps = load_dependencies()

    print("🏠 ENHANCED HOUSE PRICE PREDICTOR")
    print("🤖 Combined Random Forest + Linear Regression")
    print("=" * 60)

    try:
        df = load_data(deps.pd, args.data, args.sheet)
    except FileNotFoundError:
        return 1

    price_column = identify_price_column(df.columns)
    numeric_columns = list(DEFAULT_FEATURES) + [price_column]
    df_clean = clean_numeric_columns(deps.pd, df, numeric_columns)

    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=[price_column])
    removed_rows = initial_rows - len(df_clean)
    if removed_rows:
        print(f"⚠️ Removed {removed_rows} rows with invalid data")

    print(f"Clean data shape: {df_clean.shape}")
    if len(df_clean) < 3:
        print("❌ Not enough valid data after cleaning!")
        return 1

    X, y, available_features = prepare_features(df_clean, price_column)
    results = train_models(deps, X, y, args.test_size, args.random_state)
    (
        optimal_rf_weight,
        optimal_lr_weight,
        best_mae,
        combined_r2,
    ) = optimize_weights(deps, results)
    best_method = display_model_performance(
        results, best_mae, optimal_rf_weight, optimal_lr_weight, combined_r2
    )

    print("\n💾 Models trained successfully!")
    print(f"🌳 Random Forest Weight: {optimal_rf_weight:.0%}")
    print(f"📈 Linear Regression Weight: {optimal_lr_weight:.0%}")
    print(f"🎯 Combined Model Error: ${best_mae:,.2f}")
    print(f"✨ Recommended approach: {best_method}")

    if not args.no_interactive:
        interactive_loop(
            deps,
            available_features,
            results,
            optimal_rf_weight,
            optimal_lr_weight,
            best_mae,
            combined_r2,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main(sys.argv[1:]))
