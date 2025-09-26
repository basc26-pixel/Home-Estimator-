"""House price estimator powered by a pre-trained linear regression model."""
from __future__ import annotations

import argparse
import sys
from typing import Dict, Mapping, MutableMapping, Sequence

# Ordered list of features expected by the model. Keeping the order ensures a
# consistent mapping when we dot-product the coefficients with user-provided
# values.
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

# Friendly prompts for the CLI experience.
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

# Linear regression parameters extracted from the historical Excel dataset.
LINEAR_INTERCEPT: float = 884_705.6325072331
LINEAR_COEFFICIENTS: Mapping[str, float] = {
    "inStoreys": -1.9485014490783215e-08,
    "inBedrooms": -1_973.0091819562947,
    "inBathrooms": 7_935.708990153786,
    "inCarSpaces": -2.0372681319713593e-10,
    "dcTotalAreaM2": -5_036.923200158048,
    "dcHouseLength": 7_024.642514593681,
    "dcHouseWidth": 10_353.325474032796,
    "dcGroundFloorArea": 5_131.470072730301,
    "dcAlfrescoArea": 5_174.5203472688345,
    "dcPorchArea": 4_305.901915665378,
    "dcGarageArea": -19_648.812960581832,
}

# Median values from the training data. These provide sensible defaults for the
# CLI and Streamlit experiences without requiring the original spreadsheet.
FEATURE_DEFAULTS: Mapping[str, float] = {
    "inStoreys": 1.0,
    "inBedrooms": 4.0,
    "inBathrooms": 2.0,
    "inCarSpaces": 2.0,
    "dcTotalAreaM2": 233.55,
    "dcHouseLength": 22.61,
    "dcHouseWidth": 11.15,
    "dcGroundFloorArea": 180.25,
    "dcAlfrescoArea": 10.9,
    "dcPorchArea": 6.2,
    "dcGarageArea": 36.4,
}

# Training metrics for the embedded model. These numbers give users a sense of
# expected accuracy without needing to re-train the estimator.
TRAINING_MAE: float = 1_891.1966878103183
TRAINING_R2: float = 0.9883236920813254


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Estimate house prices using the baked-in linear regression model. "
            "Provide features as flags or let the script prompt you interactively."
        )
    )
    for feature in DEFAULT_FEATURES:
        parser.add_argument(
            f"--{feature}",
            type=float,
            dest=feature,
            help=f"Value for {feature} (optional).",
        )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Do not prompt for missing values; instead require them all via flags.",
    )
    return parser.parse_args(argv)


def predict_price(
    feature_values: Mapping[str, float],
    *,
    coefficients: Mapping[str, float] | None = None,
    intercept: float | None = None,
) -> float:
    """Return the estimated price for the provided features."""

    coeffs = dict(LINEAR_COEFFICIENTS if coefficients is None else coefficients)
    bias = LINEAR_INTERCEPT if intercept is None else intercept
    price = bias
    for feature, weight in coeffs.items():
        price += weight * float(feature_values.get(feature, 0.0))
    return price


def feature_contributions(
    feature_values: Mapping[str, float],
    *,
    coefficients: Mapping[str, float] | None = None,
) -> Mapping[str, float]:
    """Return per-feature contributions to the final prediction."""

    coeffs = dict(LINEAR_COEFFICIENTS if coefficients is None else coefficients)
    return {
        feature: weight * float(feature_values.get(feature, 0.0))
        for feature, weight in coeffs.items()
    }


def _format_prompt(feature: str, default: float | None) -> str:
    prompt = PROMPTS.get(feature, f"Enter {feature}: ")
    if default is not None:
        prompt = f"{prompt}[{default}] "
    return prompt


def prompt_for_features(
    initial_values: Mapping[str, float] | None = None,
) -> Mapping[str, float] | None:
    """Interactively ask the user for feature values.

    The user can press Enter to accept defaults, or type ``quit`` to exit.
    """

    collected: MutableMapping[str, float] = {}
    base_values = initial_values or {}
    for feature in DEFAULT_FEATURES:
        default = base_values.get(feature, FEATURE_DEFAULTS.get(feature))
        while True:
            raw = input(_format_prompt(feature, default)).strip()
            if raw.lower() == "quit":
                return None
            if raw == "":
                if default is None:
                    print("❌ Please enter a value or type 'quit' to exit.")
                    continue
                collected[feature] = float(default)
                break
            try:
                collected[feature] = float(raw)
            except ValueError:
                print("❌ Please enter a valid number!")
                continue
            break
    return collected


def display_estimate(feature_values: Mapping[str, float]) -> None:
    """Print the model estimate and attribution breakdown."""

    price = predict_price(feature_values)
    contributions = feature_contributions(feature_values)
    sorted_contribs = sorted(
        contributions.items(), key=lambda item: abs(item[1]), reverse=True
    )

    print("\n" + "💰" * 40)
    print("🎯 Estimated price:")
    print(f"   ${price:,.2f}")
    print("💰" * 40)

    print("\n🔍 Contribution breakdown:")
    for feature, value in sorted_contribs:
        print(f"   • {feature}: ${value:,.2f}")
    print(f"   • Intercept: ${LINEAR_INTERCEPT:,.2f}")

    print("\n📈 Training performance benchmarks:")
    print(f"   - Mean Absolute Error: ±${TRAINING_MAE:,.2f}")
    print(f"   - R² Score: {TRAINING_R2:.3f}")


def interactive_loop(initial_values: Mapping[str, float] | None = None) -> None:  # pragma: no cover - relies on input
    """Repeatedly prompt the user for features and show predictions."""

    current_defaults = dict(initial_values or {})
    while True:
        features = prompt_for_features(current_defaults)
        if features is None:
            print("👋 Thank you for using the house price estimator!")
            return
        display_estimate(features)

        again = input("\nWould you like to estimate another house? (y/n): ").strip()
        if again.lower() != "y":
            print("👋 Thank you for using the house price estimator!")
            return
        current_defaults = features


def gather_flag_values(args: argparse.Namespace) -> Dict[str, float]:
    """Extract feature values supplied via command-line flags."""

    provided: Dict[str, float] = {}
    for feature in DEFAULT_FEATURES:
        value = getattr(args, feature)
        if value is not None:
            provided[feature] = float(value)
    return provided


def main(argv: Sequence[str]) -> int:
    """Entry point for the command-line interface."""

    args = parse_args(argv)
    provided = gather_flag_values(args)

    if args.no_interactive:
        missing = [feature for feature in DEFAULT_FEATURES if feature not in provided]
        if missing:
            print(
                "❌ Missing required features: "
                + ", ".join(missing)
                + ". Provide them via command-line flags or remove --no-interactive."
            )
            return 1
        display_estimate(provided)
        return 0

    print("🏠 Linear Regression House Price Estimator")
    print("📈 Model weights extracted from the original Excel workbook")
    print("=" * 60)
    interactive_loop(provided)
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main(sys.argv[1:]))
