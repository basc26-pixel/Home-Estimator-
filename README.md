# Home Estimator Codebase Overview

## Repository layout
- `house_price_predictor.py` — Main entry point that loads training data, prepares features, trains two estimators (Random Forest and Linear Regression), builds an ensemble, and drives an interactive CLI for predictions.
- `your_house_data.xlsx` — Expected Excel workbook containing the historical house data (specifically the `Table1` sheet).

## Execution flow
1. **Startup messaging and imports** — Announces the combined-model predictor and configures dependencies such as pandas, scikit-learn, and NumPy. Warnings are suppressed to keep the CLI output clean. 【F:house_price_predictor.py†L1-L15】
2. **Data loading** — Reads `Table1` from `your_house_data.xlsx`, handling a missing file with a friendly error message. 【F:house_price_predictor.py†L16-L24】
3. **Data cleaning & feature setup**
   - Detects which price column exists, converts relevant columns to numeric, fills missing values with medians, and removes rows with missing prices. 【F:house_price_predictor.py†L26-L63】
   - Builds the feature matrix from available numeric fields before splitting into train/test sets. 【F:house_price_predictor.py†L65-L84】
4. **Model training** — Fits a Random Forest and a Linear Regression model in parallel and captures their evaluation metrics. 【F:house_price_predictor.py†L85-L105】
5. **Ensembling & evaluation**
   - Defines a weighted-average helper (`combined_prediction`) and computes ensemble performance metrics. 【F:house_price_predictor.py†L106-L118】
   - Performs a simple grid search over the weight ratio to find the best-performing combination and reports model comparisons. 【F:house_price_predictor.py†L119-L153】
6. **Interactive CLI** — After training, the script enters a prompt-driven loop where users provide feature values, receive combined and individual model estimates, see an error range, and review top feature importances. 【F:house_price_predictor.py†L164-L254】
7. **Summary output** — Once the session ends, the script prints the tuned weights and combined error for future reference. 【F:house_price_predictor.py†L256-L262】

## Key concepts & components
- **Data schema assumptions** — The script expects the feature columns listed in `feature_columns`, but gracefully adapts if some are absent by using only the available subset. 【F:house_price_predictor.py†L65-L78】
- **Model ensemble strategy** — A straightforward weighted average of Random Forest and Linear Regression predictions; weights are optimized via brute-force search over 0.1 increments. 【F:house_price_predictor.py†L106-L134】
- **User interaction loop** — Uses prompts tailored to each feature, includes validation for numeric input, and allows users to exit cleanly by typing `quit`. 【F:house_price_predictor.py†L186-L214】
- **Explainability touchpoints** — Displays feature importances from the trained Random Forest model to highlight influential variables. 【F:house_price_predictor.py†L239-L248】

## Getting started
1. Ensure Python 3 with pandas, NumPy, matplotlib, and scikit-learn installed.
2. Populate `your_house_data.xlsx` with historical data following the expected column names.
3. Run the script: `python house_price_predictor.py` and follow the prompts to estimate a house price.

## Suggested next steps for contributors
- **Persist trained models** — Introduce serialization (e.g., `joblib`) so the ensemble can be reused without retraining each run.
- **Automate evaluation** — Add unit tests or a notebook showcasing model performance across different datasets.
- **Enhance UX** — Wrap the CLI in a simple GUI or web app, or provide command-line flags for batch predictions.
- **Feature engineering** — Explore additional derived metrics (e.g., price per square meter) or normalization techniques to improve accuracy.

