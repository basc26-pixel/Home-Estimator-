# Home Estimator Codebase Overview

## Repository layout
- `house_price_predictor.py` — Pre-trained linear regression estimator with a CLI experience.
- `streamlit_app.py` — Streamlit UI that reuses the same baked-in coefficients for quick experimentation.

## How it works
1. **Pre-trained parameters** — The linear regression weights, intercept, training medians, and performance benchmarks are stored directly in code so the original Excel workbook is no longer required. 【F:house_price_predictor.py†L28-L82】
2. **Prediction helpers** — `predict_price` performs the dot product between user supplied values and the stored weights, while `feature_contributions` exposes a per-feature attribution breakdown. 【F:house_price_predictor.py†L108-L132】
3. **CLI workflow** — Users can supply feature values via command-line flags or interactively. Defaults fall back to the historical medians, and the tool reports the estimate alongside training MAE/R² figures. 【F:house_price_predictor.py†L134-L206】
4. **Streamlit workflow** — The web UI mirrors the CLI: sliders default to the same medians, the estimate appears instantly, and a contribution table/chart explain the prediction. 【F:streamlit_app.py†L1-L86】

## Getting started
1. Ensure Python 3 with Streamlit installed (`pip install streamlit`).
2. Run the CLI estimator directly: `python house_price_predictor.py`.
   - Pass `--no-interactive` plus feature flags for non-interactive usage.
3. Launch the Streamlit UI: `streamlit run streamlit_app.py`.

## Suggested next steps for contributors
- **Expand test coverage** — Add fixtures covering additional feature combinations or edge-case values.
- **Persist scenario presets** — Allow saving/loading named feature configurations for quick comparisons.
- **Refine explainability** — Incorporate sensitivity analyses or partial dependence visuals based on the static weights.

