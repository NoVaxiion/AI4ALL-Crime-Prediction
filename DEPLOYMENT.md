# ProjeCT360 Training and Deployment

## Train in Colab

1. Place or clone this repository at the notebook's `PROJECT_ROOT`.
2. Open `gc_train_model.ipynb` in Google Colab.
3. Set `DATA_PATH`, `OUTPUT_DIR`, `FINAL_TEST_START_DATE`, and development/full mode.
4. Keep `RUN_FINAL_TEST=False`, run through classification validation, and inspect the recursive 30-day forecast comparison, selected blend weight, and classification tables.
5. Repeat with full tuning if the development run is healthy. Freeze all choices before viewing the final test.
6. Set `RUN_FINAL_TEST=True` and run the final-test cell once. Export is blocked unless both forecast and classification validation gates pass.
7. Run export and compatibility verification. The verification cell confirms the feature schema, forecast mode, model weight, and 30-day horizon expected by the app.

The one-step forecast table is only a diagnostic. Use the recursive 30-day table to decide whether the model improves on the seven-day rolling baseline. A selected model weight of `0.0` honestly means the trained model added no validation value; the forecast task stays rejected, while classification may continue to its own evaluation.

## Upload to Hugging Face

Set `UPLOAD_TO_HUGGING_FACE=True` and use a temporary write token in Colab. The notebook uploads the selected models, `app_data_bundle.pkl`, reports, and `model_manifest.json`. It also removes obsolete per-city model files from the deployment repository after the locked evaluation succeeds.

The Streamlit deployment needs these secrets:

```toml
HF_REPO_ID = "NoVaxiion/project360-assets"
HF_REPO_TYPE = "dataset"
HF_TOKEN = "a read-only token for the private repository"
```

Public Hugging Face repositories work without `HF_TOKEN`. `data.py` caches downloads and supports a local ignored `Models/` directory for development.

`app_data_bundle.pkl` contains only the daily city totals, officer trends, historical offense counts, years, and location names needed by the unchanged dashboard. This prevents Streamlit from downloading and retaining the complete incident-level CSV. Until version 2 is uploaded, the app keeps the existing CSV fallback.

## Optional per-city models

Do not upload or load the two monolithic per-city dictionaries in Streamlit. Split them once locally:

```bash
python scripts/split_per_city_assets.py
hf upload NoVaxiion/project360-assets Models/per_city per_city --repo-type dataset
```

This creates an index plus one file per city. The app downloads only the forecasters for the currently selected cities and only a matching city classifier when Risk Analysis is requested. Cities without a dedicated classifier continue to use the statewide model.

For the current legacy models, also replace the 298 MB runtime CSV with its equivalent compact summaries and lookup tables:

```bash
python scripts/build_app_data_bundle.py
hf upload NoVaxiion/project360-assets Models/app_data_bundle.pkl app_data_bundle.pkl --repo-type dataset
```

The generated bundle is approximately 1 MB and retains the city histories, staffing charts, offense distributions, selectors, and historical lookup values consumed by the current dashboard. The full CSV remains the training source but is no longer downloaded by Streamlit.

## Deploy Streamlit

- Python runtime: 3.11
- Entry point: `app.py`
- Install from `requirements.txt`
- Light deployment is the default. It skips the monolithic per-city dictionaries while still allowing indexed, individually downloaded city models. Set `PROJECT360_DEPLOY_LIGHT=false` only to load the old dictionaries on a local machine with enough memory.
- Never commit `Models/`, `combined_data.csv`, backup ZIP files, or model binaries to normal Git history

## Verify

Run locally before pushing:

```bash
python -m pytest -q
python -m py_compile app.py data.py predict.py feature_engineering.py training_pipeline.py
```
