# ProjeCT360 Model Card

## Model Status

Version 2 training and untouched-test results are **pending execution in Google Colab**. This document intentionally contains no invented performance numbers. After training, use `model_manifest.json`, `final_test_results.json`, and `model_comparison.csv` as the source of truth.

## Intended Use

ProjeCT360 is an educational portfolio project for exploring aggregate patterns in historically reported Connecticut incidents. It estimates daily city-level reported-incident volume and probability distributions over broad and specific reported-offense categories from a selected city, date, hour, and location category.

## Prohibited Uses

- Person-level prediction or profiling
- Address-level targeting
- Enforcement, patrol, sentencing, or resource-allocation decisions
- Claims that a person, neighborhood, or city is dangerous
- Replacement of legal, professional, or community judgment

## Data

- Period in the current source file: 2016-01-01 through 2024-12-31
- Geography: Connecticut reporting agencies represented in the source data
- Incident source: FBI Crime Data Explorer and Connecticut public data used by the project
- Staffing and population fields: descriptive context and optional model features subject to ablation; annual values are delayed until the following calendar year

The data records reported incidents, not all harm that occurred. Missing reports, reporting access, enforcement activity, agency coverage, and data-quality differences affect observed patterns.

## Targets

- Volume: nonnegative daily count of reported incidents for a city
- Broad classification: Other, Property, or Violent grouping
- Specific classification: training-supported offense categories after a training-only rare-label mapping

## Evaluation

- Chronological training, validation, and untouched final-test periods
- App-aligned recursive 30-day rolling-origin validation before the final test
- Hidden validation/test outcomes are never fed into later days of a forecast trajectory
- A validation-selected model weight blends the trained predictor with a seven-day rolling baseline; weight zero is allowed when the model adds no validation value
- Meaningful naive/frequency and linear baselines
- Calibration fitted on validation only
- Forecast metrics: MAE, RMSE, WAPE, SMAPE, Poisson deviance, bias, and subgroup errors
- Classification metrics: macro/weighted F1, balanced and top-k accuracy, log loss, calibration error, Brier score, class/city/year/location-type results

## Limitations and Bias

Historical reporting can differ across communities and time. Staffing and crime-rate variables may reflect enforcement intensity. Class imbalance makes uncommon categories harder to estimate. Future distribution shifts, unknown locations, and missing data reduce confidence. A probability is model uncertainty conditioned on historical data, not certainty that an event will occur.

The deployed app uses a compact set of descriptive aggregates for charts and selectors. The incident-level CSV remains training data and is not downloaded into Streamlit after version 2 artifacts are uploaded.

## Final Model and Metrics

Pending the locked Colab final-test cell. Do not replace this section with validation metrics or hand-entered numbers; link or summarize generated manifest values only after verification.

The one-step forecast table in the notebook is diagnostic only. Forecast model selection, blend selection, and the reported final forecast metrics use complete recursive 30-day trajectories because that is how the Streamlit app is used.

Forecast and classification use independent validation gates. A task that fails its strongest validation baseline is not opened on the final test or exported as an improvement, but it does not prevent the other task from producing its own locked evaluation and poster diagnostics.
