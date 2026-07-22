# ProjeCT360 Leakage and Evaluation Audit

## Status

The previous saved models remain legacy artifacts until `gc_train_model.ipynb` is run in Colab and the version 2 artifacts pass the locked final-test cell. Their historical metrics must not be presented as trustworthy final-test performance.

## Confirmed Problems

| Area | Previous behavior | Risk | Version 2 decision |
|---|---|---|---|
| Rare labels | Categories were selected from the final 90-day distribution | Final-test contamination | Fit and save the mapping from training labels only |
| Early stopping | The test period was supplied as `eval_set` | Test-set tuning | Use validation windows only |
| Calibration | Isotonic calibration was fitted on training rows and assessed on the repeatedly inspected test period | Improper calibration and optimistic selection | Fit on a dedicated validation slice and score on a later validation slice |
| `offense_group` | Used as a classifier input | Direct target leakage | Prohibited as a feature |
| Offense label columns | Used to construct row-shifted category lags | Target and neighboring-event leakage | Prohibited as inputs |
| `city_hour_common_crime` / `hour_typical_crime` | Mode computed over the complete dataset | Target-derived future leakage | Removed |
| `crime_diversity` | Counted same-day/hour outcome categories | Unavailable at prediction time | Removed |
| `location_daily_freq` | Counted all incidents for the target day | Same-day outcome leakage | Replaced by frozen training-period historical mean |
| `location_total_crimes` | Counted the complete dataset | Future leakage | Replaced by frozen training-period count |
| Forecast row lags | Some experiments shifted incident rows rather than complete calendar days | Incorrect lag meaning | Complete city/date panel, then calendar shifts |
| Rolling statistics | Not consistently isolated from the current row | Possible target leakage | Every rolling feature begins with `shift(1)` |
| Multi-day validation | One-step predictions used observed prior holdout values while the app recursively uses its own predictions | Evaluation/deployment mismatch and optimistic horizon estimates | Select models and blend weights on complete recursive 30-day origins; only prior-origin history is available |
| Category encoding | Independent category codes could drift between train/test/app | Inconsistent inference and arbitrary unknown mapping | Frozen training vocabulary, unknown code `-1` |
| SMOTE | Ordinary SMOTE interpolated nominal category codes | Invalid synthetic categories | Compare no resampling, class weights, undersampling, and valid SMOTENC |
| Weighting | Class weights and sample weights were both applied | Duplicate correction | Exactly one imbalance strategy per experiment |
| Hierarchy | Broad and specific models predicted independently | Not a true hierarchy | Describe honestly; only select routing if it beats the shared model |
| Per-city models | Large artifacts selected from one holdout despite weak results | Overfitting and deployment memory | Off by default; require repeated validation wins |
| Annual staffing/population/rate fields | Current-year summaries could include information finalized after the target date | Future aggregate leakage and enforcement/reporting proxy risk | Year Y values become available only in year Y+1; also compare a full resource-feature ablation |
| Unseen future offense labels | Labels absent from training were left unchanged | Encoder failure and implicit future label knowledge | Persist known training labels and map rare/unseen labels to a saved training-supported fallback (normally `Other`) |
| Broad target definition | Several person/property categories fell through to `Other` | Incorrect target semantics | Use one explicit, tested mapping shared by every training run |

## Feature Decisions

### Safe and known before prediction

- City, location area, inferred location type
- Calendar date, hour, weekday, month, seasonality, and Connecticut holidays
- Payday calendar indicators
- Forecast lags and rolling values calculated from earlier calendar dates
- Historical counts frozen at the training cutoff

### Safe only as frozen historical artifacts

- City frequency
- Location frequency and historical daily mean
- City-hour historical frequency
- Category vocabularies and rare-label mapping

### Removed because unsafe

- `offense_group`
- `offense_name`
- Current-event offense category fields
- Incident-row category lags
- `city_hour_common_crime`
- `hour_typical_crime`
- Same-day `crime_diversity`
- Same-day location frequency
- Any aggregate fitted with validation or final-test rows

### Bias-sensitive and subject to ablation

- `population`
- `total_officers`
- `officers_per_1000_people`
- `crime_rate_per_1000_people`

These variables are retained only if the validation ablation improves the documented selection metrics. Annual values are shifted to the next calendar year before use. Even when retained, they may encode reporting and enforcement differences rather than underlying harm.

## Evaluation Contract

1. Chronological training, validation, and untouched final-test periods are saved exactly.
2. Rolling-origin folds stay before the final test.
3. Model and preprocessing choices use validation only.
4. Forecast candidates are ranked on recursive 30-day MAE, not the diagnostic one-step table.
5. During recursive evaluation, predicted counts are appended to history; hidden holdout outcomes are never exposed to later horizons.
6. The model-versus-seven-day-baseline weight is selected on validation only and saved as artifact metadata.
7. The final test is unlocked once, after choices are frozen.
8. Validation and final-test metrics are stored under separate names.
9. Baselines use the same cities, origins, horizons, and target rows as candidate models.
10. No metric is copied into the UI or documentation by hand.
11. Version 2 Streamlit deployments load compact descriptive summaries rather than the incident-level training CSV.
