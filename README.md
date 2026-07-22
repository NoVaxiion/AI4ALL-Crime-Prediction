## Video Demo
https://github.com/user-attachments/assets/1630c405-6d5f-4a0f-a8ad-bb4374b909bd


---

# CT-Crimes: AI4ALL Ignite Crime Prediction Project

CT-Crimes is a collaborative machine learning project developed during the AI4ALL Ignite Fellowship by Kenneth Maeda, Manushri Pendekanti, and Min Thaw Zin.

The project investigates how artificial intelligence can be used to explore, forecast, and communicate reported crime patterns across Connecticut using public historical data. The goal is not to replace human judgment, but to make complex public datasets easier to understand through responsible analytics, visualization, and model experimentation.

---

## Purpose and Outcomes

Crime and public safety data can be difficult to interpret because patterns vary across time, location, offense type, reporting practices, and community context. This project was built to make those patterns more visible through an interactive dashboard.

The project focuses on:

- exploring spatial and temporal incident-reporting patterns
- forecasting short-term incident volume trends
- estimating offense category probability distributions
- comparing police staffing and officer demographic trends
- communicating model limitations clearly and responsibly

Through this work, the team practiced the full machine learning lifecycle: data collection, cleaning, feature engineering, model training, evaluation, dashboard development, ethical reflection, and deployment preparation.

---

## Problem Statement

Crime remains a serious social issue that affects safety, resource planning, and community trust. At the same time, predictive tools for local crime analysis can be inaccessible, hard to interpret, or ethically risky when presented without context.

CT-Crimes addresses this gap as an educational analytics project. It uses historical Connecticut reporting data to build an interpretable dashboard for exploring trends, comparing cities, and understanding how machine learning models behave on public safety data.

---

## Current Dashboard

The Streamlit dashboard is organized into three main sections.

### Volume Forecasting

- Estimates expected daily incident counts for the next 30 days.
- Supports comparing the selected city with up to two additional cities.
- Marks holidays globally across all compared cities.
- Uses global and optional per-city forecasting models when available.
- Downloads indexed per-city models lazily for only the selected comparison cities, avoiding the old all-model startup allocation.

### Risk Analysis

- Shows broad offense category probabilities: Other, Property, and Violent.
- Shows top specific offense probability estimates.
- Allows filtering by city, date, hour, specific location area, and inferred location type.
- Supports comparison across selected cities.
- Includes visible methodology caveats because outputs reflect historical reporting patterns, not individual risk.

### Resource Analytics

- Shows statewide and city-specific police staffing trends.
- Visualizes total officer counts over time.
- Visualizes male and female officer counts over time.
- Displays latest officer totals, gender counts, and officer rate per 1,000 residents.

---

## Methodology

The project uses public Connecticut incident data and officer staffing context to train and support several model-driven views. Model development uses chronological training, validation, and untouched final-test periods rather than random splitting.

The current app uses:

- shared leakage-safe feature engineering for date, seasonality, holidays, location context, and city context
- complete city/date panels so forecast lags refer to earlier calendar days
- app-aligned recursive 30-day validation in which predictions, never hidden holdout outcomes, feed later forecast days
- validation-learned blending between the selected model and a seven-day rolling baseline, including a zero-model fallback when the model adds no value
- training-only category vocabularies, rare-label decisions, and historical lookup tables
- previous-complete-year handling for annual staffing, population, and rate fields
- count-regression comparisons for incident volume forecasting
- calibrated classification comparisons for broad and specific offense probability estimates
- rolling-origin validation, meaningful baselines, and one untouched chronological test period used only after all choices are frozen
- historical aggregation for city, location, officer, and offense-type comparisons
- Streamlit and Plotly for interactive visualization

The Colab notebook compares unsuccessful and successful experiments rather than reporting only the winning model. Final metrics are generated into the model manifest and reports; they are not hardcoded in the dashboard or README.

For deployment, Hugging Face stores compact selected models and a summarized dashboard data bundle. The full incident-level dataset is not downloaded by Streamlit after version 2 artifacts are uploaded.

---

## Project Structure

```text
app.py              Streamlit user interface and charts
data.py             Data loading, aggregation, and lookup-table helpers
predict.py          Forecasting and risk inference helpers
feature_engineering.py  Shared training/inference feature definitions
training_pipeline.py    Evaluation, baseline, checkpoint, and manifest helpers
scripts/split_per_city_assets.py  One-time splitter for memory-safe per-city deployment
scripts/build_app_data_bundle.py  Compact replacement for the runtime incident CSV
gc_train_model.ipynb    Google Colab training and untouched-test workflow
requirements.txt    Python dependencies
requirements-train.txt  Additional Colab-only training dependencies
Models/             Ignored local runtime artifacts; production assets use Hugging Face
.streamlit/         Streamlit theme configuration
tests/              Leakage, schema, deployment, and frontend-preservation tests
```

The full leakage findings are documented in `LEAKAGE_AUDIT.md`. Responsible-use details are in `MODEL_CARD.md`, and Colab/Hugging Face instructions are in `DEPLOYMENT.md`.

---

## Data Sources

- FBI Crime Data Explorer: Crime Incident-Based Data by State for Connecticut.
- Connecticut public data sources used for population and police staffing context.

FBI Crime Data Explorer:

https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/downloads

---

## Learning Outcomes

Through this project, the team gained experience across the machine learning and data science workflow:

- End-to-end ML development: designing, training, testing, and deploying models on structured public datasets.
- Feature engineering: transforming raw incident data into useful temporal, spatial, and contextual features.
- Model evaluation: comparing model behavior, checking for leakage, and validating whether predictions were meaningful.
- Data visualization: building an interactive dashboard for technical and non-technical audiences.
- Data ethics and bias awareness: considering how reporting bias, class imbalance, and historical patterns affect model outputs.
- Collaboration: building a shared technical project through the AI4ALL Ignite Fellowship.

---

## Important Limitations

This dashboard is for educational and exploratory analysis only.

It should not be used to:

- make enforcement decisions
- assess risk to any person
- justify increased policing
- replace professional, legal, or public policy judgment

Model outputs may reflect missing data, reporting bias, class imbalance, historical enforcement patterns, and limitations in the available data. Forecasts and probabilities should be interpreted as model-generated estimates from historical reporting data, not as statements about future real-world events or individual behavior.

---

## Authors

- Kenneth Maeda: https://github.com/NoVaxiion
- Manushri Pendekanti https://github.com/manushrip06
- Min Thaw Zin: https://github.com/Min-13

---

## Educational Use

Developed for educational and research purposes as part of the AI4ALL Ignite Fellowship, with an emphasis on responsible AI, interpretability, and careful communication of limitations.
