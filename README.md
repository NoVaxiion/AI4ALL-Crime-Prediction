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

The project uses public Connecticut incident data and officer staffing context to train and support several model-driven views.

The current app uses:

- feature engineering for date, seasonality, holidays, location context, city context, and staffing context
- regression models for incident volume forecasting
- classification models for broad and specific offense category probability estimates
- historical aggregation for city, location, officer, and offense-type comparisons
- Streamlit and Plotly for interactive visualization

Earlier research and experimentation included multiple modeling approaches, class imbalance handling, and model evaluation workflows. The final app focuses on the models and artifacts required for the current dashboard experience.

---

## Project Structure

```text
app.py              Streamlit user interface and charts
data.py             Data loading, aggregation, and lookup-table helpers
predict.py          Forecasting, feature engineering, and risk prediction helpers
requirements.txt    Python dependencies
Models/             Runtime model and data artifacts
.streamlit/         Streamlit theme configuration
tests/              Unit tests for forecast and feature-engineering logic
```

---

## Data Sources

- FBI Crime Data Explorer: Crime Incident-Based Data by State for Connecticut.
- Connecticut public data sources used for population and police staffing context.

FBI Crime Data Explorer:

https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/downloads

---

## Video Demo

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
