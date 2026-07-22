"""Leakage-safe feature definitions shared by training and Streamlit inference.

The functions in this module intentionally accept explicit training frames and
plain dictionaries. Saved artifacts therefore remain small, inspectable, and
portable between Google Colab and Streamlit Community Cloud.
"""

from __future__ import annotations

import calendar
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Iterable, Mapping

import holidays
import numpy as np
import pandas as pd


SCHEMA_VERSION = "2.1"
CT_HOLIDAYS = holidays.US(subdiv="CT")

TARGET_COLUMNS = {
    "crime_count",
    "offense_group",
    "offense_name",
    "offense_category_name",
    "offense_category_clean",
    "crime_category_l1",
    "crime_category_l2",
}

RESOURCE_FEATURES = [
    "population",
    "total_officers",
    "officers_per_1000_people",
    "crime_rate_per_1000_people",
]

FORECAST_FEATURES = [
    "city_code",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_365",
    "roll_mean_7",
    "roll_mean_14",
    "roll_mean_28",
    "roll_mean_90",
    "roll_std_28",
    "ewm_mean_7",
    "ewm_mean_28",
    "same_weekday_mean",
    "trend_28_90",
    "day_of_week",
    "month",
    "quarter",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "is_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
] + RESOURCE_FEATURES

CLASSIFICATION_FEATURES = [
    "city_code",
    "location_area_code",
    "location_type_code",
    "hour",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "dow_sin",
    "dow_cos",
    "month",
    "month_sin",
    "month_cos",
    "day_of_month",
    "day_of_month_sin",
    "day_of_month_cos",
    "week_of_year",
    "week_sin",
    "week_cos",
    "is_weekend",
    "is_holiday",
    "is_payday",
    "days_from_payday",
    "is_university",
    "city_train_count",
    "location_train_count",
    "location_train_daily_mean",
    "city_hour_train_count",
] + RESOURCE_FEATURES

CATEGORICAL_MODEL_FEATURES = ["city_code", "location_area_code", "location_type_code"]

FORECAST_PREDICTION_MODES = {"direct", "residual_to_roll_mean_7"}

UNIVERSITY_CITIES = {
    "Yale University",
    "Southern Connecticut State University",
    "Western Connecticut State University",
    "Eastern Connecticut State University",
    "Central Connecticut State University",
    "University of Connecticut:",
    "University of Connecticut: Trumbull",
}

VIOLENT_OFFENSES = {
    "Assault Offenses",
    "Homicide Offenses",
    "Human Trafficking",
    "Kidnapping/Abduction",
    "Robbery",
    "Sex Offenses",
    "Sex Offenses, Non-forcible",
}

PROPERTY_OFFENSES = {
    "Arson",
    "Burglary/Breaking & Entering",
    "Counterfeiting/Forgery",
    "Destruction/Damage/Vandalism of Property",
    "Embezzlement",
    "Extortion/Blackmail",
    "Fraud Offenses",
    "Larceny/Theft Offenses",
    "Motor Vehicle Theft",
    "Stolen Property Offenses",
}

LOCATION_TYPE_KEYWORDS = [
    ("commercial", ["office", "building", "bank", "company", "business"]),
    ("education", ["school", "college", "university", "campus"]),
    ("nightlife", ["bar", "restaurant", "nightclub", "club", "tavern"]),
    ("retail", ["store", "shop", "market", "mall", "retail"]),
    ("residential", ["residence", "home", "apartment", "house", "roof"]),
    ("street", ["street", "highway", "road", "alley", "sidewalk", "parking"]),
]


@dataclass(frozen=True)
class TemporalBoundaries:
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict[str, str]:
        return {key: str(pd.Timestamp(value).date()) for key, value in asdict(self).items()}


def infer_location_type(location_area: object) -> str:
    """Map a specific location label to the same broad type at train and inference time."""
    text = str(location_area).lower()
    for location_type, keywords in LOCATION_TYPE_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return location_type
    return "other"


def map_broad_category(offense_category: object) -> str:
    """Map the specific reported-offense target to an explicit broad target."""
    label = str(offense_category)
    if label in VIOLENT_OFFENSES:
        return "Violent"
    if label in PROPERTY_OFFENSES:
        return "Property"
    return "Other"


def get_payday_features(ts: object) -> tuple[int, int]:
    """Return month-start/mid-month/month-end payday indicators."""
    ts = pd.Timestamp(ts)
    last_day = calendar.monthrange(ts.year, ts.month)[1]
    paydays = (1, 15, last_day)
    return int(ts.day in paydays), min(abs(ts.day - payday) for payday in paydays)


def make_temporal_boundaries(
    dates: Iterable[object],
    final_test_start: object | None = None,
    validation_days: int = 90,
) -> TemporalBoundaries:
    """Create non-overlapping chronological train, validation, and test periods."""
    date_series = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
    if date_series.empty:
        raise ValueError("Cannot create temporal boundaries without valid dates.")
    test_end = date_series.max().normalize()
    test_start = (
        pd.Timestamp(final_test_start).normalize()
        if final_test_start is not None
        else test_end - pd.Timedelta(days=41)
    )
    validation_end = test_start - pd.Timedelta(days=1)
    validation_start = validation_end - pd.Timedelta(days=validation_days - 1)
    train_end = validation_start - pd.Timedelta(days=1)
    if train_end < date_series.min().normalize():
        raise ValueError("The requested validation and test windows leave no training period.")
    return TemporalBoundaries(train_end, validation_start, validation_end, test_start, test_end)


def temporal_masks(dates: Iterable[object], boundaries: TemporalBoundaries) -> dict[str, pd.Series]:
    date_series = pd.to_datetime(pd.Series(dates), errors="coerce")
    masks = {
        "train": date_series <= boundaries.train_end,
        "validation": date_series.between(boundaries.validation_start, boundaries.validation_end),
        "test": date_series.between(boundaries.test_start, boundaries.test_end),
    }
    if (masks["train"] & masks["validation"]).any() or (masks["validation"] & masks["test"]).any():
        raise AssertionError("Temporal periods overlap.")
    return masks


def rolling_origin_boundaries(
    train_end: object,
    minimum_date: object,
    validation_days: int = 28,
    folds: int = 3,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Return expanding-window folds that stay strictly before the final validation period."""
    end = pd.Timestamp(train_end).normalize()
    minimum = pd.Timestamp(minimum_date).normalize()
    windows = []
    for offset in reversed(range(folds)):
        fold_valid_end = end - pd.Timedelta(days=offset * validation_days)
        fold_valid_start = fold_valid_end - pd.Timedelta(days=validation_days - 1)
        fold_train_end = fold_valid_start - pd.Timedelta(days=1)
        if fold_train_end > minimum:
            windows.append((fold_train_end, fold_valid_start, fold_valid_end))
    return windows


def fit_category_vocabulary(values: Iterable[object]) -> list[str]:
    return sorted(pd.Series(values, dtype="string").dropna().unique().tolist())


def encode_with_vocabulary(values: Iterable[object], vocabulary: list[str]) -> pd.Series:
    mapping = {value: index for index, value in enumerate(vocabulary)}
    return pd.Series(values, dtype="string").map(mapping).fillna(-1).astype("int32")


def assert_safe_features(feature_columns: Iterable[str]) -> None:
    leaked = sorted(TARGET_COLUMNS.intersection(feature_columns))
    if leaked:
        raise ValueError(f"Target-derived columns are not valid prediction features: {leaked}")


def add_calendar_features(frame: pd.DataFrame, date_col: str = "date", include_hour: bool = False) -> pd.DataFrame:
    """Add deterministic calendar features known before the predicted event."""
    data = frame.copy()
    dates = pd.to_datetime(data[date_col], errors="coerce")
    data["day_of_week"] = dates.dt.dayofweek.astype("int8")
    data["month"] = dates.dt.month.astype("int8")
    data["quarter"] = dates.dt.quarter.astype("int8")
    data["day_of_year"] = dates.dt.dayofyear.astype("int16")
    data["day_of_month"] = dates.dt.day.astype("int8")
    data["week_of_year"] = dates.dt.isocalendar().week.astype("int16")
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype("int8")
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
    data["day_of_month_sin"] = np.sin(2 * np.pi * data["day_of_month"] / 31)
    data["day_of_month_cos"] = np.cos(2 * np.pi * data["day_of_month"] / 31)
    data["week_sin"] = np.sin(2 * np.pi * data["week_of_year"] / 52)
    data["week_cos"] = np.cos(2 * np.pi * data["week_of_year"] / 52)
    holiday_dates = dates.dt.date
    data["is_holiday"] = holiday_dates.map(lambda value: int(value in CT_HOLIDAYS)).astype("int8")
    data["is_day_before_holiday"] = holiday_dates.map(
        lambda value: int((value + timedelta(days=1)) in CT_HOLIDAYS)
    ).astype("int8")
    data["is_day_after_holiday"] = holiday_dates.map(
        lambda value: int((value - timedelta(days=1)) in CT_HOLIDAYS)
    ).astype("int8")
    if include_hour:
        data["hour"] = pd.to_numeric(data["hour"], errors="coerce").fillna(0).clip(0, 23).astype("int8")
        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
        payday = dates.map(get_payday_features)
        data["is_payday"] = payday.map(lambda pair: pair[0]).astype("int8")
        data["days_from_payday"] = payday.map(lambda pair: pair[1]).astype("int8")
    return data


def build_daily_city_panel(incidents: pd.DataFrame) -> pd.DataFrame:
    """Aggregate incidents and insert explicit zero-count calendar days for every city."""
    required = {"date", "city"}
    missing = required.difference(incidents.columns)
    if missing:
        raise ValueError(f"Daily panel input is missing columns: {sorted(missing)}")
    source = incidents.copy()
    source["date"] = pd.to_datetime(source["date"]).dt.normalize()
    daily = source.groupby(["date", "city"], observed=True).size().rename("crime_count").reset_index()
    dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    cities = fit_category_vocabulary(daily["city"])
    complete_index = pd.MultiIndex.from_product([dates, cities], names=["date", "city"])
    daily = daily.set_index(["date", "city"]).reindex(complete_index, fill_value=0).reset_index()

    available_resources = [column for column in RESOURCE_FEATURES if column in source.columns]
    if available_resources:
        # Annual staffing, population, and crime-rate fields may summarize a
        # complete calendar year. Expose year Y values only from year Y + 1 so
        # a forecast row cannot see an unfinished target-year aggregate.
        source["resource_year"] = source["date"].dt.year
        resources = (
            source.sort_values("date")
            .groupby(["city", "resource_year"], observed=True)[available_resources]
            .last()
            .reset_index()
        )
        resources["resource_year"] += 1
        daily["resource_year"] = daily["date"].dt.year
        daily = daily.merge(resources, on=["city", "resource_year"], how="left")
        daily[available_resources] = (
            daily.groupby("city", observed=True)[available_resources].ffill().fillna(0)
        )
        daily = daily.drop(columns="resource_year")
    return daily.sort_values(["city", "date"]).reset_index(drop=True)


def build_forecast_features(
    daily: pd.DataFrame,
    city_vocabulary: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build calendar-day lags and rolling statistics that exclude the target row."""
    data = daily.sort_values(["city", "date"]).copy()
    city_vocabulary = city_vocabulary or fit_category_vocabulary(data["city"])
    data["city_code"] = encode_with_vocabulary(data["city"], city_vocabulary).to_numpy()
    group = data.groupby("city", observed=True)["crime_count"]
    for lag in (1, 7, 14, 28, 365):
        data[f"lag_{lag}"] = group.shift(lag)
    for window in (7, 14, 28, 90):
        data[f"roll_mean_{window}"] = group.transform(
            lambda values, size=window: values.shift(1).rolling(size, min_periods=size).mean()
        )
    data["roll_std_28"] = group.transform(lambda values: values.shift(1).rolling(28, min_periods=7).std())
    data["ewm_mean_7"] = group.transform(lambda values: values.shift(1).ewm(span=7, adjust=False).mean())
    data["ewm_mean_28"] = group.transform(lambda values: values.shift(1).ewm(span=28, adjust=False).mean())
    data = add_calendar_features(data)
    data["same_weekday_mean"] = data.groupby(["city", "day_of_week"], observed=True)["crime_count"].transform(
        lambda values: values.shift(1).expanding(min_periods=4).mean()
    )
    data["trend_28_90"] = data["roll_mean_28"] - data["roll_mean_90"]
    artifacts = {
        "schema_version": SCHEMA_VERSION,
        "feature_columns": FORECAST_FEATURES.copy(),
        "city_vocabulary": city_vocabulary,
        "categorical_features": ["city_code"],
    }
    assert_safe_features(artifacts["feature_columns"])
    return data, artifacts


def build_forecast_inference_row(
    city: str,
    current_date: object,
    history: pd.DataFrame,
    latest_stats: Mapping[str, float],
    artifacts: Mapping[str, object],
) -> pd.DataFrame:
    """Build one recursive forecast row from observations strictly before current_date."""
    current_date = pd.Timestamp(current_date).normalize()
    prior = history.copy()
    prior["date"] = pd.to_datetime(prior["date"]).dt.normalize()
    prior = prior[prior["date"] < current_date].sort_values("date")
    if prior.empty:
        raise ValueError("Forecast history must contain a row before the target date.")
    series = prior.drop_duplicates("date", keep="last").set_index("date")["crime_count"].astype(float)
    fallback = float(series.tail(28).mean())

    row = {"date": current_date, "city": city}
    for lag in (1, 7, 14, 28, 365):
        row[f"lag_{lag}"] = float(series.get(current_date - pd.Timedelta(days=lag), fallback))
    for window in (7, 14, 28, 90):
        values = series.tail(window)
        row[f"roll_mean_{window}"] = float(values.mean()) if not values.empty else fallback
    row["roll_std_28"] = float(series.tail(28).std(ddof=1)) if len(series.tail(28)) > 1 else 0.0
    row["ewm_mean_7"] = float(series.ewm(span=7, adjust=False).mean().iloc[-1])
    row["ewm_mean_28"] = float(series.ewm(span=28, adjust=False).mean().iloc[-1])
    same_weekday = series[series.index.dayofweek == current_date.dayofweek]
    row["same_weekday_mean"] = float(same_weekday.mean()) if not same_weekday.empty else fallback
    row["trend_28_90"] = row["roll_mean_28"] - row["roll_mean_90"]
    row.update({feature: float(latest_stats.get(feature, 0.0)) for feature in RESOURCE_FEATURES})
    frame = add_calendar_features(pd.DataFrame([row]))
    frame["city_code"] = encode_with_vocabulary(frame["city"], artifacts["city_vocabulary"]).to_numpy()
    feature_columns = list(artifacts["feature_columns"])
    assert_safe_features(feature_columns)
    return frame.reindex(columns=feature_columns, fill_value=0)


def apply_forecast_strategy(
    raw_prediction: Iterable[float],
    feature_frame: pd.DataFrame,
    artifacts: Mapping[str, object] | None = None,
    *,
    prediction_mode: str | None = None,
    model_weight: float | None = None,
) -> np.ndarray:
    """Apply the same residual correction and learned blend at train and inference time."""
    settings = artifacts or {}
    mode = prediction_mode or str(settings.get("prediction_mode", "direct"))
    if mode not in FORECAST_PREDICTION_MODES:
        raise ValueError(f"Unknown forecast prediction mode: {mode}")
    weight = float(settings.get("model_weight", 1.0) if model_weight is None else model_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("Forecast model_weight must be between 0 and 1.")
    if "roll_mean_7" not in feature_frame:
        raise ValueError("Forecast strategy requires the leakage-safe roll_mean_7 baseline feature.")

    raw = np.asarray(raw_prediction, dtype=float).reshape(-1)
    baseline = feature_frame["roll_mean_7"].to_numpy(dtype=float)
    if len(raw) != len(baseline):
        raise ValueError("Prediction and feature row counts do not match.")
    model_prediction = baseline + raw if mode == "residual_to_roll_mean_7" else raw
    combined = weight * model_prediction + (1.0 - weight) * baseline
    return np.clip(combined, 0, None)


def fit_classification_artifacts(train: pd.DataFrame, include_resources: bool = True) -> dict[str, object]:
    """Fit target-independent lookup tables using only the chronological training period."""
    required = {"date", "city", "location_area", "hour"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Classification training data is missing columns: {sorted(missing)}")
    data = train.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["location_type"] = data["location_area"].map(infer_location_type)
    vocabularies = {
        "city": fit_category_vocabulary(data["city"]),
        "location_area": fit_category_vocabulary(data["location_area"]),
        "location_type": fit_category_vocabulary(data["location_type"]),
    }
    location_daily = data.groupby(["date", "location_area"], observed=True).size()
    location_daily_mean = location_daily.groupby("location_area", observed=True).mean().to_dict()
    city_hour_counts = data.groupby(["city", "hour"], observed=True).size().to_dict()
    city_resources = {}
    city_year_resources = {feature: {} for feature in RESOURCE_FEATURES}
    resource_defaults = {feature: 0.0 for feature in RESOURCE_FEATURES}
    available_resources = [feature for feature in RESOURCE_FEATURES if feature in data.columns]
    if available_resources:
        data["resource_year"] = data["date"].dt.year
        cutoff_year = int(data["resource_year"].max())
        # A source-year summary is considered available in the following year.
        # Excluding the cutoff year prevents a partial/current-year aggregate
        # from entering training or future inference artifacts.
        completed = data[data["resource_year"] < cutoff_year]
        annual = (
            completed.sort_values("date")
            .groupby(["city", "resource_year"], observed=True)[available_resources]
            .last()
            .reset_index()
        )
        annual["available_year"] = annual["resource_year"] + 1
        for feature in available_resources:
            numeric = pd.to_numeric(annual[feature], errors="coerce")
            resource_defaults[feature] = float(numeric.median()) if numeric.notna().any() else 0.0
            city_year_resources[feature] = {
                f"{city}\u241f{int(year)}": float(value)
                for city, year, value in zip(annual["city"], annual["available_year"], numeric)
                if pd.notna(value)
            }
        if not annual.empty:
            latest = annual.sort_values("available_year").groupby("city", observed=True).last()
            for city, row in latest.iterrows():
                city_resources[str(city)] = {
                    feature: float(row.get(feature, resource_defaults[feature]))
                    if pd.notna(row.get(feature)) else resource_defaults[feature]
                    for feature in RESOURCE_FEATURES
                }
    feature_columns = CLASSIFICATION_FEATURES.copy()
    if not include_resources:
        feature_columns = [feature for feature in feature_columns if feature not in RESOURCE_FEATURES]
    artifacts = {
        "schema_version": SCHEMA_VERSION,
        "fitted_through": str(data["date"].max().date()),
        "feature_columns": feature_columns,
        "categorical_features": CATEGORICAL_MODEL_FEATURES.copy(),
        "vocabularies": vocabularies,
        "city_counts": data["city"].astype(str).value_counts().to_dict(),
        "location_counts": data["location_area"].astype(str).value_counts().to_dict(),
        "location_daily_mean": {str(key): float(value) for key, value in location_daily_mean.items()},
        "city_hour_counts": {f"{city}\u241f{int(hour)}": int(value) for (city, hour), value in city_hour_counts.items()},
        "city_resources": city_resources,
        "city_year_resources": city_year_resources,
        "resource_defaults": resource_defaults,
        "resource_policy": "previous_complete_calendar_year_only",
        "unknown_policy": "category_code_minus_one_and_zero_historical_counts",
    }
    assert_safe_features(feature_columns)
    return artifacts


def build_classification_features(frame: pd.DataFrame, artifacts: Mapping[str, object]) -> pd.DataFrame:
    """Transform incident contexts using frozen, training-only lookup artifacts."""
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["city"] = data["city"].astype("string")
    data["location_area"] = data["location_area"].astype("string")
    data["location_type"] = data.get("location_type", data["location_area"].map(infer_location_type))
    data["location_type"] = data["location_type"].fillna(data["location_area"].map(infer_location_type)).astype("string")
    data = add_calendar_features(data, include_hour=True)
    data["is_university"] = data["city"].isin(UNIVERSITY_CITIES).astype("int8")

    vocabularies = artifacts["vocabularies"]
    data["city_code"] = encode_with_vocabulary(data["city"], vocabularies["city"]).to_numpy()
    data["location_area_code"] = encode_with_vocabulary(
        data["location_area"], vocabularies["location_area"]
    ).to_numpy()
    data["location_type_code"] = encode_with_vocabulary(
        data["location_type"], vocabularies["location_type"]
    ).to_numpy()
    data["city_train_count"] = data["city"].map(artifacts["city_counts"]).fillna(0).astype("float32")
    data["location_train_count"] = data["location_area"].map(artifacts["location_counts"]).fillna(0).astype("float32")
    data["location_train_daily_mean"] = data["location_area"].map(
        artifacts["location_daily_mean"]
    ).fillna(0).astype("float32")
    city_hour_keys = data["city"].astype(str) + "\u241f" + data["hour"].astype(str)
    data["city_hour_train_count"] = city_hour_keys.map(artifacts["city_hour_counts"]).fillna(0).astype("float32")

    defaults = artifacts.get("resource_defaults", {})
    city_resources = artifacts.get("city_resources", {})
    city_year_resources = artifacts.get("city_year_resources", {})
    fitted_year = pd.Timestamp(artifacts.get("fitted_through", data["date"].max())).year
    resource_keys = data["city"].astype(str) + "\u241f" + data["date"].dt.year.astype(str)
    future_rows = data["date"].dt.year > fitted_year
    for feature in RESOURCE_FEATURES:
        if feature not in artifacts["feature_columns"]:
            continue
        historical = resource_keys.map(city_year_resources.get(feature, {}))
        latest_frozen = data["city"].map(
            lambda city: city_resources.get(str(city), {}).get(feature, defaults.get(feature, 0.0))
        )
        # Training/evaluation rows never borrow a resource value from a later
        # year. Truly future inference rows use the last completed training-year
        # value; unknown cities use the frozen training median.
        historical = historical.where(historical.notna() | ~future_rows, latest_frozen)
        data[feature] = historical.fillna(0.0).astype("float32")

    feature_columns = list(artifacts["feature_columns"])
    assert_safe_features(feature_columns)
    transformed = data.reindex(columns=feature_columns, fill_value=0)
    if transformed.isna().any().any():
        missing = transformed.columns[transformed.isna().any()].tolist()
        raise ValueError(f"Classification feature transform produced NaNs in: {missing}")
    return transformed


def fit_rare_class_mapping(
    train_labels: Iterable[object],
    minimum_count: int,
    other_label: str = "Other",
) -> dict[str, object]:
    """Determine rare labels from training labels only and persist the decision."""
    counts = pd.Series(train_labels, dtype="string").value_counts()
    if counts.empty:
        raise ValueError("Rare-class mapping requires at least one training label.")
    rare = sorted(counts[counts < minimum_count].index.tolist())
    fallback_label = other_label if other_label in counts.index or rare else str(counts.index[0])
    return {
        "minimum_count": int(minimum_count),
        "other_label": other_label,
        "fallback_label": fallback_label,
        "rare_labels": rare,
        "known_labels": sorted(counts.index.astype(str).tolist()),
        "training_counts": {str(key): int(value) for key, value in counts.items()},
    }


def apply_rare_class_mapping(labels: Iterable[object], mapping: Mapping[str, object]) -> pd.Series:
    values = pd.Series(labels, dtype="string")
    known_labels = set(mapping.get("known_labels", mapping.get("training_counts", {})))
    replace = values.isin(mapping["rare_labels"]) | ~values.isin(known_labels)
    return values.where(~replace, mapping.get("fallback_label", mapping["other_label"]))
