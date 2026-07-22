"""Evaluation helpers for the Colab training notebook.

This module contains no Streamlit code and never chooses a model from final-test
metrics. The notebook owns experiment orchestration and artifact export.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import (
    RESOURCE_FEATURES,
    apply_forecast_strategy,
    assert_safe_features,
    build_forecast_inference_row,
)


class RoutedClassifier(ClassifierMixin, BaseEstimator):
    """Route each row through a broad prediction to a broad-specific classifier."""

    def __init__(self, broad_model, route_models: Mapping[int, object], class_count: int):
        self.broad_model = broad_model
        self.route_models = dict(route_models)
        self.class_count = int(class_count)
        self.classes_ = np.arange(self.class_count)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        broad_routes = np.asarray(self.broad_model.predict(X), dtype=int)
        output = np.zeros((len(X), self.class_count), dtype=float)
        for route, route_model in self.route_models.items():
            selected = np.flatnonzero(broad_routes == int(route))
            if not len(selected):
                continue
            if isinstance(route_model, (int, np.integer)):
                output[selected, int(route_model)] = 1.0
                continue
            probabilities = route_model.predict_proba(X.iloc[selected])
            output[np.ix_(selected, np.asarray(route_model.classes_, dtype=int))] = probabilities
        empty = output.sum(axis=1) == 0
        if empty.any():
            output[empty] = 1.0 / self.class_count
        return output / output.sum(axis=1, keepdims=True)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def regression_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), 0, None)
    error = predicted - actual
    denominator = np.abs(actual).sum()
    smape_denominator = np.abs(actual) + np.abs(predicted)
    smape_terms = np.zeros_like(smape_denominator, dtype=float)
    np.divide(
        2 * np.abs(error),
        smape_denominator,
        out=smape_terms,
        where=smape_denominator > 0,
    )
    positive_predicted = np.clip(predicted, 1e-9, None)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "wape": float(np.abs(error).sum() / denominator) if denominator else math.nan,
        "smape": float(np.mean(smape_terms)),
        "poisson_deviance": float(mean_poisson_deviance(actual, positive_predicted)),
        "mean_bias": float(error.mean()),
    }


def classification_metrics(
    actual: Iterable[int],
    probabilities: np.ndarray,
    labels: Iterable[int] | None = None,
) -> dict[str, float]:
    actual = np.asarray(actual)
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(list(labels) if labels is not None else np.arange(probabilities.shape[1]))
    predicted = labels[np.argmax(probabilities, axis=1)]
    top_k = min(3, probabilities.shape[1])
    one_hot = np.eye(len(labels))[np.searchsorted(labels, actual)]
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "top_3_accuracy": float(top_k_accuracy_score(actual, probabilities, k=top_k, labels=labels)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "log_loss": float(log_loss(actual, probabilities, labels=labels)),
        "brier_multiclass": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": float(expected_calibration_error(actual, probabilities, labels)),
    }


def expected_calibration_error(
    actual: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
) -> float:
    predicted_positions = np.argmax(probabilities, axis=1)
    predicted = labels[predicted_positions]
    confidence = probabilities[np.arange(len(probabilities)), predicted_positions]
    correct = predicted == actual
    edges = np.linspace(0, 1, bins + 1)
    score = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            score += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(score)


def per_class_metrics(actual: Iterable[int], predicted: Iterable[int], class_names: list[str]) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )


def forecast_baseline_predictions(frame: pd.DataFrame, train: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return meaningful baselines using features already shifted before each target date."""
    city_mean = train.groupby("city", observed=True)["crime_count"].mean().to_dict()
    city_weekday = train.groupby(["city", "day_of_week"], observed=True)["crime_count"].mean().to_dict()
    seasonal = frame["lag_365"].where(frame["lag_365"].notna(), frame["lag_7"])
    return {
        "Previous day": frame["lag_1"].to_numpy(float),
        "Previous week": frame["lag_7"].to_numpy(float),
        "Seven-day rolling average": frame["roll_mean_7"].to_numpy(float),
        "Same weekday historical average": np.array(
            [city_weekday.get((city, weekday), city_mean.get(city, 0.0)) for city, weekday in zip(frame["city"], frame["day_of_week"])],
            dtype=float,
        ),
        "City historical mean": frame["city"].map(city_mean).fillna(train["crime_count"].mean()).to_numpy(float),
        "Seasonal naive": seasonal.to_numpy(float),
    }


def forecast_origins(
    evaluation_start: object,
    evaluation_end: object,
    horizon: int = 30,
    stride: int = 30,
) -> list[pd.Timestamp]:
    """Return full-horizon forecast origins, including an end-aligned origin."""
    if horizon < 1 or stride < 1:
        raise ValueError("Forecast horizon and stride must be positive.")
    start = pd.Timestamp(evaluation_start).normalize()
    end = pd.Timestamp(evaluation_end).normalize()
    latest_origin = end - pd.Timedelta(days=horizon - 1)
    if latest_origin < start:
        return []
    origins = list(pd.date_range(start, latest_origin, freq=f"{stride}D"))
    if not origins or origins[-1] != latest_origin:
        origins.append(latest_origin)
    return origins


def recursive_forecast_backtest(
    model,
    daily: pd.DataFrame,
    artifacts: Mapping[str, object],
    evaluation_start: object,
    evaluation_end: object,
    *,
    horizon: int = 30,
    stride: int = 30,
    prediction_mode: str = "direct",
    model_weight: float = 1.0,
    cities: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Backtest app-like recursive forecasts without exposing holdout outcomes.

    Each origin starts with actual rows strictly before that origin. Predictions
    are appended to the recursive history, so horizon 2-30 cannot consume actual
    outcomes from the same holdout trajectory. Later origins may use observations
    that would have become available by their real forecast date.
    """
    required = {"date", "city", "crime_count"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Recursive backtest data is missing columns: {sorted(missing)}")
    feature_columns = list(artifacts.get("feature_columns", []))
    if not feature_columns:
        raise ValueError("Forecast artifacts do not contain feature_columns.")

    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["date", "city"]).sort_values(["city", "date"])
    data["_city_key"] = data["city"].astype(str)
    available_cities = set(data["_city_key"].unique())
    requested_cities = list(cities) if cities is not None else list(artifacts.get("city_vocabulary", []))
    selected_cities = [str(city) for city in requested_cities if str(city) in available_cities]
    if not selected_cities:
        selected_cities = sorted(available_cities)

    actual_lookup = data.set_index(["date", "_city_key"])["crime_count"]
    origins = forecast_origins(evaluation_start, evaluation_end, horizon=horizon, stride=stride)
    rows: list[dict[str, object]] = []
    for origin in origins:
        histories: dict[str, pd.DataFrame] = {}
        latest_stats: dict[str, dict[str, float]] = {}
        for city in selected_cities:
            prior = data[(data["_city_key"] == city) & (data["date"] < origin)]
            if prior.empty:
                continue
            histories[city] = prior[["date", "crime_count"]].copy()
            latest = prior.iloc[-1]
            latest_stats[city] = {
                feature: float(latest.get(feature, 0.0)) if pd.notna(latest.get(feature, 0.0)) else 0.0
                for feature in RESOURCE_FEATURES
            }

        for horizon_step in range(1, horizon + 1):
            target_date = origin + pd.Timedelta(days=horizon_step - 1)
            feature_rows = []
            active_cities = []
            actual_values = []
            for city, history in histories.items():
                key = (target_date, city)
                if key not in actual_lookup.index:
                    continue
                feature_rows.append(
                    build_forecast_inference_row(
                        city, target_date, history, latest_stats[city], artifacts
                    )
                )
                active_cities.append(city)
                actual_values.append(float(actual_lookup.loc[key]))
            if not feature_rows:
                continue

            features = pd.concat(feature_rows, ignore_index=True).reindex(columns=feature_columns)
            raw_prediction = np.zeros(len(features), dtype=float) if model is None else model.predict(features)
            prediction = apply_forecast_strategy(
                raw_prediction,
                features,
                prediction_mode=prediction_mode,
                model_weight=model_weight,
            )
            model_prediction = (
                np.full(len(features), np.nan)
                if model is None
                else apply_forecast_strategy(
                    raw_prediction,
                    features,
                    prediction_mode=prediction_mode,
                    model_weight=1.0,
                )
            )
            baseline_prediction = features["roll_mean_7"].to_numpy(dtype=float)
            for position, city in enumerate(active_cities):
                value = float(prediction[position])
                rows.append(
                    {
                        "origin": origin,
                        "date": target_date,
                        "city": city,
                        "horizon": horizon_step,
                        "actual": actual_values[position],
                        "predicted": value,
                        "model_prediction": float(model_prediction[position]),
                        "baseline_prediction": float(baseline_prediction[position]),
                    }
                )
                histories[city] = pd.concat(
                    [
                        histories[city],
                        pd.DataFrame({"date": [target_date], "crime_count": [value]}),
                    ],
                    ignore_index=True,
                )
    return pd.DataFrame(rows)


def select_forecast_blend_weight(
    model,
    daily: pd.DataFrame,
    artifacts: Mapping[str, object],
    evaluation_start: object,
    evaluation_end: object,
    *,
    prediction_mode: str = "direct",
    horizon: int = 30,
    stride: int = 30,
    weights: Iterable[float] | None = None,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Learn a constrained model/baseline weight using validation origins only."""
    candidate_weights = np.asarray(
        list(weights) if weights is not None else np.linspace(0.0, 1.0, 21), dtype=float
    )
    if not len(candidate_weights) or ((candidate_weights < 0) | (candidate_weights > 1)).any():
        raise ValueError("Blend weights must contain values between 0 and 1.")
    comparisons = []
    backtests: dict[float, pd.DataFrame] = {}
    for weight in np.unique(candidate_weights):
        result = recursive_forecast_backtest(
            model,
            daily,
            artifacts,
            evaluation_start,
            evaluation_end,
            horizon=horizon,
            stride=stride,
            prediction_mode=prediction_mode,
            model_weight=float(weight),
        )
        if result.empty:
            raise ValueError("The requested period does not contain a complete forecast horizon.")
        metrics = regression_metrics(result["actual"], result["predicted"])
        comparisons.append({"model_weight": float(weight), **metrics})
        backtests[float(weight)] = result
    table = pd.DataFrame(comparisons)
    ranked = table.assign(abs_mean_bias=table["mean_bias"].abs()).sort_values(
        ["mae", "rmse", "wape", "abs_mean_bias", "model_weight"], kind="stable"
    )
    selected_weight = float(ranked.iloc[0]["model_weight"])
    return selected_weight, table.sort_values("model_weight").reset_index(drop=True), backtests[selected_weight]


def frequency_probability_baseline(
    train_keys: pd.Series,
    train_labels: np.ndarray,
    test_keys: pd.Series,
    class_count: int,
    smoothing: float = 1.0,
) -> np.ndarray:
    table = pd.crosstab(train_keys.astype(str), train_labels)
    table = table.reindex(columns=np.arange(class_count), fill_value=0).astype(float) + smoothing
    table = table.div(table.sum(axis=1), axis=0)
    overall = np.bincount(train_labels, minlength=class_count).astype(float) + smoothing
    overall /= overall.sum()
    return np.vstack([table.loc[str(key)].to_numpy() if str(key) in table.index else overall for key in test_keys])


def classification_baselines(
    train_context: pd.DataFrame,
    train_labels: np.ndarray,
    eval_context: pd.DataFrame,
    class_count: int,
) -> dict[str, np.ndarray]:
    overall_key_train = pd.Series("all", index=train_context.index)
    overall_key_eval = pd.Series("all", index=eval_context.index)
    return {
        "Overall majority/frequency": frequency_probability_baseline(
            overall_key_train, train_labels, overall_key_eval, class_count
        ),
        "Per-city frequency": frequency_probability_baseline(
            train_context["city"], train_labels, eval_context["city"], class_count
        ),
        "Per-location-type frequency": frequency_probability_baseline(
            train_context["location_type"], train_labels, eval_context["location_type"], class_count
        ),
        "Per-city-hour frequency": frequency_probability_baseline(
            train_context["city"].astype(str) + "|" + train_context["hour"].astype(str),
            train_labels,
            eval_context["city"].astype(str) + "|" + eval_context["hour"].astype(str),
            class_count,
        ),
    }


def make_poisson_regression(feature_columns: list[str]) -> Pipeline:
    assert_safe_features(feature_columns)
    categorical = [column for column in ["city_code"] if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    preprocessing = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ],
        sparse_threshold=0.3,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", PoissonRegressor(alpha=1.0, max_iter=500))])


def make_logistic_classifier(feature_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    assert_safe_features(feature_columns)
    numeric = [column for column in feature_columns if column not in categorical_columns]
    preprocessing = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=10), categorical_columns),
            ("numeric", StandardScaler(), numeric),
        ],
        sparse_threshold=0.3,
    )
    model = LogisticRegression(max_iter=400, class_weight="balanced", solver="saga", n_jobs=1)
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def make_xgboost_count_regressor(feature_columns: list[str], random_state: int, n_jobs: int) -> Pipeline:
    """Use one-hot nominal city values before XGBoost count regression."""
    from xgboost import XGBRegressor

    categorical = [column for column in ["city_code"] if column in feature_columns]
    preprocessing = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)], remainder="passthrough"
    )
    model = XGBRegressor(
        objective="count:poisson",
        n_estimators=900,
        max_depth=8,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def make_xgboost_classifier(
    feature_columns: list[str],
    categorical_columns: list[str],
    class_count: int,
    random_state: int,
    n_jobs: int,
) -> Pipeline:
    """Use one-hot nominal values so XGBoost never treats category IDs as ordered numbers."""
    from xgboost import XGBClassifier

    preprocessing = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns)],
        remainder="passthrough",
    )
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=class_count,
        n_estimators=800,
        max_depth=8,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def predict_timed(model, X: pd.DataFrame, probabilities: bool = False) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    prediction = model.predict_proba(X) if probabilities else model.predict(X)
    elapsed = time.perf_counter() - started
    return np.asarray(prediction), float(1000 * elapsed / max(len(X), 1))


def artifact_size_mb(model: object, path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return float(path.stat().st_size / (1024 ** 2))


def select_classification_candidate(
    comparison: pd.DataFrame,
    max_artifact_mb: float | None = None,
) -> str:
    """Select a validation winner that fits the configured deployment budget."""
    required = {"model_name", "macro_f1", "log_loss", "balanced_accuracy", "top_3_accuracy", "artifact_size_mb"}
    missing = required.difference(comparison.columns)
    if missing:
        raise ValueError(f"Classification comparison is missing columns: {sorted(missing)}")
    eligible = comparison
    if max_artifact_mb is not None:
        within_budget = comparison[comparison["artifact_size_mb"] <= max_artifact_mb]
        if not within_budget.empty:
            eligible = within_budget
    ranked = eligible.sort_values(
        ["macro_f1", "log_loss", "balanced_accuracy", "top_3_accuracy", "artifact_size_mb"],
        ascending=[False, True, False, False, True],
        kind="stable",
    )
    return str(ranked.iloc[0]["model_name"])


def select_forecast_candidate(comparison: pd.DataFrame) -> str:
    required = {"model_name", "mae", "rmse", "wape", "mean_bias", "artifact_size_mb"}
    missing = required.difference(comparison.columns)
    if missing:
        raise ValueError(f"Forecast comparison is missing columns: {sorted(missing)}")
    ranked = comparison.sort_values(
        ["mae", "rmse", "wape", "mean_bias", "artifact_size_mb"],
        key=lambda column: column.abs() if column.name == "mean_bias" else column,
        kind="stable",
    )
    return str(ranked.iloc[0]["model_name"])


def paired_absolute_error_bootstrap(
    actual: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    random_state: int = 42,
    samples: int = 2000,
    groups: Iterable[object] | None = None,
) -> dict[str, float]:
    """Estimate a paired MAE interval, optionally resampling forecast trajectories."""
    rng = np.random.default_rng(random_state)
    differences = np.abs(candidate - actual) - np.abs(baseline - actual)
    if groups is not None:
        group_values = np.asarray(list(groups))
        if len(group_values) != len(differences):
            raise ValueError("Bootstrap groups must align with forecast rows.")
        unique_groups = np.unique(group_values)
        differences = np.asarray(
            [differences[group_values == group].mean() for group in unique_groups], dtype=float
        )
    means = np.empty(samples)
    for index in range(samples):
        means[index] = rng.choice(differences, size=len(differences), replace=True).mean()
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {"mean_mae_difference": float(differences.mean()), "ci_95_lower": float(lower), "ci_95_upper": float(upper)}


def paired_accuracy_bootstrap(
    actual: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    random_state: int = 42,
    samples: int = 2000,
) -> dict[str, float]:
    """Estimate a paired 95% interval for candidate-minus-baseline accuracy."""
    rng = np.random.default_rng(random_state)
    differences = (candidate == actual).astype(float) - (baseline == actual).astype(float)
    means = np.empty(samples)
    for index in range(samples):
        means[index] = rng.choice(differences, size=len(differences), replace=True).mean()
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "mean_accuracy_difference": float(differences.mean()),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
    }


def comparison_row(
    task: str,
    model_name: str,
    feature_set: str,
    periods: Mapping[str, str],
    validation_metrics: Mapping[str, float],
    test_metrics: Mapping[str, float] | None,
    **details,
) -> dict[str, object]:
    return {
        "task": task,
        "model_name": model_name,
        "feature_set": feature_set,
        "training_period": periods["training_period"],
        "validation_period": periods["validation_period"],
        "test_period": periods["test_period"],
        "validation_metrics": json.dumps(dict(validation_metrics), sort_keys=True),
        "final_test_metrics": json.dumps(dict(test_metrics), sort_keys=True) if test_metrics else "PENDING_COLAB_FINAL_TEST",
        "imbalance_method": details.get("imbalance_method", "None"),
        "calibration_method": details.get("calibration_method", "None"),
        "important_hyperparameters": json.dumps(details.get("hyperparameters", {}), sort_keys=True, default=str),
        "training_time_seconds": details.get("training_time_seconds"),
        "inference_ms_per_row": details.get("inference_ms_per_row"),
        "artifact_size_mb": details.get("artifact_size_mb"),
        "selected_or_rejected": details.get("decision", "Pending"),
        "selection_reason": details.get("reason", "Pending validation comparison"),
    }


def package_versions(packages: Iterable[str]) -> dict[str, str]:
    import importlib.metadata

    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(
    path: Path,
    model_version: str,
    dataset_range: Mapping[str, str],
    selected_models: Mapping[str, str],
    feature_schemas: Mapping[str, list[str]],
    evaluation_metrics: Mapping[str, object],
    artifact_paths: Iterable[Path],
    project_root: Path,
) -> dict[str, object]:
    manifest = {
        "model_version": model_version,
        "training_date_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset_date_range": dict(dataset_range),
        "git_commit": git_commit(project_root),
        "selected_models": dict(selected_models),
        "feature_schemas": {key: list(value) for key, value in feature_schemas.items()},
        "evaluation_metrics": dict(evaluation_metrics),
        "package_versions": package_versions(
            ["pandas", "numpy", "scikit-learn", "lightgbm", "catboost", "xgboost", "joblib"]
        ),
        "python_version": platform.python_version(),
        "artifacts": [
            {"filename": item.name, "size_bytes": item.stat().st_size}
            for item in artifact_paths
            if item.exists()
        ],
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def checkpoint(path: Path, factory: Callable[[], object], resume: bool = True) -> object:
    """Load a completed expensive experiment or run and save it once."""
    if resume and path.exists():
        return joblib.load(path)
    result = factory()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, path)
    return result
