import unittest

import numpy as np
import pandas as pd

from feature_engineering import apply_forecast_strategy, build_forecast_features
from training_pipeline import (
    forecast_origins,
    recursive_forecast_backtest,
    select_forecast_blend_weight,
)


class LagOneRegressor:
    def predict(self, frame):
        return frame["lag_1"].to_numpy(dtype=float)


class ConstantRegressor:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, frame):
        return np.full(len(frame), self.value)


def daily_counts(values):
    values = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=len(values), freq="D"),
            "city": ["Alpha"] * len(values),
            "crime_count": values,
            "population": [10_000] * len(values),
            "total_officers": [20] * len(values),
            "officers_per_1000_people": [2.0] * len(values),
            "crime_rate_per_1000_people": [15.0] * len(values),
        }
    )


class ForecastStrategyTests(unittest.TestCase):
    def test_residual_and_blend_are_reconstructed_from_rolling_baseline(self):
        features = pd.DataFrame({"roll_mean_7": [10.0, 20.0]})
        prediction = apply_forecast_strategy(
            [2.0, -4.0],
            features,
            prediction_mode="residual_to_roll_mean_7",
            model_weight=0.5,
        )
        self.assertTrue(np.allclose(prediction, [11.0, 18.0]))

    def test_invalid_blend_weight_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            apply_forecast_strategy(
                [1.0], pd.DataFrame({"roll_mean_7": [1.0]}), model_weight=1.1
            )


class RecursiveBacktestTests(unittest.TestCase):
    def test_end_aligned_origin_is_included_for_short_test_window(self):
        origins = forecast_origins("2024-11-20", "2024-12-31", horizon=30, stride=30)
        self.assertEqual(origins, [pd.Timestamp("2024-11-20"), pd.Timestamp("2024-12-02")])

    def test_holdout_actuals_are_never_fed_into_same_trajectory(self):
        daily = daily_counts([1.0] * 370 + [100.0] * 30)
        _, artifacts = build_forecast_features(daily)
        result = recursive_forecast_backtest(
            LagOneRegressor(),
            daily,
            artifacts,
            daily.loc[370, "date"],
            daily.loc[399, "date"],
            horizon=30,
            stride=30,
        )
        self.assertEqual(len(result), 30)
        self.assertTrue(np.allclose(result["predicted"], 1.0))
        self.assertTrue(np.allclose(result["actual"], 100.0))

    def test_validation_selects_model_when_it_beats_recursive_baseline(self):
        daily = daily_counts([1.0] * 370 + [5.0] * 30)
        _, artifacts = build_forecast_features(daily)
        weight, table, result = select_forecast_blend_weight(
            ConstantRegressor(5.0),
            daily,
            artifacts,
            daily.loc[370, "date"],
            daily.loc[399, "date"],
            weights=[0.0, 0.5, 1.0],
        )
        self.assertEqual(weight, 1.0)
        self.assertEqual(float(table.loc[table["model_weight"] == 1.0, "mae"].iloc[0]), 0.0)
        self.assertTrue(np.allclose(result["predicted"], 5.0))


if __name__ == "__main__":
    unittest.main()
