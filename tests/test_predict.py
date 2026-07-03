import unittest

import pandas as pd

from predict import compute_blended_prediction, get_location_type, get_payday_features, prepare_forecast_history


class ForecastBlendTests(unittest.TestCase):
    def test_blend_uses_documented_weights_without_trend(self):
        result = compute_blended_prediction(
            model_pred=100,
            analog_pred=40,
            recent_pred=20,
            seasonal_pred=10,
            buffer=[1, 2, 3],
        )
        self.assertEqual(result, 72.5)

    def test_blend_adds_small_recent_trend_adjustment(self):
        buffer = [10] * 7 + [20] * 7
        result = compute_blended_prediction(
            model_pred=100,
            analog_pred=40,
            recent_pred=20,
            seasonal_pred=10,
            buffer=buffer,
        )
        self.assertEqual(result, 73.3)

    def test_blend_is_never_negative(self):
        result = compute_blended_prediction(
            model_pred=-100,
            analog_pred=-100,
            recent_pred=-100,
            seasonal_pred=-100,
            buffer=[20] * 7 + [1] * 7,
        )
        self.assertEqual(result, 0)

    def test_forecast_history_excludes_target_date_and_future_rows(self):
        history = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'crime_count': [10, 99, 999],
        })
        filtered = prepare_forecast_history(history, pd.Timestamp('2024-01-02'))
        self.assertEqual(filtered['date'].max(), pd.Timestamp('2024-01-01'))
        self.assertEqual(filtered['crime_count'].tolist(), [10])


class FeatureEngineeringTests(unittest.TestCase):
    def test_location_type_prioritizes_commercial_office_over_highway(self):
        self.assertEqual(get_location_type('Highway Patrol Office'), 'commercial')

    def test_payday_uses_last_day_of_month(self):
        is_payday, days_from_payday = get_payday_features(pd.Timestamp('2024-02-29'))
        self.assertEqual(is_payday, 1)
        self.assertEqual(days_from_payday, 0)


if __name__ == '__main__':
    unittest.main()
