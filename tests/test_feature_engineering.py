import unittest

import numpy as np
import pandas as pd

from feature_engineering import (
    CLASSIFICATION_FEATURES,
    FORECAST_FEATURES,
    TARGET_COLUMNS,
    apply_rare_class_mapping,
    assert_safe_features,
    build_classification_features,
    build_daily_city_panel,
    build_forecast_features,
    build_forecast_inference_row,
    fit_classification_artifacts,
    fit_rare_class_mapping,
    make_temporal_boundaries,
    map_broad_category,
    rolling_origin_boundaries,
    temporal_masks,
)


def daily_frame(periods=420):
    dates = pd.date_range("2020-01-01", periods=periods, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "city": ["Alpha"] * periods,
            "crime_count": np.arange(periods, dtype=float),
            "population": [10_000] * periods,
            "total_officers": [25] * periods,
            "officers_per_1000_people": [2.5] * periods,
            "crime_rate_per_1000_people": [20.0] * periods,
        }
    )


class ForecastFeatureTests(unittest.TestCase):
    def test_prediction_schemas_exclude_every_target_column(self):
        assert_safe_features(FORECAST_FEATURES)
        assert_safe_features(CLASSIFICATION_FEATURES)
        self.assertFalse(TARGET_COLUMNS.intersection(FORECAST_FEATURES))
        self.assertFalse(TARGET_COLUMNS.intersection(CLASSIFICATION_FEATURES))

    def test_daily_panel_inserts_missing_calendar_day(self):
        incidents = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
                "city": ["Alpha", "Alpha"],
            }
        )
        panel = build_daily_city_panel(incidents)
        jan_2 = panel.loc[panel["date"] == pd.Timestamp("2024-01-02"), "crime_count"]
        self.assertEqual(jan_2.tolist(), [0])

    def test_annual_resource_values_are_available_only_the_following_year(self):
        incidents = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2024-01-01"]),
                "city": ["Alpha", "Alpha"],
                "population": [10_000, 11_000],
            }
        )
        panel = build_daily_city_panel(incidents)
        self.assertEqual(panel.loc[panel["date"] == "2023-01-01", "population"].iloc[0], 0)
        self.assertEqual(panel.loc[panel["date"] == "2024-01-01", "population"].iloc[0], 10_000)

    def test_lags_and_rolling_features_exclude_target_row(self):
        featured, _ = build_forecast_features(daily_frame())
        row = featured.iloc[-1]
        self.assertEqual(row["lag_1"], 418.0)
        self.assertEqual(row["lag_7"], 412.0)
        self.assertEqual(row["roll_mean_7"], np.mean(np.arange(412, 419)))
        self.assertNotEqual(row["roll_mean_7"], np.mean(np.arange(413, 420)))

    def test_training_and_inference_feature_order_matches(self):
        featured, artifacts = build_forecast_features(daily_frame())
        history = featured[["date", "crime_count"]]
        latest = daily_frame().iloc[-1]
        inference = build_forecast_inference_row(
            "Alpha",
            featured["date"].max() + pd.Timedelta(days=1),
            history,
            latest.to_dict(),
            artifacts,
        )
        self.assertEqual(inference.columns.tolist(), artifacts["feature_columns"])
        self.assertFalse(inference.isna().any().any())


class ClassificationFeatureTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "city": ["Alpha", "Alpha", "Beta"],
                "location_area": ["School", "Bank", "Road"],
                "hour": [8, 12, 22],
                "population": [10_000, 10_000, 8_000],
                "total_officers": [25, 25, 20],
                "officers_per_1000_people": [2.5, 2.5, 2.5],
                "crime_rate_per_1000_people": [20.0, 20.0, 15.0],
            }
        )

    def test_vocabularies_and_lookups_are_fitted_from_training_only(self):
        artifacts = fit_classification_artifacts(self.train)
        future = pd.DataFrame(
            {
                "date": [pd.Timestamp("2025-01-01")],
                "city": ["Future City"],
                "location_area": ["Future Location"],
                "hour": [5],
            }
        )
        transformed = build_classification_features(future, artifacts)
        self.assertEqual(transformed.loc[0, "city_code"], -1)
        self.assertEqual(transformed.loc[0, "location_area_code"], -1)
        self.assertEqual(transformed.loc[0, "city_train_count"], 0)
        self.assertEqual(transformed.loc[0, "location_train_count"], 0)
        self.assertEqual(transformed.columns.tolist(), artifacts["feature_columns"])

    def test_broad_category_definition_covers_person_and_property_offenses(self):
        self.assertEqual(map_broad_category("Homicide Offenses"), "Violent")
        self.assertEqual(map_broad_category("Kidnapping/Abduction"), "Violent")
        self.assertEqual(map_broad_category("Fraud Offenses"), "Property")
        self.assertEqual(map_broad_category("Drug/Narcotic Offenses"), "Other")

    def test_rare_class_mapping_uses_only_given_training_labels(self):
        mapping = fit_rare_class_mapping(["Common"] * 5 + ["Rare"], minimum_count=2)
        mapped = apply_rare_class_mapping(["Common", "Rare", "Future"], mapping)
        self.assertEqual(mapping["rare_labels"], ["Rare"])
        self.assertEqual(mapped.tolist(), ["Common", "Other", "Other"])

    def test_unseen_label_falls_back_to_a_trained_class_when_other_has_no_samples(self):
        mapping = fit_rare_class_mapping(["Common"] * 3, minimum_count=2)
        mapped = apply_rare_class_mapping(["Future"], mapping)
        self.assertEqual(mapped.tolist(), ["Common"])

    def test_resource_lookups_use_previous_complete_year(self):
        training = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2023-12-01", "2024-01-01"]),
                "city": ["Alpha", "Alpha", "Alpha"],
                "location_area": ["School", "School", "School"],
                "hour": [8, 8, 8],
                "population": [9_000, 10_000, 99_000],
                "total_officers": [18, 20, 999],
                "officers_per_1000_people": [2.0, 2.0, 10.0],
                "crime_rate_per_1000_people": [12.0, 13.0, 99.0],
            }
        )
        artifacts = fit_classification_artifacts(training)
        transformed = build_classification_features(training, artifacts)
        self.assertEqual(transformed.loc[0, "population"], 0)
        self.assertEqual(transformed.loc[2, "population"], 10_000)


class TemporalSplitTests(unittest.TestCase):
    def test_periods_are_disjoint_and_final_test_is_after_validation(self):
        dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
        boundaries = make_temporal_boundaries(dates, "2024-11-20", validation_days=90)
        masks = temporal_masks(dates, boundaries)
        self.assertFalse((masks["train"] & masks["validation"]).any())
        self.assertFalse((masks["validation"] & masks["test"]).any())
        self.assertLess(boundaries.validation_end, boundaries.test_start)

    def test_walk_forward_folds_never_reach_final_test(self):
        folds = rolling_origin_boundaries("2024-08-21", "2020-01-01", folds=4)
        self.assertTrue(folds)
        for train_end, validation_start, validation_end in folds:
            self.assertLess(train_end, validation_start)
            self.assertLessEqual(validation_end, pd.Timestamp("2024-08-21"))


if __name__ == "__main__":
    unittest.main()
