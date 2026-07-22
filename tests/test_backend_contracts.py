import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import holidays
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import data
from feature_engineering import (
    add_calendar_features,
    build_forecast_features,
    fit_classification_artifacts,
)
from predict import predict_crime_risk, run_forecast_loop
from training_pipeline import RoutedClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConstantRegressor:
    def predict(self, frame):
        return np.full(len(frame), 2.0)


class ConstantClassifier:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.classes_ = np.arange(len(self.probabilities))

    def predict_proba(self, frame):
        return np.tile(self.probabilities, (len(frame), 1))

    def predict(self, frame):
        return np.full(len(frame), int(np.argmax(self.probabilities)))


class BackendContractTests(unittest.TestCase):
    def test_true_hierarchy_routes_to_the_selected_broad_model(self):
        broad = ConstantClassifier([0.1, 0.8])
        routed = RoutedClassifier(broad, {0: 0, 1: ConstantClassifier([0.25, 0.75])}, class_count=2)
        probabilities = routed.predict_proba(pd.DataFrame({"x": [1, 2]}))
        self.assertTrue(np.allclose(probabilities, [[0.25, 0.75], [0.25, 0.75]]))

    def test_v2_forecast_output_keeps_app_columns(self):
        periods = 420
        daily = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=periods),
                "city": ["Alpha"] * periods,
                "crime_count": np.ones(periods),
                "population": [10_000] * periods,
                "total_officers": [20] * periods,
                "officers_per_1000_people": [2.0] * periods,
                "crime_rate_per_1000_people": [15.0] * periods,
            }
        )
        profile, forecast_artifacts = build_forecast_features(daily)
        output = run_forecast_loop(
            city="Alpha",
            forecaster=ConstantRegressor(),
            forecast_features=forecast_artifacts["feature_columns"],
            per_city_forecasters={},
            per_city_forecast_features=forecast_artifacts["feature_columns"],
            forecast_profiles={"Alpha": profile},
            ct_holidays=holidays.US(subdiv="CT"),
            steps=3,
            target_start_date=profile["date"].max() + pd.Timedelta(days=1),
            feature_artifacts={"schema_version": "2.0", "forecast": forecast_artifacts},
        )
        self.assertTrue({"Date", "Predicted Count", "is_holiday"}.issubset(output.columns))
        self.assertEqual(len(output), 3)

    def test_v2_forecast_applies_saved_validation_blend(self):
        periods = 420
        daily = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=periods),
                "city": ["Alpha"] * periods,
                "crime_count": np.ones(periods),
                "population": [10_000] * periods,
                "total_officers": [20] * periods,
                "officers_per_1000_people": [2.0] * periods,
                "crime_rate_per_1000_people": [15.0] * periods,
            }
        )
        profile, forecast_artifacts = build_forecast_features(daily)
        forecast_artifacts.update({"prediction_mode": "direct", "model_weight": 0.0})
        output = run_forecast_loop(
            city="Alpha",
            forecaster=ConstantRegressor(),
            forecast_features=forecast_artifacts["feature_columns"],
            per_city_forecasters={},
            per_city_forecast_features=forecast_artifacts["feature_columns"],
            forecast_profiles={"Alpha": profile},
            ct_holidays=holidays.US(subdiv="CT"),
            steps=3,
            target_start_date=profile["date"].max() + pd.Timedelta(days=1),
            feature_artifacts={"schema_version": "2.1", "forecast": forecast_artifacts},
        )
        self.assertTrue(np.allclose(output["Predicted Count"], 1.0))

    def test_v2_risk_output_keeps_app_fields_and_handles_unknowns(self):
        training = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "city": ["Alpha", "Beta"],
                "location_area": ["School", "Bank"],
                "hour": [8, 12],
                "population": [10_000, 8_000],
                "total_officers": [20, 15],
                "officers_per_1000_people": [2.0, 1.9],
                "crime_rate_per_1000_people": [15.0, 12.0],
            }
        )
        artifacts = fit_classification_artifacts(training)
        broad_encoder = LabelEncoder().fit(["Other", "Property", "Violent"])
        specific_encoder = LabelEncoder().fit(["Assault", "Fraud", "Theft"])
        context = {
            "feature_artifacts": {"schema_version": "2.0", "classification": artifacts},
            "city_stats_lookup": {}, "loc_total_lookup": {}, "avg_loc_lookup": {},
            "hour_typical_lookup": {}, "avg_div_lookup": {},
            "loc_type_cats": ["commercial", "education", "other"], "htc_cats": ["Other"],
            "city_cats": ["Alpha", "Beta"], "loc_cats": ["School", "Bank"],
            "ct_holidays": holidays.US(subdiv="CT"),
            "broad_classifier": ConstantClassifier([0.6, 0.3, 0.1]),
            "broad_label_encoder": broad_encoder,
            "classifier": ConstantClassifier([0.2, 0.5, 0.3]),
            "label_encoder": specific_encoder,
            "classifier_features": artifacts["feature_columns"],
            "per_city_classifiers": {},
        }
        result = predict_crime_risk(
            "Unknown City", "Unknown Place", "other", 9, pd.Timestamp("2025-01-01"), context
        )
        expected = {
            "broad_label", "broad_probability", "broad_probs", "broad_model_classes",
            "specific_label", "specific_probability", "specific_probs", "specific_model_classes",
            "model_source",
        }
        self.assertTrue(expected.issubset(result))
        self.assertEqual(result["model_source"], "Statewide")


class AssetLoadingTests(unittest.TestCase):
    def test_git_lfs_pointer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 10\n")
            self.assertTrue(data.is_lfs_pointer(path))

    def test_missing_optional_asset_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(data, "MODELS_DIR", Path(directory)), patch.object(
                data, "hf_hub_download", side_effect=FileNotFoundError("missing")
            ):
                self.assertIsNone(data.resolve_asset_path("optional.pkl", required=False))

    def test_missing_required_asset_has_readable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(data, "MODELS_DIR", Path(directory)), patch.object(
                data, "hf_hub_download", side_effect=FileNotFoundError("missing")
            ), patch.object(data.st, "error") as error, patch.object(
                data.st, "exception"
            ), patch.object(data.st, "stop", side_effect=RuntimeError("stopped")):
                with self.assertRaisesRegex(RuntimeError, "stopped"):
                    data.resolve_asset_path("required.pkl", required=True)
                self.assertIn("Could not load required asset", error.call_args.args[0])

    def test_compact_bundle_supports_ui_lookups_and_distribution(self):
        bundle = {
            "daily_city": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-01"]),
                    "city": ["Alpha"],
                    "population": [10_000],
                    "total_officers": [20],
                    "officers_per_1000_people": [2.0],
                    "crime_rate_per_1000_people": [12.0],
                }
            ),
            "location_areas": ["School"],
            "legacy_lookups": {
                "loc_total_lookup": {"School": 12},
                "hour_typical_lookup": {("Alpha", 8): "Theft"},
                "avg_div_lookup": {("Alpha", 8): 2.5},
                "avg_loc_lookup": {"School": 3.0},
                "htc_cats": ["Theft"],
            },
        }
        lookups = data.build_bundle_lookup_tables(bundle)
        self.assertEqual(lookups["city_cats"], ["Alpha"])
        self.assertEqual(lookups["loc_cats"], ["School"])
        self.assertEqual(lookups["loc_total_lookup"]["School"], 12)
        self.assertEqual(lookups["hour_typical_lookup"][("Alpha", 8)], "Theft")

        distribution = pd.DataFrame(
            {
                "city": ["Alpha", "Alpha"],
                "year": [2024, 2024],
                "offense_category_name": ["Assault", "Theft"],
                "count": [2, 5],
            }
        )
        result = data.get_crime_distribution(distribution, "Alpha", 2024)
        self.assertEqual(result.iloc[0].to_dict(), {"Crime Type": "Theft", "Count": 5})

    def test_split_city_model_loads_only_indexed_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "alpha.pkl"
            index_path = root / "index.json"
            joblib.dump(ConstantRegressor(), model_path)
            index_path.write_text(json.dumps({
                "forecasters": {"Alpha": "per_city/forecasters/alpha.pkl"},
                "classifiers": {},
                "forecast_features": ["lag_1"],
            }))

            def resolve(filename, required=False):
                paths = {
                    data.PER_CITY_INDEX_PATH: index_path,
                    "per_city/forecasters/alpha.pkl": model_path,
                }
                return paths.get(filename)

            data.load_per_city_index.clear()
            data.load_split_city_model.clear()
            try:
                with patch.object(data, "resolve_asset_path", side_effect=resolve):
                    model = data.load_split_city_model("forecasters", "Alpha")
                    self.assertIsInstance(model, ConstantRegressor)
                    self.assertIsNone(data.load_split_city_model("forecasters", "Beta"))
            finally:
                data.load_per_city_index.clear()
                data.load_split_city_model.clear()


class FrontendPreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (PROJECT_ROOT / "app.py").read_text()
        cls.tree = ast.parse(cls.source)

    def test_app_parses_and_preserves_three_tabs(self):
        self.assertIsInstance(self.tree, ast.Module)
        self.assertIn('st.tabs(["📈 Volume Forecast", "🔍 Risk Analysis", "👮 Officer Trends"])', self.source)

    def test_theme_branding_and_comparison_are_preserved(self):
        for text in [
            'st.title("🚓 ProjeCT 360")',
            '["Cream", "Dark"]',
            '"Compare with up to 2 cities"',
            '"Statewide", "City-Specific"',
        ]:
            self.assertIn(text, self.source)

    def test_app_validates_saved_forecast_strategy_before_inference(self):
        for text in [
            "validate_forecast_contract",
            "FORECAST_PREDICTION_MODES",
            "Forecast model weight must be between 0 and 1",
            "but the app requires",
        ]:
            self.assertIn(text, self.source)

    def test_existing_widget_keys_are_preserved(self):
        for key in [
            "theme_mode", "sb_city_select", "sidebar_compare_cities", "risk_date_input",
            "risk_time_slider", "risk_loc_area_select", "risk_compare_selected_cities",
            "risk_calculate_button", "pie_year_select", "officer_view_mode",
        ]:
            self.assertIn(key, self.source)


class NotebookSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((PROJECT_ROOT / "gc_train_model.ipynb").read_text())
        cls.source = "\n".join("".join(cell.get("source", [])) for cell in cls.notebook["cells"])

    def test_notebook_has_no_saved_results(self):
        output_count = sum(len(cell.get("outputs", [])) for cell in self.notebook["cells"])
        self.assertEqual(output_count, 0)

    def test_final_test_is_explicitly_locked(self):
        self.assertIn("RUN_FINAL_TEST = False", self.source)
        self.assertIn("Final test is locked", self.source)
        self.assertIn("FINAL_TEST_START_DATE", self.source)

    def test_calibration_uses_validation_slice_not_test(self):
        self.assertIn("X_cal_c", self.source)
        self.assertNotIn("calibration_candidates(\n    selected_specific_base, X_test", self.source)

    def test_required_colab_and_model_comparison_sections_exist(self):
        for text in [
            "RUN_FULL_TUNING", "RUN_PER_CITY_EXPERIMENTS", "UPLOAD_TO_HUGGING_FACE",
            "LightGBM Poisson", "LightGBM Tweedie", "CatBoost Poisson", "XGBoost Count",
            "LightGBM L1", "LightGBM Residual L1", "CatBoost MAE",
            "recursive_forecast_backtest", "select_forecast_blend_weight",
            "FORECAST_HORIZON = 30", "Multinomial Logistic Regression",
            "model_manifest.json", "model_comparison.csv",
        ]:
            self.assertIn(text, self.source)

    def test_forecast_failure_does_not_prevent_classification_validation(self):
        self.assertIn("FORECAST_VALIDATION_PASSED =", self.source)
        self.assertIn("Classification may continue", self.source)
        self.assertNotIn("No forecast candidate beats the strongest validation baseline", self.source)

    def test_final_test_tasks_have_independent_validation_gates(self):
        self.assertIn('validation_eligibility = {', self.source)
        self.assertIn('Forecast final test skipped because it did not beat validation baseline.', self.source)
        self.assertIn('if specific_test_probabilities is not None:', self.source)

    def test_export_verifies_saved_forecast_strategy(self):
        for text in [
            'reloaded_forecast_contract["prediction_mode"] in FORECAST_PREDICTION_MODES',
            'float(reloaded_forecast_contract["model_weight"])',
            'int(reloaded_forecast_contract["forecast_horizon"]) == FORECAST_HORIZON',
        ]:
            self.assertIn(text, self.source)


if __name__ == "__main__":
    unittest.main()
