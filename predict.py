from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from feature_engineering import (
    UNIVERSITY_CITIES,
    apply_forecast_strategy,
    build_classification_features,
    build_forecast_inference_row,
    get_payday_features,
    infer_location_type,
)

# Keep the public helper name used by app.py and the existing tests.
get_location_type = infer_location_type


def compute_blended_prediction(model_pred, analog_pred, recent_pred, seasonal_pred, buffer):
    """Blend model and historical signals using the backtested weights."""
    blend = 0.60 * model_pred + 0.25 * analog_pred + 0.10 * recent_pred + 0.05 * seasonal_pred
    trend = 0.0
    if len(buffer) >= 14:
        trend = (float(np.mean(buffer[-7:])) - float(np.mean(buffer[-14:-7]))) * 0.08
    return max(0, blend + trend)


def decode_model_class(label_encoder, model_classes, class_position):
    """Decode a predict_proba column position using the estimator's class IDs."""
    if len(model_classes) == 0:
        return ''
    class_id = model_classes[class_position]
    try:
        return label_encoder.inverse_transform([class_id])[0]
    except (TypeError, ValueError):
        return str(class_id)


def prepare_forecast_history(city_profile, target_start_date):
    target_ts = pd.Timestamp(target_start_date)
    # Leakage guard: all historical blend components are computed only from real
    # rows strictly before the forecast origin. This prevents backtests from
    # accidentally using same-day or future holdout rows.
    return city_profile[city_profile['date'] < target_ts].copy()


def run_forecast_loop(
    city,
    forecaster,
    forecast_features,
    per_city_forecasters,
    per_city_forecast_features,
    forecast_profiles,
    ct_holidays,
    steps=30,
    target_start_date=None,
    feature_artifacts=None,
):
    city_profile = forecast_profiles.get(city)
    if city_profile is None or city_profile.empty:
        return None

    if target_start_date is None:
        target_start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        target_start_date = pd.Timestamp(target_start_date).to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
    target_dates = [target_start_date + timedelta(days=i) for i in range(steps)]

    city_history = prepare_forecast_history(city_profile, target_start_date)
    if city_history.empty:
        return None

    latest_stats = city_history.iloc[-1][
        ['population', 'total_officers', 'officers_per_1000_people', 'crime_rate_per_1000_people']
    ]
    city_avg = float(city_history['crime_count'].mean())
    seasonal_profile = city_history.groupby(['month', 'day_of_week'])['crime_count'].mean()
    weekday_profile = city_history.groupby('day_of_week')['crime_count'].mean()

    seed_months = {target_dates[0].month, target_dates[-1].month}
    seasonal_seed = city_history[city_history['month'].isin(seed_months)].tail(30)
    if len(seasonal_seed) < 7:
        seasonal_seed = city_history.tail(30)

    buffer = seasonal_seed['crime_count'].tail(14).astype(float).tolist()
    if len(buffer) < 7:
        buffer = city_history['crime_count'].tail(7).astype(float).tolist()

    use_city_model = city in per_city_forecasters
    model = per_city_forecasters[city] if use_city_model else forecaster
    features = per_city_forecast_features if use_city_model else forecast_features
    forecast_artifacts = None
    if feature_artifacts and str(feature_artifacts.get('schema_version', '')).startswith('2'):
        forecast_artifacts = feature_artifacts.get('forecast')
    recursive_history = city_history[['date', 'crime_count']].copy()

    future_dates = []
    future_values = []
    is_holiday_list = []
    lower_bounds = []
    upper_bounds = []

    for current_date in target_dates:
        buf = buffer[-7:]
        lag1 = buf[-1] if len(buf) >= 1 else 0.0
        lag3 = buf[-3] if len(buf) >= 3 else lag1
        lag7 = buf[-7] if len(buf) >= 7 else buf[0]
        roll7 = float(np.mean(buf)) if buf else 0.0
        std7 = float(np.std(buf)) if len(buf) >= 2 else 0.0
        max7 = float(max(buf)) if buf else 0.0
        min7 = float(min(buf)) if buf else 0.0
        dow = current_date.weekday()
        month = current_date.month
        is_hol = 1 if current_date in ct_holidays else 0
        is_before_hol = 1 if (current_date + timedelta(days=1)) in ct_holidays else 0
        is_after_hol = 1 if (current_date - timedelta(days=1)) in ct_holidays else 0

        if forecast_artifacts:
            X_in = build_forecast_inference_row(
                city,
                current_date,
                recursive_history,
                latest_stats.to_dict(),
                forecast_artifacts,
            )
        else:
            row = {
                'lag_1': lag1, 'lag_3': lag3, 'lag_7': lag7,
                'roll_mean_7': roll7, 'rolling_mean_7': roll7,
                'roll_std_7': std7, 'rolling_std_7': std7,
                'roll_max_7': max7, 'roll_min_7': min7,
                'day_of_week': dow, 'month': month,
                'quarter': (month - 1) // 3 + 1,
                'day_of_year': current_date.timetuple().tm_yday,
                'is_weekend': int(dow in [5, 6]),
                'is_monday': int(dow == 0),
                'is_friday': int(dow == 4),
                'month_sin': np.sin(2 * np.pi * month / 12),
                'month_cos': np.cos(2 * np.pi * month / 12),
                'dow_sin': np.sin(2 * np.pi * dow / 7),
                'dow_cos': np.cos(2 * np.pi * dow / 7),
                'is_holiday': is_hol,
                'is_day_before_holiday': is_before_hol,
                'is_day_after_holiday': is_after_hol,
                'population': latest_stats['population'],
                'total_officers': latest_stats['total_officers'],
                'officers_per_1000_people': latest_stats['officers_per_1000_people'],
                'crime_rate_per_1000_people': latest_stats['crime_rate_per_1000_people'],
            }
            X_in = pd.DataFrame([row])[features]
        raw_model_pred = float(model.predict(X_in)[0])
        model_pred = (
            float(apply_forecast_strategy([raw_model_pred], X_in, forecast_artifacts)[0])
            if forecast_artifacts
            else max(0, raw_model_pred)
        )

        day_of_year = current_date.timetuple().tm_yday
        seasonal_key = (month, dow)
        seasonal_pred = float(seasonal_profile.get(seasonal_key, weekday_profile.get(dow, city_avg)))
        doy_distance = (city_history['day_of_year'] - day_of_year).abs()
        circular_doy_distance = np.minimum(doy_distance, 366 - doy_distance)
        analog_window = city_history[circular_doy_distance <= 14]['crime_count']
        analog_pred = float(analog_window.mean()) if not analog_window.empty else seasonal_pred
        # Backtest finding: same-weekday history had perfect lag-7 autocorrelation
        # and reintroduced the weekly echo. Use recent city level instead.
        recent_city_level = city_history['crime_count'].tail(14)
        recent_pred = float(recent_city_level.mean()) if not recent_city_level.empty else seasonal_pred

        blend = compute_blended_prediction(model_pred, analog_pred, recent_pred, seasonal_pred, buffer)
        # Backtest policy: per-city forecasters beat the blend on MAE and had
        # lower lag-7 autocorrelation, while global fallback cities benefited
        # from blending. Use the blend only when falling back to the global model.
        pred = model_pred if forecast_artifacts or use_city_model else blend

        future_dates.append(current_date)
        future_values.append(pred)
        is_holiday_list.append(is_hol)
        buffer.append(pred)
        recursive_history = pd.concat(
            [recursive_history, pd.DataFrame({'date': [current_date], 'crime_count': [pred]})],
            ignore_index=True,
        )
        residuals = forecast_artifacts.get('residual_quantiles', {}) if forecast_artifacts else {}
        lower_bounds.append(max(0, pred + float(residuals.get('lower', 0.0))))
        upper_bounds.append(max(0, pred + float(residuals.get('upper', 0.0))))

    output = pd.DataFrame({'Date': future_dates, 'Predicted Count': future_values, 'is_holiday': is_holiday_list})
    if forecast_artifacts and forecast_artifacts.get('residual_quantiles'):
        output['Lower Bound'] = lower_bounds
        output['Upper Bound'] = upper_bounds
    return output


def predict_crime_risk(
    city,
    location_area,
    location_type,
    hour,
    input_date,
    classifier_context,
):
    ts = pd.Timestamp(input_date)
    selected_location_type = str(location_type or get_location_type(location_area)).lower()
    feature_artifacts = classifier_context.get('feature_artifacts')
    classification_artifacts = None
    if feature_artifacts and str(feature_artifacts.get('schema_version', '')).startswith('2'):
        classification_artifacts = feature_artifacts.get('classification')
    if classification_artifacts:
        raw_context = pd.DataFrame([{
            'date': ts,
            'city': city,
            'location_area': location_area,
            'location_type': selected_location_type,
            'hour': hour,
        }])
        in_data = build_classification_features(raw_context, classification_artifacts)
    else:
        dow = ts.dayofweek
        month = ts.month
        dom = ts.day
        week = int(ts.isocalendar()[1])
        is_payday, days_from_payday = get_payday_features(ts)
        city_stats = classifier_context['city_stats_lookup'].get(str(city), {})
        htc = str(classifier_context['hour_typical_lookup'].get((str(city), int(hour)), 'Other'))
        if selected_location_type not in classifier_context['loc_type_cats']:
            selected_location_type = 'other'
        if htc not in classifier_context['htc_cats']:
            htc = classifier_context['htc_cats'][0] if classifier_context['htc_cats'] else 'Other'
        row = {
            'city': city,
            'location_area': location_area,
            'location_type': selected_location_type,
            'hour': hour,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'dayofweek': dow,
            'dow_sin': np.sin(2 * np.pi * dow / 7),
            'dow_cos': np.cos(2 * np.pi * dow / 7),
            'month': month,
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
            'day_of_month': dom,
            'day_of_month_sin': np.sin(2 * np.pi * dom / 31),
            'day_of_month_cos': np.cos(2 * np.pi * dom / 31),
            'week_of_year': week,
            'week_sin': np.sin(2 * np.pi * week / 52),
            'week_cos': np.cos(2 * np.pi * week / 52),
            'is_weekend': int(dow in [5, 6]),
            'is_holiday': int(ts in classifier_context['ct_holidays']),
            'is_payday': is_payday,
            'days_from_payday': days_from_payday,
            'is_university': int(city in UNIVERSITY_CITIES),
            'crime_diversity': float(classifier_context['avg_div_lookup'].get((str(city), int(hour)), 1.5)),
            'location_daily_freq': float(classifier_context['avg_loc_lookup'].get(str(location_area), 5.0)),
            'location_total_crimes': float(classifier_context['loc_total_lookup'].get(str(location_area), 100.0)),
            'population': float(city_stats.get('population', 0.0)),
            'total_officers': float(city_stats.get('total_officers', 0.0)),
            'officers_per_1000_people': float(city_stats.get('officers_per_1000_people', 0.0)),
            'crime_rate_per_1000_people': float(city_stats.get('crime_rate_per_1000_people', 0.0)),
            'hour_typical_crime': htc,
        }
        classifier_features = classifier_context['classifier_features']
        in_data = pd.DataFrame([row])[classifier_features]
        cat_map = {
            'city': classifier_context['city_cats'],
            'location_area': classifier_context['loc_cats'],
            'location_type': classifier_context['loc_type_cats'],
            'hour_typical_crime': classifier_context['htc_cats'],
        }
        for col, cats in cat_map.items():
            if col in in_data.columns:
                values = in_data[col].astype(str)
                if col == 'location_type':
                    values = values.str.lower()
                fallback = 'other' if col == 'location_type' and 'other' in cats else (cats[0] if cats else '')
                values = values.where(values.isin(cats), fallback)
                in_data[col] = pd.Categorical(values, categories=cats).codes.astype(int)

    broad_classifier = classifier_context['broad_classifier']
    broad_label_encoder = classifier_context['broad_label_encoder']
    broad_probs = broad_classifier.predict_proba(in_data)[0]
    broad_top_idx = int(np.argmax(broad_probs))
    broad_model_classes = getattr(broad_classifier, 'classes_', np.arange(len(broad_probs)))

    per_city_classifiers = classifier_context['per_city_classifiers']
    city_specific_model = per_city_classifiers.get(city)
    if city_specific_model is not None:
        specific_model = city_specific_model
        model_source = 'City-specific'
    else:
        specific_model = classifier_context['classifier']
        model_source = 'Statewide'

    label_encoder = classifier_context['label_encoder']
    specific_probs = specific_model.predict_proba(in_data)[0]
    specific_top_idx = int(np.argmax(specific_probs))
    specific_model_classes = getattr(specific_model, 'classes_', np.arange(len(specific_probs)))

    result = {
        'broad_probs': broad_probs,
        'broad_model_classes': broad_model_classes,
        'broad_label': decode_model_class(broad_label_encoder, broad_model_classes, broad_top_idx),
        'broad_probability': float(broad_probs[broad_top_idx]),
        'specific_probs': specific_probs,
        'specific_model_classes': specific_model_classes,
        'specific_label': decode_model_class(label_encoder, specific_model_classes, specific_top_idx),
        'specific_probability': float(specific_probs[specific_top_idx]),
        'model_source': model_source,
    }
    if feature_artifacts:
        result['model_version'] = feature_artifacts.get('model_version')
        result['low_confidence'] = bool(result['specific_probability'] < 0.50)
    return result
