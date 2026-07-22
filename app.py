import os
import pickle
import logging
import traceback
from html import escape
from datetime import datetime, timedelta

# 1. FORCE CPU MODE
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import holidays
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from feature_engineering import FORECAST_PREDICTION_MODES
from data import (
    build_bundle_lookup_tables,
    build_forecast_profiles,
    build_lookup_tables,
    get_aggregate_data,
    get_crime_distribution,
    get_officer_trends,
    is_lfs_pointer,
    load_app_data_bundle,
    load_data,
    load_model_manifest,
    load_per_city_index,
    load_split_city_model,
    resolve_asset_path,
)
from predict import decode_model_class, get_location_type, predict_crime_risk, run_forecast_loop


ct_holidays = holidays.US(subdiv='CT')
FORECAST_DAYS = 30
DEPLOY_LIGHT_MODE = os.getenv("PROJECT360_DEPLOY_LIGHT", "true").lower() not in {"0", "false", "no"}
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def report_tab_error(context, exc):
    """Show a readable app error and write the full traceback to Streamlit logs."""
    logger.exception("%s failed", context)
    st.error(f"{context} failed: {type(exc).__name__}: {exc}")
    with st.expander("Technical details"):
        st.code(traceback.format_exc())


def render_risk_summary_card(label, value, probability):
    st.markdown(
        f"""
        <div class="risk-summary-card">
            <div class="risk-summary-label">{escape(str(label))}</div>
            <div class="risk-summary-value">{escape(str(value))}</div>
            <div class="risk-summary-probability">↑ {probability:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_probability_frame(city, result, label_encoder, probs_key, classes_key, label_col, top_n=None):
    probs = result[probs_key]
    model_classes = result.get(classes_key, range(len(probs)))
    rows = []
    if top_n is None:
        positions = range(len(probs))
    else:
        positions = probs.argsort()[-top_n:][::-1]
    for pos in positions:
        rows.append({
            'City': city,
            label_col: decode_model_class(label_encoder, model_classes, int(pos)),
            'Probability': float(probs[int(pos)]),
        })
    return pd.DataFrame(rows)


def apply_theme(theme_mode):
    if theme_mode == "Dark":
        colors = {
            "background": "#0F172A",
            "secondary": "#1E293B",
            "surface": "#111827",
            "text": "#E5E7EB",
            "muted": "#CBD5E1",
            "border": "#334155",
            "accent": "#38BDF8",
            "grid": "#334155",
            "input": "#243244",
        }
        px.defaults.template = "plotly_dark"
    else:
        colors = {
            "background": "#FFF7ED",
            "secondary": "#FEF3C7",
            "surface": "#FFF7ED",
            "text": "#111827",
            "muted": "#374151",
            "border": "#FED7AA",
            "accent": "#0F766E",
            "grid": "#E5E7EB",
            "input": "#FFFBEB",
        }
        px.defaults.template = "plotly_white"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {colors["background"]};
            color: {colors["text"]};
        }}
        [data-testid="stSidebar"] {{
            background-color: {colors["secondary"]};
            border-right: 1px solid {colors["border"]};
        }}
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] h5,
        [data-testid="stSidebar"] h6,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {{
            color: {colors["text"]} !important;
        }}
        [data-testid="stHeader"] {{
            background-color: {colors["background"]};
        }}
        h1, h2, h3, h4, h5, h6, p, label, span {{
            color: {colors["text"]};
        }}
        [data-testid="stCaptionContainer"] p {{
            color: {colors["muted"]};
        }}
        div[data-testid="stMetric"] {{
            background-color: {colors["secondary"]};
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            padding: 0.75rem;
        }}
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {{
            color: {colors["text"]};
        }}
        .risk-summary-card {{
            background-color: {colors["secondary"]};
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            padding: 1rem;
            min-height: 9rem;
        }}
        .risk-summary-label {{
            color: {colors["text"]};
            font-size: 1rem;
            line-height: 1.2;
            margin-bottom: 0.75rem;
        }}
        .risk-summary-value {{
            color: {colors["text"]};
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.1;
            overflow-wrap: anywhere;
            word-break: normal;
        }}
        .risk-summary-probability {{
            display: inline-block;
            background-color: rgba(187, 247, 208, 0.8);
            color: #052E16;
            border-radius: 999px;
            font-size: 0.95rem;
            margin-top: 0.75rem;
            padding: 0.25rem 0.55rem;
        }}
        @media (max-width: 900px) {{
            .risk-summary-value {{
                font-size: 1.6rem;
            }}
        }}
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] input {{
            background-color: {colors["input"]};
            color: {colors["text"]};
            border-color: {colors["border"]};
        }}
        div[data-baseweb="select"] * {{
            color: {colors["text"]} !important;
        }}
        div[data-baseweb="select"] svg {{
            fill: {colors["text"]} !important;
            color: {colors["text"]} !important;
        }}
        div[data-baseweb="select"] [class*="placeholder"],
        div[data-baseweb="select"] [aria-disabled="true"] {{
            color: {colors["muted"]} !important;
            opacity: 1 !important;
        }}
        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] li * {{
            color: {colors["text"]} !important;
            background-color: {colors["secondary"]};
        }}
        div[data-baseweb="popover"] {{
            background-color: {colors["secondary"]};
            color: {colors["text"]};
        }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            color: {colors["accent"]};
        }}
        [data-testid="stAlert"] {{
            background-color: {colors["secondary"]};
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            color: {colors["text"]};
        }}
        [data-testid="stAlert"] > div,
        [data-testid="stAlert"] [data-testid="stAlertContent"] {{
            background-color: {colors["secondary"]} !important;
            color: {colors["text"]} !important;
        }}
        [data-testid="stAlert"] svg {{
            fill: {colors["accent"]} !important;
            color: {colors["accent"]} !important;
        }}
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] *,
        [data-testid="stAlert"] li,
        [data-testid="stAlert"] li *,
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] span {{
            color: {colors["text"]} !important;
        }}
        [data-testid="stExpander"] {{
            background-color: {colors["secondary"]};
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            overflow: hidden;
        }}
        [data-testid="stExpander"] details {{
            border: 0;
        }}
        [data-testid="stExpander"] summary {{
            background-color: {colors["input"]};
            border-bottom: 1px solid {colors["border"]};
        }}
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary *,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"],
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] * {{
            color: {colors["text"]} !important;
        }}
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] li,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] li * {{
            color: {colors["muted"]} !important;
        }}
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] strong,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] h4 {{
            color: {colors["text"]} !important;
        }}
        [data-testid="stExpander"] hr {{
            border-color: {colors["border"]} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return colors


def style_plotly_figure(fig, theme):
    if not fig.layout.title.text:
        fig.update_layout(title_text="")
    fig.update_layout(
        paper_bgcolor=theme["surface"],
        plot_bgcolor=theme["surface"],
        font=dict(color=theme["text"]),
        title_font=dict(color=theme["text"]),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme["text"]),
        ),
        margin=dict(l=16, r=16, t=56, b=24),
    )
    fig.update_xaxes(
        color=theme["text"],
        gridcolor=theme["grid"],
        zerolinecolor=theme["grid"],
        title_font=dict(color=theme["text"]),
    )
    fig.update_yaxes(
        color=theme["text"],
        gridcolor=theme["grid"],
        zerolinecolor=theme["grid"],
        title_font=dict(color=theme["text"]),
    )
    return fig


def build_probability_bar_chart(df, y_col, theme, show_legend):
    fig = px.bar(
        df,
        x='Probability',
        y=y_col,
        orientation='h',
        color='City',
        custom_data=['City'],
        color_discrete_sequence=px.colors.qualitative.Set2,
        barmode='group',
        text_auto='.2%',
    )
    fig.update_traces(hovertemplate='%{customdata[0]}<br>%{y}<br>Probability=%{x:.2%}<extra></extra>')
    fig.update_layout(
        height=320,
        showlegend=show_legend,
        xaxis_tickformat='.2%',
        yaxis={'categoryorder': 'total ascending'},
    )
    style_plotly_figure(fig, theme)
    return fig


st.set_page_config(page_title="CT Crime Insight 360", page_icon="🚓", layout="wide")
st.sidebar.header("🌍 Settings")
theme_mode = st.sidebar.radio("Theme", ["Cream", "Dark"], horizontal=True, key="theme_mode")
theme = apply_theme(theme_mode)
st.title("🚓 ProjeCT 360")
st.markdown("### Crime, Population & Officer Awareness")


def load_required_model(filename):
    """Load a required joblib asset or stop with a deployment-friendly error."""
    path = resolve_asset_path(filename, required=True)
    try:
        if is_lfs_pointer(path):
            st.error(
                f"Required model file `{filename}` resolved to a Git LFS pointer, not the real file. "
                "Upload the real file to Hugging Face and confirm the filename matches."
            )
            st.stop()
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"Required model file not found: `{filename}`")
        st.stop()
    except (
        KeyError,
        pickle.UnpicklingError,
        EOFError,
        ValueError,
        TypeError,
        AttributeError,
        ImportError,
        OSError,
    ) as exc:
        st.error(
            f"Required model file `{filename}` could not be loaded. "
            "The file may be incomplete or the deployment Python/package versions may not match "
            f"the training environment. Details: {type(exc).__name__}"
        )
        st.stop()


def load_optional_model(filename, fallback, feature_name, warnings):
    """Load an optional joblib asset, recording why its feature was disabled."""
    path = resolve_asset_path(filename, required=False)
    if path is None:
        warnings.append(f"{feature_name} disabled: missing `{filename}`.")
        return fallback
    try:
        if is_lfs_pointer(path):
            warnings.append(f"{feature_name} disabled: `{filename}` is a Git LFS pointer, not the real file.")
            return fallback
        return joblib.load(path)
    except FileNotFoundError:
        warnings.append(f"{feature_name} disabled: missing `{filename}`.")
    except (
        KeyError,
        pickle.UnpicklingError,
        EOFError,
        ValueError,
        TypeError,
        AttributeError,
        ImportError,
        OSError,
    ) as exc:
        warnings.append(f"{feature_name} disabled: `{filename}` could not be loaded ({type(exc).__name__}).")
    return fallback


@st.cache_resource
def load_feature_artifacts():
    """Load the shared v2 feature contract when retrained artifacts are available."""
    path = resolve_asset_path('feature_artifacts.pkl', required=False)
    if path is None or is_lfs_pointer(path):
        return None
    try:
        feature_artifacts = joblib.load(path)
        manifest = load_model_manifest()
        if manifest:
            artifact_version = str(feature_artifacts.get('model_version', ''))
            manifest_version = str(manifest.get('model_version', ''))
            if artifact_version and manifest_version and artifact_version != manifest_version:
                st.error('Model metadata does not match the downloaded feature artifacts.')
                st.stop()
            feature_artifacts['model_manifest'] = manifest
        return feature_artifacts
    except (
        FileNotFoundError,
        KeyError,
        pickle.UnpicklingError,
        EOFError,
        ValueError,
        TypeError,
        AttributeError,
        ImportError,
        OSError,
    ):
        return None


def validate_forecast_contract(feature_artifacts, forecast_features):
    """Reject incompatible forecast metadata before the first UI prediction."""
    forecast_artifacts = feature_artifacts.get('forecast', {})
    expected_features = list(forecast_artifacts.get('feature_columns', []))
    if not expected_features:
        raise ValueError('Feature artifacts do not define the forecast feature schema.')
    if list(forecast_features) != expected_features:
        raise ValueError('The forecaster feature list does not match `feature_artifacts.pkl`.')

    prediction_mode = str(forecast_artifacts.get('prediction_mode', 'direct'))
    if prediction_mode not in FORECAST_PREDICTION_MODES:
        raise ValueError(f'Unsupported forecast prediction mode: {prediction_mode}.')

    model_weight = float(forecast_artifacts.get('model_weight', 1.0))
    if not 0.0 <= model_weight <= 1.0:
        raise ValueError('Forecast model weight must be between 0 and 1.')

    forecast_horizon = int(forecast_artifacts.get('forecast_horizon', FORECAST_DAYS))
    if forecast_horizon != FORECAST_DAYS:
        raise ValueError(
            f'The saved forecaster targets {forecast_horizon} days, but the app requires {FORECAST_DAYS}.'
        )


@st.cache_resource
def load_forecast_models():
    optional_warnings = []

    forecaster = load_required_model('crime_forecaster.pkl')
    forecast_features = load_required_model('forecast_features.pkl')
    feature_artifacts = load_feature_artifacts()
    if 'city_code' in forecast_features and feature_artifacts is None:
        st.error('The version 2 forecaster requires `feature_artifacts.pkl`, but that file is unavailable.')
        st.stop()
    if feature_artifacts is not None:
        try:
            validate_forecast_contract(feature_artifacts, forecast_features)
        except (KeyError, TypeError, ValueError) as exc:
            st.error(f'The saved forecast artifacts are incompatible: {exc}')
            st.stop()

    if DEPLOY_LIGHT_MODE:
        per_city_forecasters = {}
        per_city_forecast_features = forecast_features
    elif feature_artifacts is not None:
        per_city_forecasters = {}
        per_city_forecast_features = forecast_features
    else:
        per_city_forecasters = load_optional_model(
            'per_city_forecasters.pkl', {}, 'Per-city volume forecasting', optional_warnings
        )
        per_city_forecast_features = load_optional_model(
            'per_city_forecast_features.pkl', None, 'Per-city volume forecasting', optional_warnings
        )
        if per_city_forecasters and per_city_forecast_features is None:
            optional_warnings.append(
                'Per-city volume forecasting disabled: forecasters were found but feature schema is missing.'
            )
            per_city_forecasters = {}
        elif per_city_forecast_features is None:
            per_city_forecast_features = forecast_features

    return {
        'forecaster': forecaster,
        'forecast_features': forecast_features,
        'per_city_forecasters': per_city_forecasters,
        'per_city_forecast_features': per_city_forecast_features,
        'feature_artifacts': feature_artifacts,
        'optional_warnings': optional_warnings,
    }


@st.cache_resource
def load_risk_models():
    optional_warnings = []

    broad_classifier = load_required_model('crime_classifier_l1_violent_property.pkl')
    broad_label_encoder = load_required_model('label_encoder_l1.pkl')
    classifier = load_required_model('crime_classifier_l2_specific.pkl')
    label_encoder = load_required_model('label_encoder_l2.pkl')
    classifier_features = load_required_model('advanced_features.pkl')
    feature_artifacts = load_feature_artifacts()
    if 'city_code' in classifier_features and feature_artifacts is None:
        st.error('The version 2 classifiers require `feature_artifacts.pkl`, but that file is unavailable.')
        st.stop()

    if DEPLOY_LIGHT_MODE:
        per_city_classifiers = {}
    elif feature_artifacts is not None:
        per_city_classifiers = {}
    else:
        per_city_classifiers = load_optional_model(
            'per_city_models.pkl', {}, 'City-specific risk classification', optional_warnings
        )

    return {
        'broad_classifier': broad_classifier,
        'broad_label_encoder': broad_label_encoder,
        'classifier': classifier,
        'label_encoder': label_encoder,
        'classifier_features': classifier_features,
        'per_city_classifiers': per_city_classifiers,
        'feature_artifacts': feature_artifacts,
        'optional_warnings': optional_warnings,
    }


def forecast_for_city(city, start_date, models, forecast_profiles):
    return run_forecast_loop(
        city=city,
        forecaster=models['forecaster'],
        forecast_features=models['forecast_features'],
        per_city_forecasters=models['per_city_forecasters'],
        per_city_forecast_features=models['per_city_forecast_features'],
        forecast_profiles=forecast_profiles,
        ct_holidays=ct_holidays,
        steps=FORECAST_DAYS,
        target_start_date=start_date,
        feature_artifacts=models['feature_artifacts'],
    )


def build_risk_context(models, lookups):
    return {
        **lookups,
        'ct_holidays': ct_holidays,
        'broad_classifier': models['broad_classifier'],
        'broad_label_encoder': models['broad_label_encoder'],
        'classifier': models['classifier'],
        'label_encoder': models['label_encoder'],
        'classifier_features': models['classifier_features'],
        'per_city_classifiers': models['per_city_classifiers'],
        'feature_artifacts': models['feature_artifacts'],
    }


def with_selected_city_forecasters(models, cities):
    """Attach only the requested legacy city forecasters to the cached global models."""
    if models['feature_artifacts'] is not None:
        return models
    index = load_per_city_index()
    if index is None:
        return models

    selected = dict(models['per_city_forecasters'])
    for city in cities:
        if city in selected:
            continue
        city_model = load_split_city_model('forecasters', city)
        if city_model is not None:
            selected[city] = city_model
    return {
        **models,
        'per_city_forecasters': selected,
        'per_city_forecast_features': index.get(
            'forecast_features', models['per_city_forecast_features']
        ),
    }


def with_selected_city_classifiers(models, cities):
    """Attach only requested legacy city classifiers, retaining statewide fallback."""
    if models['feature_artifacts'] is not None:
        return models
    selected = dict(models['per_city_classifiers'])
    for city in cities:
        if city in selected:
            continue
        city_model = load_split_city_model('classifiers', city)
        if city_model is not None:
            selected[city] = city_model
    return {**models, 'per_city_classifiers': selected}


with st.spinner("Booting up system..."):
    forecast_models = load_forecast_models()
    app_data_bundle = load_app_data_bundle()
    if (
        app_data_bundle is not None
        and forecast_models['feature_artifacts'] is not None
        and str(app_data_bundle['model_version'])
        != str(forecast_models['feature_artifacts'].get('model_version'))
    ):
        app_data_bundle = None
    if app_data_bundle is not None:
        ts_data = app_data_bundle['daily_city']
        officer_raw_data = app_data_bundle['officer_trends']
        crime_distribution_data = app_data_bundle['crime_distribution']
        lookups = build_bundle_lookup_tables(app_data_bundle)
        available_years = sorted(app_data_bundle['years'], reverse=True)
    else:
        raw_df = load_data()
        ts_data = get_aggregate_data(raw_df)
        officer_raw_data = get_officer_trends(raw_df)
        crime_distribution_data = raw_df
        lookups = build_lookup_tables(raw_df, include_legacy=forecast_models['feature_artifacts'] is None)
        available_years = sorted(raw_df['year'].unique(), reverse=True)
    forecast_profiles = build_forecast_profiles(ts_data)

if forecast_models['optional_warnings']:
    for warning in forecast_models['optional_warnings']:
        st.warning(warning)

cities = lookups['city_cats']
locations = lookups['loc_cats']

selected_city = st.sidebar.selectbox("Select City", cities, key="sb_city_select")
compare_options = [city for city in cities if city != selected_city]
compare_cities = st.sidebar.multiselect(
    "Compare with up to 2 cities",
    compare_options,
    max_selections=2,
    key="sidebar_compare_cities",
)
comparison_cities = [selected_city] + compare_cities
forecast_models_for_selection = with_selected_city_forecasters(forecast_models, comparison_cities)

st.sidebar.markdown("---")
st.sidebar.markdown("**📍 Selected City Stats**")
for city in comparison_cities:
    city_stats_row = ts_data[ts_data['city'].astype(str) == city].iloc[-1]
    with st.sidebar.container(border=True):
        st.markdown(f"**{city}**")
        st.metric("👮 Officers", f"{int(city_stats_row['total_officers']):,}")
        st.metric("👥 Population", f"{int(city_stats_row['population']):,}")
        st.caption(f"Officer Rate: {city_stats_row['officers_per_1000_people']:.2f} per 1k")

with st.sidebar.expander("ℹ️ About ProjeCT 360"):
    st.markdown("""
    **CT Crime ProjeCT 360** is an educational analytics dashboard that explores historical Connecticut incident-reporting patterns, staffing context, and model-generated estimates.

    #### Core Capabilities

    **1. Volume Forecasting (Regression)**
    * **Function:** Estimates expected daily incident counts for the next 30 days using historical reporting data.
    * **Evaluation:** Assessed on chronological held-out dates after model training.

    **2. Risk Classification (Probability)**
    * **Function:** Shows probability distributions across broad and specific offense categories for a selected scenario.
    * **Evaluation:** Assessed with class-aware and probability-quality metrics on chronological held-out data.

    **3. Resource Analytics**
    * **Function:** Summarizes police staffing levels and demographic trends over time.

    ---
    **Important limitations:** This tool is for learning and exploratory analysis only. It should not be used to make enforcement decisions, assess risk to any person, justify increased policing, or replace professional/legal judgment. Model outputs may reflect missing data, reporting bias, class imbalance, and historical patterns rather than real-world risk.

    *Student project note: This is an early ML project and may contain mistakes or limitations as the models and data pipeline continue to improve.*
    """)

tab1, tab2, tab3 = st.tabs(["📈 Volume Forecast", "🔍 Risk Analysis", "👮 Officer Trends"])

with tab1:
    st.subheader(f"30-Day Volume Forecast: {selected_city}")

    target_start_date = datetime.now().date()
    forecast_cities = comparison_cities
    forecast_end_date = pd.Timestamp(target_start_date) + timedelta(days=FORECAST_DAYS - 1)
    st.caption(
        f"Forecast window starts today: {pd.Timestamp(target_start_date).strftime('%b %d, %Y')} "
        f"through {forecast_end_date.strftime('%b %d, %Y')} ({FORECAST_DAYS} days)."
    )

    forecast_frames = []
    missing_forecasts = []
    with st.spinner("Calculating forecast..."):
        for city in forecast_cities:
            try:
                city_forecast = forecast_for_city(
                    city, target_start_date, forecast_models_for_selection, forecast_profiles
                )
            except Exception as exc:
                report_tab_error(f"Volume forecast for {city}", exc)
                missing_forecasts.append(city)
                continue
            if city_forecast is None:
                missing_forecasts.append(city)
                continue
            city_forecast = city_forecast.copy()
            city_forecast['City'] = city
            forecast_frames.append(city_forecast)

    if forecast_frames:
        forecast_df = pd.concat(forecast_frames, ignore_index=True)
        fig = px.line(
            forecast_df,
            x='Date',
            y='Predicted Count',
            color='City',
            markers=True,
            custom_data=['City'],
        )
        fig.update_layout(
            title="Predicted Daily Incidents (30-Day Trend)",
            xaxis_title="Date",
            yaxis=dict(title="Incident Count", tickformat=',.0f'),
        )
        fig.update_traces(
            line_width=3,
            hovertemplate='%{customdata[0]}<br>%{x|%b %d, %Y}<br>Predicted Count = %{y:.2f}<extra></extra>',
        )
        style_plotly_figure(fig, theme)

        holiday_points = forecast_df[forecast_df['is_holiday'] == 1]
        if not holiday_points.empty:
            fig.add_trace(go.Scatter(
                x=holiday_points['Date'],
                y=holiday_points['Predicted Count'],
                mode='markers',
                name='Holiday',
                marker=dict(color='gold', size=15, symbol='star', line=dict(color='black', width=1)),
                customdata=holiday_points[['City']],
                hovertemplate='%{customdata[0]} Holiday<br>%{x|%b %d, %Y}<br>Predicted Count = %{y:.2f}<extra></extra>',
            ))

        st.plotly_chart(fig, width='stretch')

        metric_cols = st.columns(len(forecast_cities))
        for i, city in enumerate(forecast_cities):
            city_total = forecast_df.loc[forecast_df['City'] == city, 'Predicted Count'].sum()
            metric_cols[i].metric(f"{city} Total", f"{round(city_total):,} Crimes")
    else:
        st.error("Insufficient data.")

    if missing_forecasts:
        st.warning(f"Insufficient forecast data for: {', '.join(missing_forecasts)}")

with tab2:
    st.subheader("Crime Type Probability & Analysis")
    st.caption(
        "Educational use only: outputs reflect historical reporting patterns and model uncertainty, not risk to any individual or a recommendation for action."
    )

    c1, c2 = st.columns(2)
    with c1:
        s_date = st.date_input("Date", datetime.now(), key="risk_date_input")
        s_time = st.slider("Hour", 0, 23, 12, key="risk_time_slider")
        s_loc = st.selectbox("Specific Location Area", locations, key="risk_loc_area_select")
        inferred_location_type = get_location_type(s_loc)
        s_loc_type = inferred_location_type
        st.text_input(
            "Location Type",
            value=s_loc_type.capitalize(),
            disabled=True,
            key=f"risk_loc_type_display_{s_loc}",
        )
        st.caption("Location type is inferred from the selected specific location.")
        compare_risk_cities = False
        if len(comparison_cities) > 1:
            compare_risk_cities = st.checkbox(
                "Compare selected cities",
                value=True,
                key="risk_compare_selected_cities",
            )
        calculate_risk = st.button("Calculate Risk", type="primary", key="risk_calculate_button")

    risk_cities = comparison_cities if compare_risk_cities else [selected_city]
    risk_results = {}
    risk_failures = []
    if calculate_risk:
        with st.spinner("Loading risk models and calculating risk..."):
            risk_models = load_risk_models()
            risk_models = with_selected_city_classifiers(risk_models, risk_cities)
            if risk_models['optional_warnings']:
                for warning in risk_models['optional_warnings']:
                    st.warning(warning)
            risk_context = build_risk_context(risk_models, lookups)
            for city in risk_cities:
                try:
                    risk_results[city] = predict_crime_risk(
                        city,
                        s_loc,
                        s_loc_type,
                        s_time,
                        s_date,
                        risk_context,
                    )
                except Exception as exc:
                    logger.exception("Risk prediction failed for %s", city)
                    risk_failures.append(f"{city}: {exc}")
    else:
        with c2:
            st.info("Choose a date, time, and location, then click **Calculate Risk**.")

    if risk_failures:
        st.warning("Risk prediction unavailable for: " + "; ".join(risk_failures))

    if risk_results:
        primary_risk_result = risk_results.get(selected_city) or next(iter(risk_results.values()))
        broad_frames = []
        specific_frames = []
        for city, result in risk_results.items():
            try:
                broad_frames.append(build_probability_frame(
                    city,
                    result,
                    risk_models['broad_label_encoder'],
                    'broad_probs',
                    'broad_model_classes',
                    'Category',
                ))
                specific_frames.append(build_probability_frame(
                    city,
                    result,
                    risk_models['label_encoder'],
                    'specific_probs',
                    'specific_model_classes',
                    'Crime Type',
                    top_n=5,
                ))
            except Exception as exc:
                report_tab_error(f"Risk chart data for {city}", exc)

        risk_charts_available = bool(broad_frames and specific_frames)
        if risk_charts_available:
            broad_chart_df = pd.concat(broad_frames, ignore_index=True).sort_values('Probability', ascending=True)
            chart_df = pd.concat(specific_frames, ignore_index=True).sort_values('Probability', ascending=True)
        else:
            st.warning("Risk chart data is unavailable for the selected city comparison.")
        with c2:
            st.info(f"""
            **Analysis Context:**
            - **Cities:** {', '.join(risk_results.keys())}
            - **Specific Location:** {s_loc}
            - **Location Type:** {s_loc_type.capitalize()}
            - **Time:** {s_date.strftime('%Y-%m-%d')} @ {s_time}:00
            - **Primary Model:** {primary_risk_result['model_source']}
            """)
            card_cols = st.columns(2)
            with card_cols[0]:
                render_risk_summary_card(
                    "Broad Category",
                    primary_risk_result['broad_label'],
                    primary_risk_result['broad_probability'],
                )
            with card_cols[1]:
                render_risk_summary_card(
                    "Top Specific",
                    primary_risk_result['specific_label'],
                    primary_risk_result['specific_probability'],
                )

        if risk_charts_available:
            st.divider()
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.markdown("#### Which broad category is most likely?")
                try:
                    fig_broad = build_probability_bar_chart(
                        broad_chart_df,
                        'Category',
                        theme,
                        show_legend=len(risk_results) > 1,
                    )
                    st.plotly_chart(fig_broad, width='stretch')
                except Exception as exc:
                    report_tab_error("Broad risk chart", exc)

            with chart_cols[1]:
                st.markdown("#### Which specific offense is most likely?")
                try:
                    fig_bar = build_probability_bar_chart(
                        chart_df,
                        'Crime Type',
                        theme,
                        show_legend=len(risk_results) > 1,
                    )
                    st.plotly_chart(fig_bar, width='stretch')
                except Exception as exc:
                    report_tab_error("Specific offense risk chart", exc)

    st.divider()
    c_pie_title, c_pie_select = st.columns([3, 1])
    with c_pie_title:
        historical_title = selected_city
        if compare_risk_cities and len(risk_results) > 1:
            historical_title = "selected cities"
        st.markdown(f"#### How does this compare historically? ({historical_title})")
    with c_pie_select:
        year_options = ["All Years"] + available_years
        selected_year = st.selectbox("Filter Year", year_options, key="pie_year_select")

    historical_cities = risk_cities if compare_risk_cities else [selected_city]
    historical_frames = []
    for city in historical_cities:
        city_dist = get_crime_distribution(crime_distribution_data, city, selected_year)
        if city_dist is not None and not city_dist.empty:
            city_dist = city_dist.copy()
            city_dist['City'] = city
            historical_frames.append(city_dist)

    if historical_frames and len(historical_frames) > 1:
        dist_data = pd.concat(historical_frames, ignore_index=True)
        try:
            fig_history = px.bar(
                dist_data,
                x='Count',
                y='Crime Type',
                color='City',
                custom_data=['City'],
                orientation='h',
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_history.update_traces(hovertemplate='%{customdata[0]}<br>%{y}<br>Count=%{x:,}<extra></extra>')
            fig_history.update_layout(
                height=520,
                xaxis_title='Reported Incidents',
                yaxis={'categoryorder': 'total ascending'},
            )
            style_plotly_figure(fig_history, theme)
            st.plotly_chart(fig_history, width='stretch')
        except Exception as exc:
            report_tab_error("Historical comparison chart", exc)
    elif historical_frames:
        dist_data = historical_frames[0]
        try:
            fig_pie = px.pie(
                dist_data,
                values='Count',
                names='Crime Type',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(showlegend=True, height=500)
            style_plotly_figure(fig_pie, theme)
            st.plotly_chart(fig_pie, width='stretch')
        except Exception as exc:
            report_tab_error("Historical distribution chart", exc)
    else:
        st.info(f"No historical distribution data available for {', '.join(historical_cities)} in {selected_year}.")

with tab3:
    st.subheader("Police Force Analysis")

    officer_view_mode = st.radio(
        "View Mode:",
        ["Statewide", "City-Specific"],
        horizontal=True,
        key="officer_view_mode",
    )
    if officer_view_mode == "Statewide":
        chart_data = (
            officer_raw_data
            .groupby('month_start')[['population', 'total_officers', 'male_officer', 'female_officer']]
            .sum()
            .reset_index()
        )
        chart_data['officers_per_1000_people'] = (
            chart_data['total_officers'] / chart_data['population'] * 1000
        )
        trend_cities = []
        title_text = "Statewide Police Force Trend"
        stat_label = "Statewide"
        gender_title = "Officer Demographics: Statewide"
    else:
        trend_cities = comparison_cities
        chart_data = officer_raw_data[
            officer_raw_data['city'].astype(str).isin(trend_cities)
        ].sort_values(['city', 'month_start'])
        title_text = "Police Force Trend: " + " vs ".join(trend_cities)
        stat_label = ", ".join(trend_cities)
        gender_title = "Officer Demographics: " + " vs ".join(trend_cities)

    if not chart_data.empty:
        st.markdown("#### 📈 Total Force Size")
        fig_total = px.line(
            chart_data,
            x='month_start',
            y='total_officers',
            color='city' if officer_view_mode == "City-Specific" else None,
            markers=True,
            custom_data=['city'] if officer_view_mode == "City-Specific" else None,
        )
        fig_total.update_layout(
            title=title_text,
            xaxis=dict(
                tickformat="%Y",
                dtick="M12",
                title="Date",
                hoverformat="%b %Y",
                tickangle=45,
            ),
            yaxis=dict(title="Total Count", tickformat=",.0f"),
            hovermode="x unified",
        )
        if officer_view_mode == "City-Specific":
            fig_total.update_traces(
                line_width=3,
                hovertemplate='<b>%{customdata[0]}</b><br>Total Officers = %{y:,.0f}<extra></extra>',
            )
        else:
            fig_total.update_traces(
                line_width=3,
                line_color='#2ca02c',
                hovertemplate='%{y:,.0f} Officers<extra></extra>',
            )
        style_plotly_figure(fig_total, theme)
        st.plotly_chart(fig_total, width='stretch')

        st.divider()
        st.markdown("#### 👥 Gender Distribution Breakdown")
        gender_sections = trend_cities if officer_view_mode == "City-Specific" else [None]
        if officer_view_mode == "City-Specific" and len(gender_sections) > 1:
            fig_gender = make_subplots(
                rows=len(gender_sections),
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.15,
            )
            gender_hover_lookup = {}
            for month_start, month_rows in chart_data.groupby('month_start'):
                male_lines = []
                female_lines = []
                for hover_city in gender_sections:
                    city_rows = month_rows[month_rows['city'].astype(str) == hover_city]
                    if city_rows.empty:
                        continue
                    city_row = city_rows.iloc[-1]
                    male_lines.append(f"<b>{hover_city}</b>: {int(city_row['male_officer']):,}")
                    female_lines.append(f"<b>{hover_city}</b>: {int(city_row['female_officer']):,}")
                gender_hover_lookup[pd.Timestamp(month_start)] = {
                    'male': "<br><b>Male Officers</b><br>" + "<br>".join(male_lines),
                    'female': "<b>Female Officers</b><br>" + "<br>".join(female_lines) + "<br>",
                }

            for row_num, city in enumerate(gender_sections, start=1):
                gender_data = chart_data[chart_data['city'].astype(str) == city]
                male_hover_summary = gender_data['month_start'].map(
                    lambda month_start: gender_hover_lookup.get(pd.Timestamp(month_start), {}).get('male', "")
                )
                female_hover_summary = gender_data['month_start'].map(
                    lambda month_start: gender_hover_lookup.get(pd.Timestamp(month_start), {}).get('female', "")
                )
                fig_gender.add_trace(go.Scatter(
                    x=gender_data['month_start'],
                    y=gender_data['male_officer'],
                    mode='lines',
                    name='Male Officers',
                    stackgroup=f'gender_{row_num}',
                    line=dict(width=0, color='#1f77b4'),
                    fillcolor='rgba(31, 119, 180, 0.8)',
                    showlegend=row_num == 1,
                    customdata=male_hover_summary,
                    hovertemplate='%{customdata}<extra></extra>',
                ), row=row_num, col=1)
                fig_gender.add_trace(go.Scatter(
                    x=gender_data['month_start'],
                    y=gender_data['female_officer'],
                    mode='lines',
                    name='Female Officers',
                    stackgroup=f'gender_{row_num}',
                    line=dict(width=0, color='#ff7f0e'),
                    fillcolor='rgba(255, 127, 14, 0.8)',
                    showlegend=row_num == 1,
                    customdata=female_hover_summary,
                    hovertemplate='%{customdata}<extra></extra>',
                ), row=row_num, col=1)
                fig_gender.update_yaxes(title_text="Officer Count", tickformat=",.0f", row=row_num, col=1)
                yaxis_name = 'yaxis' if row_num == 1 else f'yaxis{row_num}'
                y_domain_top = fig_gender.layout[yaxis_name].domain[1]
                fig_gender.add_annotation(
                    text=f"<b>Officer Demographics: {city}</b>",
                    x=0,
                    y=min(y_domain_top + 0.025, 1.04),
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    yanchor="bottom",
                    yshift=8,
                    showarrow=False,
                    font=dict(size=14, color=theme["text"]),
                )

            fig_gender.update_xaxes(
                tickformat="%Y",
                dtick="M12",
                title_text="Date",
                hoverformat="%b %Y",
                tickangle=45,
                showspikes=True,
                showticklabels=True,
            )
            fig_gender.update_yaxes(showspikes=False)
            fig_gender.update_layout(
                title="Officer Demographics by City",
                hovermode="x unified",
                hoversubplots="axis",
                height=max(540, 420 * len(gender_sections)),
                margin=dict(t=152, b=96, l=16, r=16),
            )
            style_plotly_figure(fig_gender, theme)
            fig_gender.update_layout(margin=dict(t=152, b=96, l=16, r=16))
            st.plotly_chart(fig_gender, width='stretch')
        else:
            for city in gender_sections:
                if city is None:
                    gender_data = chart_data
                    chart_title = gender_title
                else:
                    gender_data = chart_data[chart_data['city'].astype(str) == city]
                    chart_title = f"Officer Demographics: {city}"

                fig_gender = go.Figure()
                fig_gender.add_trace(go.Scatter(
                    x=gender_data['month_start'],
                    y=gender_data['male_officer'],
                    mode='lines',
                    name='Male Officers',
                    stackgroup='one',
                    line=dict(width=0, color='#1f77b4'),
                    fillcolor='rgba(31, 119, 180, 0.8)',
                    hovertemplate='<b>Male Officers</b><br>Count=%{y:,.0f}<extra></extra>',
                ))
                fig_gender.add_trace(go.Scatter(
                    x=gender_data['month_start'],
                    y=gender_data['female_officer'],
                    mode='lines',
                    name='Female Officers',
                    stackgroup='one',
                    line=dict(width=0, color='#ff7f0e'),
                    fillcolor='rgba(255, 127, 14, 0.8)',
                    hovertemplate='<b>Female Officers</b><br>Count=%{y:,.0f}<extra></extra>',
                ))
                fig_gender.update_layout(
                    title=chart_title,
                    xaxis=dict(
                        tickformat="%Y",
                        dtick="M12",
                        title="Date",
                        hoverformat="%b %Y",
                        tickangle=45,
                    ),
                    yaxis=dict(title="Officer Count", tickformat=",.0f"),
                    hovermode="x unified",
                )
                style_plotly_figure(fig_gender, theme)
                st.plotly_chart(fig_gender, width='stretch')

        st.divider()
        st.markdown(f"### Current Force Status: {stat_label}")
        if officer_view_mode == "Statewide":
            latest = chart_data.iloc[-1]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Officers ", f"{int(latest['total_officers']):,}")
            with c2:
                st.metric("Male", f"{int(latest['male_officer']):,}")
            with c3:
                st.metric("Female", f"{int(latest['female_officer']):,}")
            st.caption(f"Officer Rate: {latest['officers_per_1000_people']:.2f} per 1k")
            st.caption(f"Data as of: {latest['month_start'].strftime('%B %Y')}")
        else:
            status_cities = [
                city for city in trend_cities
                if not chart_data[chart_data['city'].astype(str) == city].empty
            ]
            missing_status_cities = [
                city for city in trend_cities
                if city not in status_cities
            ]
            if missing_status_cities:
                st.warning(
                    "No current officer status data available for: "
                    + ", ".join(missing_status_cities)
                )
            status_cols = st.columns(len(status_cities))
            for i, city in enumerate(status_cities):
                latest = chart_data[chart_data['city'].astype(str) == city].iloc[-1]
                status_cols[i].metric(f"{city} Officers", f"{int(latest['total_officers']):,}")
                status_cols[i].caption(
                    f"Male: {int(latest['male_officer']):,} | Female: {int(latest['female_officer']):,}"
                )
                status_cols[i].caption(f"Officer Rate: {latest['officers_per_1000_people']:.2f} per 1k")
            latest_date = chart_data['month_start'].max()
            st.caption(f"Data as of: {latest_date.strftime('%B %Y')}")
    else:
        st.warning("No officer data available for this selection.")
