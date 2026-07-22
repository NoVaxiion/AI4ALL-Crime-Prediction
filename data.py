import json
import os
import pickle
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'Models'
DEFAULT_HF_REPO_ID = 'NoVaxiion/project360-assets'
DEFAULT_HF_REPO_TYPE = 'dataset'
PER_CITY_INDEX_PATH = 'per_city/index.json'
LOCATION_TYPE_CATEGORIES = [
    'commercial',
    'education',
    'nightlife',
    'other',
    'residential',
    'retail',
    'street',
]
APP_DATA_COLUMNS = [
    'year',
    'month',
    'day',
    'hour',
    'city',
    'location_area',
    'offense_category_name',
    'population',
    'total_officers',
    'male_officer',
    'female_officer',
    'officers_per_1000_people',
    'crime_rate_per_1000_people',
]


def get_secret_or_env(name, default=None):
    """Read deployment configuration from Streamlit secrets or the environment."""
    try:
        value = st.secrets.get(name)
        if value is not None:
            return value
    except (AttributeError, FileNotFoundError, KeyError):
        return os.getenv(name, default)
    return os.getenv(name, default)


def is_lfs_pointer(path):
    try:
        with open(path, 'rb') as file:
            return file.read(80).startswith(b'version https://git-lfs.github.com/spec')
    except FileNotFoundError:
        raise
    except OSError:
        return False


def resolve_asset_path(filename, required=True):
    """Return an app asset path, preferring Hugging Face over Git/LFS."""
    repo_id = get_secret_or_env('HF_REPO_ID', DEFAULT_HF_REPO_ID)
    repo_type = get_secret_or_env('HF_REPO_TYPE', DEFAULT_HF_REPO_TYPE)
    token = get_secret_or_env('HF_TOKEN')
    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                token=token,
            )
        )
    except Exception as exc:
        local_path = MODELS_DIR / filename
        if local_path.exists() and not is_lfs_pointer(local_path):
            return local_path
        if required:
            st.error(
                f"Could not load required asset `{filename}` from Hugging Face or local `Models/`. "
                "If the Hugging Face repo is private, add `HF_TOKEN`, `HF_REPO_ID`, and "
                "`HF_REPO_TYPE` to Streamlit secrets."
            )
            st.exception(exc)
            st.stop()
        return None


@st.cache_resource
def load_per_city_index():
    """Load the optional index for independently downloadable city models."""
    path = resolve_asset_path(PER_CITY_INDEX_PATH, required=False)
    if path is None or is_lfs_pointer(path):
        return None
    try:
        index = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(index, dict):
        return None
    if not isinstance(index.get('forecasters', {}), dict):
        return None
    if not isinstance(index.get('classifiers', {}), dict):
        return None
    return index


@st.cache_resource(max_entries=8)
def load_split_city_model(asset_group, city):
    """Load one indexed city model without materializing the full model dictionary."""
    if asset_group not in {'forecasters', 'classifiers'}:
        raise ValueError(f'Unsupported per-city asset group: {asset_group}')
    index = load_per_city_index()
    if index is None:
        return None
    filename = index.get(asset_group, {}).get(str(city))
    if not filename or not str(filename).startswith('per_city/'):
        return None
    path = resolve_asset_path(str(filename), required=False)
    if path is None or is_lfs_pointer(path):
        return None
    try:
        return joblib.load(path)
    except (
        OSError,
        KeyError,
        ValueError,
        EOFError,
        TypeError,
        AttributeError,
        ImportError,
        pickle.UnpicklingError,
    ):
        return None


@st.cache_resource
def load_data(data_path=None):
    if data_path is None:
        data_path = resolve_asset_path('combined_data.csv')
    df = pd.read_csv(data_path, usecols=APP_DATA_COLUMNS)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    # Runtime data is descriptive. Rare-label decisions belong to the saved
    # training artifact and must never be inferred from an evaluation period.
    df['offense_category_clean'] = df['offense_category_name'].astype('string')
    df['year'] = df['year'].astype('int16')
    df['month'] = df['month'].astype('int8')
    df['day'] = df['day'].astype('int8')
    df['hour'] = df['hour'].fillna(0).astype('int8')
    for col in ['population', 'total_officers', 'male_officer', 'female_officer']:
        df[col] = df[col].fillna(0).astype('int32')
    for col in ['officers_per_1000_people', 'crime_rate_per_1000_people']:
        df[col] = df[col].astype('float32')
    for col in ['city', 'location_area', 'offense_category_name']:
        df[col] = df[col].astype('category')
    return df


@st.cache_resource
def load_app_data_bundle():
    """Load the compact v2 dashboard data bundle when it is available."""
    path = resolve_asset_path('app_data_bundle.pkl', required=False)
    if path is None or is_lfs_pointer(path):
        return None
    try:
        bundle = joblib.load(path)
    except (
        OSError,
        KeyError,
        ValueError,
        EOFError,
        TypeError,
        AttributeError,
        ImportError,
        pickle.UnpicklingError,
    ):
        return None
    required = {
        'schema_version',
        'model_version',
        'daily_city',
        'officer_trends',
        'crime_distribution',
        'location_areas',
        'years',
    }
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        return None
    if not str(bundle['schema_version']).startswith('2'):
        return None
    return bundle


@st.cache_resource
def get_aggregate_data(_df):
    daily = _df.groupby(['date', 'city'], observed=False).size().reset_index(name='crime_count')
    stats_cols = [
        'population',
        'total_officers',
        'male_officer',
        'female_officer',
        'officers_per_1000_people',
        'crime_rate_per_1000_people',
    ]
    city_stats = _df[['year', 'city'] + stats_cols].drop_duplicates(subset=['year', 'city'])
    city_stats = city_stats.groupby(['year', 'city'], observed=False).max().reset_index()

    idx = pd.MultiIndex.from_product(
        [pd.date_range(daily['date'].min(), daily['date'].max(), freq='D'), daily['city'].unique()],
        names=['date', 'city'],
    )
    daily = daily.set_index(['date', 'city']).reindex(idx, fill_value=0).reset_index()
    daily['year'] = daily['date'].dt.year

    daily = daily.merge(city_stats, on=['city', 'year'], how='left')
    daily[stats_cols] = daily.groupby('city', observed=False)[stats_cols].ffill().bfill()
    daily['city'] = daily['city'].astype('category')
    return daily.dropna()


@st.cache_resource
def get_officer_trends(_df):
    df = _df.copy()
    df['month_start'] = df['date'].dt.to_period('M').dt.to_timestamp()
    return df[
        [
            'month_start',
            'city',
            'population',
            'total_officers',
            'male_officer',
            'female_officer',
            'officers_per_1000_people',
        ]
    ].drop_duplicates()


@st.cache_data
def get_crime_distribution(_df, city, year_filter):
    """Get historical crime type distribution for the pie chart, optionally filtered by year."""
    city_crimes = _df[_df['city'] == city]
    if year_filter != 'All Years':
        if 'year' in city_crimes:
            city_crimes = city_crimes[city_crimes['year'] == int(year_filter)]
        else:
            city_crimes = city_crimes[city_crimes['date'].dt.year == int(year_filter)]
    if city_crimes.empty:
        return None

    if 'count' in city_crimes:
        dist = (
            city_crimes.groupby('offense_category_name', observed=True)['count']
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        dist.columns = ['Crime Type', 'Count']
    else:
        dist = city_crimes['offense_category_name'].value_counts().reset_index()
        dist.columns = ['Crime Type', 'Count']
    return dist[dist['Count'] > 0].head(8)


@st.cache_resource
def build_bundle_lookup_tables(_bundle):
    """Build the small UI lookup contract without loading incident-level rows."""
    daily = _bundle['daily_city']
    stat_cols = ['population', 'total_officers', 'officers_per_1000_people', 'crime_rate_per_1000_people']
    latest = daily.sort_values('date').groupby('city', observed=True)[stat_cols].last()
    legacy = _bundle.get('legacy_lookups', {})
    return {
        'city_stats_lookup': latest.to_dict('index'),
        'loc_total_lookup': legacy.get('loc_total_lookup', {}),
        'hour_typical_lookup': legacy.get('hour_typical_lookup', {}),
        'avg_div_lookup': legacy.get('avg_div_lookup', {}),
        'avg_loc_lookup': legacy.get('avg_loc_lookup', {}),
        'city_cats': sorted(daily['city'].astype(str).unique().tolist()),
        'loc_cats': sorted(str(value) for value in _bundle['location_areas']),
        'loc_type_cats': LOCATION_TYPE_CATEGORIES,
        'htc_cats': legacy.get('htc_cats', []),
    }


@st.cache_resource
def build_lookup_tables(_df, include_legacy=True):
    stat_cols = ['population', 'total_officers', 'officers_per_1000_people', 'crime_rate_per_1000_people']
    city_stats = _df.groupby('city', observed=False)[stat_cols].last().reset_index()
    city_stats['city'] = city_stats['city'].astype(str)
    city_stats_lookup = city_stats.set_index('city')[stat_cols].to_dict('index')

    city_cats = sorted(_df['city'].astype(str).unique().tolist())
    loc_cats = sorted(_df['location_area'].astype(str).unique().tolist())
    loc_type_cats = LOCATION_TYPE_CATEGORIES
    loc_total_lookup = {}
    hour_typical_lookup = {}
    avg_div_lookup = {}
    avg_loc_lookup = {}
    htc_cats = []

    # Version 1 models require these target-derived tables. Version 2 models use
    # frozen training-only artifacts and skip this expensive legacy work.
    if include_legacy:
        loc_totals = _df.groupby('location_area', observed=False).size().reset_index(name='location_total_crimes')
        loc_totals['location_area'] = loc_totals['location_area'].astype(str)
        loc_total_lookup = loc_totals.set_index('location_area')['location_total_crimes'].to_dict()
        hour_typical = (
            _df.groupby(['city', 'hour'], observed=False)['offense_category_clean']
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Other')
            .reset_index(name='hour_typical_crime')
        )
        hour_typical['city'] = hour_typical['city'].astype(str)
        hour_typical['hour'] = hour_typical['hour'].astype(int)
        hour_typical_lookup = {
            (row.city, int(row.hour)): row.hour_typical_crime
            for row in hour_typical.itertuples(index=False)
        }
        div_raw = (
            _df.groupby(['date', 'hour', 'city'], observed=False)['offense_category_clean']
            .nunique()
            .reset_index(name='crime_diversity')
        )
        avg_div = div_raw.groupby(['city', 'hour'], observed=False)['crime_diversity'].mean().reset_index(name='avg_crime_diversity')
        avg_div['city'] = avg_div['city'].astype(str)
        avg_div['hour'] = avg_div['hour'].astype(int)
        avg_div_lookup = {
            (row.city, int(row.hour)): row.avg_crime_diversity
            for row in avg_div.itertuples(index=False)
        }
        loc_day = _df.groupby(['date', 'location_area'], observed=False).size().reset_index(name='loc_freq')
        avg_loc = loc_day.groupby('location_area', observed=False)['loc_freq'].mean().reset_index(name='avg_location_daily_freq')
        avg_loc['location_area'] = avg_loc['location_area'].astype(str)
        avg_loc_lookup = avg_loc.set_index('location_area')['avg_location_daily_freq'].to_dict()
        htc_cats = sorted(hour_typical['hour_typical_crime'].dropna().astype(str).unique().tolist())

    return {
        'city_stats_lookup': city_stats_lookup,
        'loc_total_lookup': loc_total_lookup,
        'hour_typical_lookup': hour_typical_lookup,
        'avg_div_lookup': avg_div_lookup,
        'avg_loc_lookup': avg_loc_lookup,
        'city_cats': city_cats,
        'loc_cats': loc_cats,
        'loc_type_cats': loc_type_cats,
        'htc_cats': htc_cats,
    }


@st.cache_resource
def build_forecast_profiles(_ts_data):
    profiles = {}
    for city, city_df in _ts_data.groupby('city', observed=False):
        profile = city_df.sort_values('date').copy()
        profile['month'] = profile['date'].dt.month
        profile['day_of_week'] = profile['date'].dt.dayofweek
        profile['day_of_year'] = profile['date'].dt.dayofyear
        profiles[str(city)] = profile
    return profiles
