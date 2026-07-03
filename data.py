from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'Models'
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


@st.cache_data
def load_data(data_path=MODELS_DIR / 'combined_data.csv'):
    df = pd.read_csv(data_path, usecols=APP_DATA_COLUMNS)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    test_period = df[df['date'] > (df['date'].max() - pd.Timedelta(days=90))]
    test_counts = test_period['offense_category_name'].value_counts()
    rare = test_counts[test_counts < 100].index.tolist()
    df['offense_category_clean'] = df['offense_category_name'].apply(
        lambda x: 'Other' if x in rare else x
    )
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


@st.cache_data
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


@st.cache_data
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
        city_crimes = city_crimes[city_crimes['date'].dt.year == int(year_filter)]
    if city_crimes.empty:
        return None

    dist = city_crimes['offense_category_name'].value_counts().reset_index()
    dist.columns = ['Crime Type', 'Count']
    return dist[dist['Count'] > 0].head(8)


@st.cache_data
def build_lookup_tables(_df):
    stat_cols = ['population', 'total_officers', 'officers_per_1000_people', 'crime_rate_per_1000_people']
    city_stats = _df.groupby('city', observed=False)[stat_cols].last().reset_index()
    city_stats['city'] = city_stats['city'].astype(str)
    city_stats_lookup = city_stats.set_index('city')[stat_cols].to_dict('index')

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

    city_cats = sorted(_df['city'].astype(str).unique().tolist())
    loc_cats = sorted(_df['location_area'].astype(str).unique().tolist())
    loc_type_cats = ['commercial', 'education', 'nightlife', 'other', 'residential', 'retail', 'street']
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


@st.cache_data
def build_forecast_profiles(_ts_data):
    profiles = {}
    for city, city_df in _ts_data.groupby('city', observed=False):
        profile = city_df.sort_values('date').copy()
        profile['month'] = profile['date'].dt.month
        profile['day_of_week'] = profile['date'].dt.dayofweek
        profile['day_of_year'] = profile['date'].dt.dayofyear
        profiles[str(city)] = profile
    return profiles
