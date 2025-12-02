import os
# 1. FORCE CPU MODE
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.express as px
import plotly.graph_objects as go
import holidays
import joblib
from datetime import datetime, timedelta

# --- APP CONFIGURATION ---
st.set_page_config(page_title="CT Crime Insight 360", page_icon="🚓", layout="wide")
st.title("🚓 ProjeCT 360")
st.markdown("### Crime, Population & Officer Awareness")

# --- 1. LOAD MODELS ---
@st.cache_resource
def load_models():
    try:
        forecaster = joblib.load('Models/crime_forecaster.pkl')
        classifier = joblib.load('Models/crime_classifier.pkl')
        label_encoder = joblib.load('Models/label_encoder.pkl')
        return forecaster, classifier, label_encoder
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}. Please run 'train_model.py' first.")
        st.stop()

# --- 2. DATA ENGINE ---
@st.cache_data
def load_data():
    df = pd.read_csv("Models/combined_data.csv")
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    for col in ['city', 'location_area', 'offense_category_name']:
        df[col] = df[col].astype('category')
    return df

@st.cache_data
def get_aggregate_data(df):
    daily = df.groupby(['date', 'city']).size().reset_index(name='crime_count')
    stats_cols = ['population', 'total_officers', 'male_officer', 'female_officer', 'officers_per_1000_people', 'crime_rate_per_1000_people']
    city_stats = df[['year', 'city'] + stats_cols].drop_duplicates(subset=['year', 'city'])
    city_stats = city_stats.groupby(['year', 'city']).max().reset_index()

    idx = pd.MultiIndex.from_product(
        [pd.date_range(daily['date'].min(), daily['date'].max(), freq='D'), daily['city'].unique()],
        names=['date', 'city']
    )
    daily = daily.set_index(['date', 'city']).reindex(idx, fill_value=0).reset_index()
    daily['year'] = daily['date'].dt.year

    daily = daily.merge(city_stats, on=['city', 'year'], how='left')
    daily[stats_cols] = daily.groupby('city')[stats_cols].ffill().bfill()
    daily['city'] = daily['city'].astype('category')
    
    return daily.dropna()

@st.cache_data
def get_officer_trends(df):
    df['month_start'] = df['date'].dt.to_period('M').dt.to_timestamp()
    unique_counts = df[['month_start', 'city', 'total_officers', 'male_officer', 'female_officer']].drop_duplicates()
    return unique_counts

@st.cache_data
def get_crime_distribution(df, city, year_filter):
    """Get historical crime type distribution for the pie chart, optionally filtered by year."""
    # 1. Filter by City
    city_crimes = df[df['city'] == city]
    
    # 2. Filter by Year (if not 'All Years')
    if year_filter != "All Years":
        city_crimes = city_crimes[city_crimes['date'].dt.year == int(year_filter)]

    if city_crimes.empty: return None
    
    # 3. Calculate Counts
    dist = city_crimes['offense_category_name'].value_counts().reset_index()
    dist.columns = ['Crime Type', 'Count']
    
    # Filter out zero counts and take top 8
    dist = dist[dist['Count'] > 0].head(8) 
    return dist

# --- INITIALIZATION ---
with st.spinner("Booting up system..."):
    forecaster, classifier, label_encoder = load_models()
    raw_df = load_data()
    ts_data = get_aggregate_data(raw_df)
    officer_raw_data = get_officer_trends(raw_df)
    city_categories = ts_data['city'].cat.categories

st.success("✅ Models Loaded")

# --- 3. FORECAST LOGIC ---
def run_forecast_loop(city, steps=30):
    city_slice = ts_data[ts_data['city'] == city].sort_values('date')
    if city_slice.empty: return None
    
    latest_stats = city_slice.iloc[-1][['population', 'total_officers', 'officers_per_1000_people', 'crime_rate_per_1000_people']]
    history = city_slice.set_index('date')['crime_count']
    
    last_data_date = history.index.max()
    target_start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    days_to_bridge = max(0, (target_start_date - last_data_date).days)
    total_loops = days_to_bridge + steps
    
    buffer = history.tail(30).tolist()
    current_date = last_data_date
    ct_holidays = holidays.US(state='CT')
    
    future_dates = []
    future_values = []
    is_holiday_list = []
    
    for _ in range(total_loops):
        current_date += timedelta(days=1)
        lag1, lag7, roll7 = buffer[-1], buffer[-7], np.mean(buffer[-7:])
        is_hol = 1 if current_date in ct_holidays else 0
        
        X_in = pd.DataFrame([{
            'lag_1': lag1, 'lag_7': lag7, 'roll_mean_7': roll7,
            'day_of_week': current_date.weekday(), 'month': current_date.month,
            'city': city, 'is_holiday': is_hol,
            'population': latest_stats['population'],
            'total_officers': latest_stats['total_officers'],
            'officers_per_1000_people': latest_stats['officers_per_1000_people'],
            'crime_rate_per_1000_people': latest_stats['crime_rate_per_1000_people']
        }])
        
        X_in['city'] = pd.Categorical(X_in['city'], categories=city_categories)
        pred = max(0, forecaster.predict(X_in)[0])
        
        if current_date >= target_start_date:
            future_dates.append(current_date)
            future_values.append(pred)
            is_holiday_list.append(is_hol)
            
        buffer.append(pred)
        
    return pd.DataFrame({'Date': future_dates, 'Predicted Count': future_values, 'is_holiday': is_holiday_list})

# --- 4. DASHBOARD UI ---
cities = sorted(raw_df['city'].unique())
locations = sorted(raw_df['location_area'].unique())
available_years = sorted(raw_df['year'].unique(), reverse=True)

st.sidebar.header("🌍 Settings")
selected_city = st.sidebar.selectbox("Select City", cities, key="sb_city_select")

stats_row = ts_data[ts_data['city'] == selected_city].iloc[-1]
st.sidebar.markdown("---")
st.sidebar.markdown(f"**📍 {selected_city} Stats**")
st.sidebar.metric("👮 Officers", int(stats_row['total_officers']))
st.sidebar.metric("👥 Population", f"{int(stats_row['population']):,}")
st.sidebar.caption(f"Officer Rate: {stats_row['officers_per_1000_people']} Officer(s) per 1,000 People")

# --- ABOUT SECTION ---
with st.sidebar.expander("ℹ️ About PrediCT 360"):
    st.markdown("""
    **CT Crime PrediCT 360** is a predictive analytics platform for informational purposes and attempting to predict future crimes in Connecticut.
    
    #### 🚀 Core Capabilities & Performance
    
    **1. Volume Forecasting (Regression)**
    * **Function:** Predicts daily incident counts for the next 30 days.
    * **Accuracy:** **~76.7%** (Global Volume).
    
    **2. Risk Classification (Probability)**
    * **Function:** Calculates likelihood of specific crime types.
    * **Accuracy:** **~48.2%** (Top-1 prediction).
    
    **3. Resource Analytics**
    * **Function:** Monitors police staffing levels and gender distribution trends over time.
    
    ---
    *Built for data-driven decision making.*
    
    *Note: This is my first ML project! Please excuse any mistakes or inaccuracies as I continue to learn and refine the models. Thanks for stopping by!*
    """)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📈 Volume Forecast (In Beta)", "🔍 Risk Analysis", "👮 Officer Trends"])

# TAB 1: VOLUME (Automatic Update)
with tab1:
    st.subheader(f"30-Day Volume Forecast: {selected_city}")
    
    # AUTOMATIC EXECUTION (No Button)
    with st.spinner("Calculating forecast..."):
        res = run_forecast_loop(selected_city)
        
    if res is not None:
        fig = px.line(res, x='Date', y='Predicted Count', markers=True)
        fig.update_layout(
            title="Predicted Daily Incidents (30-Day Trend)",
            xaxis_title="Date",
            yaxis_title="Incident Count"
        )
        fig.update_traces(line_color='#FF4B4B', line_width=3)
        
        hol_days = res[res['is_holiday'] == 1]
        if not hol_days.empty:
             fig.add_trace(go.Scatter(
                x=hol_days['Date'], y=hol_days['Predicted Count'],
                mode='markers', name='Holiday',
                marker=dict(color='gold', size=15, symbol='star', line=dict(color='black', width=1))
            ))
        
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Total Predicted (30 Days)", f"{int(res['Predicted Count'].sum())} Crimes")
    else:
        st.error("Insufficient data.")

# TAB 2: RISK TYPE (Updated Layout - Stacked)
with tab2:
    st.subheader("Crime Type Probability & Analysis")
    
    # 1. Inputs Section
    c1, c2 = st.columns(2)
    with c1:
        s_date = st.date_input("Date", datetime.now(), key="risk_date_input")
        s_time = st.slider("Hour", 0, 23, 12, key="risk_time_slider")
        s_loc = st.selectbox("Location Type", locations, key="risk_loc_select")
        
        # Prediction Logic
        in_data = pd.DataFrame([{
            'city': selected_city, 'location_area': s_loc, 
            'hour': s_time, 'dayofweek': s_date.weekday(), 'month': s_date.month
        }])
        for c in in_data.columns: in_data[c] = in_data[c].astype('category')
        
        probs = classifier.predict_proba(in_data)[0]
        top_idx = probs.argsort()[-5:][::-1]
        chart_df = pd.DataFrame({'Crime Type': label_encoder.inverse_transform(top_idx), 'Probability': probs[top_idx]})

    with c2:
        # Scenario Info Box
        st.info(f"""
        **Analysis Context:**
        - **City:** {selected_city}
        - **Location:** {s_loc}
        - **Time:** {s_date.strftime('%Y-%m-%d')} @ {s_time}:00
        """)

    st.divider()

    # 2. Current Probability Bar Chart (Full Width)
    st.markdown(f"#### 🚨 Top 5 Probable Offenses (Scenario Prediction)")
    fig_bar = px.bar(chart_df, x='Probability', y='Crime Type', orientation='h', 
                 color='Probability', color_continuous_scale='Reds', text_auto='.1%')
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.divider()

    # 3. Historical Pie Chart (Full Width, Below Bar Chart)
    # Filters for Pie Chart
    c_pie_title, c_pie_select = st.columns([3, 1])
    with c_pie_title:
        st.markdown(f"#### 🥧 Historical Crime Distribution ({selected_city})")
    with c_pie_select:
        year_options = ["All Years"] + available_years
        selected_year = st.selectbox("Filter Year", year_options, key="pie_year_select")

    # Get Data
    dist_data = get_crime_distribution(raw_df, selected_city, selected_year)
    
    if dist_data is not None and not dist_data.empty:
        fig_pie = px.pie(dist_data, values='Count', names='Crime Type', 
                         hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=True, height=500)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info(f"No historical distribution data available for {selected_city} in {selected_year}.")

# TAB 3: OFFICER TRENDS
with tab3:
    st.subheader("Police Force Analysis")
    
    view_mode = st.radio("View Mode:", ["Statewide", "City-Specific"], horizontal=True, key="trend_view_mode")
    
    if view_mode == "Statewide":
        chart_data = officer_raw_data.groupby('month_start')[['total_officers', 'male_officer', 'female_officer']].sum().reset_index()
        title_text = "Statewide Police Force Trend"
        stat_label = "Statewide"
    else:
        chart_data = officer_raw_data[officer_raw_data['city'] == selected_city].sort_values('month_start')
        title_text = f"Police Force Trend: {selected_city}"
        stat_label = selected_city

    if not chart_data.empty:
        st.markdown("#### 📈 Total Force Size")
        fig_total = px.line(chart_data, x='month_start', y='total_officers', markers=True)
        fig_total.update_layout(
            title=title_text,
            xaxis=dict(tickformat="%b %Y", dtick="M2", title="Date"),
            yaxis_title="Total Count",
            hovermode="x unified"
        )
        fig_total.update_traces(line_color='#2ca02c', line_width=3)
        st.plotly_chart(fig_total, use_container_width=True)

        st.divider()

        st.markdown("#### 👥 Gender Distribution Breakdown")
        fig_gender = go.Figure()
        
        fig_gender.add_trace(go.Scatter(
            x=chart_data['month_start'], y=chart_data['male_officer'],
            mode='lines', name='Male Officers', stackgroup='one',
            line=dict(width=0, color='#1f77b4'), fillcolor='rgba(31, 119, 180, 0.8)'
        ))
        
        fig_gender.add_trace(go.Scatter(
            x=chart_data['month_start'], y=chart_data['female_officer'],
            mode='lines', name='Female Officers', stackgroup='one',
            line=dict(width=0, color='#ff7f0e'), fillcolor='rgba(255, 127, 14, 0.8)'
        ))

        fig_gender.update_layout(
            title=f"Officer Demographics: {selected_city}",
            xaxis=dict(tickformat="%b %Y", dtick="M2", title="Date"),
            yaxis_title="Officer Count",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_gender, use_container_width=True)
        
        st.divider()
        st.markdown(f"### Current Force Status: {stat_label}")
        
        latest = chart_data.iloc[-1]
        c1, c2, c3 = st.columns(3)
        with c1: st.metric(label="Total Officers", value=int(latest['total_officers']))
        with c2: st.metric(label="Male", value=int(latest['male_officer']))
        with c3: st.metric(label="Female", value=int(latest['female_officer']))
            
        st.caption(f"Data as of: {latest['month_start'].strftime('%B %Y')}")
    else:
        st.warning("No officer data available for this selection.")
