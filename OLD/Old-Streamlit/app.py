import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

## Helpers for model inference and data loading
@st.cache_resource
def load_artifacts():
    # Directory of this script
    base_dir = os.path.dirname(__file__)
    # Artifact paths
    model_path = os.path.join(base_dir, 'best_model.keras')
    le_path = os.path.join(base_dir, 'label_encoder.pkl')
    scaler_path = os.path.join(base_dir, 'scaler.pkl')
    imputer_path = os.path.join(base_dir, 'imputer.pkl')
    ohe_path = os.path.join(base_dir, 'ohe_columns.pkl')
    # Load model and preprocessing artifacts
    model = keras.models.load_model(model_path)
    le = joblib.load(le_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    ohe_columns = joblib.load(ohe_path)
    return model, le, scaler, imputer, ohe_columns

def preprocess_input(raw_input, imputer, scaler, ohe_columns):
    df_raw = pd.DataFrame([raw_input])
    # Temporal transforms: include date components and cyclical time
    df_temp = df_raw[['year','month','day','hour','dayofweek']].copy()
    df_temp['hour_sin'] = np.sin(2*np.pi*df_temp['hour']/24)
    df_temp['hour_cos'] = np.cos(2*np.pi*df_temp['hour']/24)
    df_temp['dayofweek_sin'] = np.sin(2*np.pi*df_temp['dayofweek']/7)
    df_temp['dayofweek_cos'] = np.cos(2*np.pi*df_temp['dayofweek']/7)
    df_temp = df_temp.drop(['hour','dayofweek'], axis=1)
    # Spatial one-hot
    df_spat = pd.get_dummies(df_raw[['city','location_area']])
    df_spat = df_spat.reindex(columns=ohe_columns, fill_value=0)
    # Environmental
    df_env = df_raw[['population','crime_rate_per_1000_people']].values
    # Combine into feature matrix
    X_full = np.hstack([df_spat.values, df_temp.values, df_env])
    # Pad to match imputer's expected number of features
    try:
        n_expected = imputer.n_features_in_
    except AttributeError:
        n_expected = imputer.n_input_features_
    if X_full.shape[1] < n_expected:
        pad = np.zeros((X_full.shape[0], n_expected - X_full.shape[1]))
        X_full = np.hstack([X_full, pad])
    # Apply imputer
    X_imp = imputer.transform(X_full)
    # Split into three inputs
    s_dim = len(ohe_columns)
    t_dim = df_temp.shape[1]
    env_dim = df_env.shape[1]
    X_s = X_imp[:, :s_dim]
    X_t = X_imp[:, s_dim:s_dim + t_dim]
    # Extract only the original environmental features before scaling
    X_e_raw = X_imp[:, s_dim + t_dim : s_dim + t_dim + env_dim]
    X_e = scaler.transform(X_e_raw)
    return [X_s, X_t, X_e]

def predict_crime(raw_input, model, le, scaler, imputer, ohe_columns, top_k=5):
    inputs = preprocess_input(raw_input, imputer, scaler, ohe_columns)
    proba = model.predict(inputs, verbose=0)[0]
    idx = np.argsort(proba)[::-1][:top_k]
    return [(le.inverse_transform([i])[0], float(proba[i])) for i in idx]

@st.cache_data
def load_data():
    try:
        # Load combined data from project root
        return pd.read_csv('combined_data.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def get_unique_values():
    df = load_data()
    if df is not None:
        cities = sorted(df['city'].unique().tolist())
        areas = sorted(df['location_area'].unique().tolist())
        return cities, areas
    return [], []

# Streamlit UI
st.set_page_config(
    page_title="Connecticut Crime Prediction App",
    page_icon="🚨",
    layout="wide"
)

st.title("Connecticut Crime Prediction App")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("🔍 Navigation")
view = st.sidebar.radio("Select View", ["🔮 Predict Crime", "📊 Historical Analysis", "ℹ️ About"])

# Load data and models
with st.spinner("Loading models and data..."):
    data = load_data()
    model, le, scaler, imputer, ohe_columns = load_artifacts()

if data is None or model is None or le is None:
    st.error("❌ Failed to load required data or models. Please check file paths.")
    st.stop()

# Get unique values
cities, areas = get_unique_values()

if view == "🔮 Predict Crime":
    st.header("Crime Prediction")
    st.markdown("Select parameters to predict the most likely crime type for a specific location and time.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Location & Time")
        # Allow selecting multiple cities
        selected_cities = st.multiselect(
            "Select Cities",
            options=cities,
            default=cities[:1] if len(cities) >= 1 else cities,
            help="Choose one or more Connecticut cities for comparison"
        )
        area = st.selectbox("Location Area", areas, help="Select the type of location")
        
        # Time selection
        forecast_horizon = st.selectbox(
            "Forecast Horizon", 
            ["Next Day", "Next Week", "Next Month"],
            help="Choose the prediction timeframe"
        )
        
        # Calculate target datetime based on forecast horizon
        base_date = datetime.now()
        if forecast_horizon == "Next Day":
            target_date = base_date + timedelta(days=1)
        elif forecast_horizon == "Next Week":
            target_date = base_date + timedelta(weeks=1)
        else:  # Next Month
            target_date = base_date + timedelta(days=30)
            
        date = st.date_input("Target Date", target_date)
        
        # Time selection with hourly intervals
        hour = st.selectbox(
            "Target Hour", 
            options=list(range(0, 24)),
            index=target_date.hour,
            format_func=lambda x: f"{x:02d}:00",
            help="Select the hour (0-23)"
        )
        
    with col2:
        st.subheader("🏙️ Environmental Factors")
        
        # Get default values from the first selected city (fallback to first in list)
        fallback_city = selected_cities[0] if selected_cities else cities[0]
        city_data = data[data['city'] == fallback_city]
        default_pop = int(city_data['population'].iloc[0]) if not city_data.empty else 50000
        default_crime_rate = float(city_data['crime_rate_per_1000_people'].iloc[0]) if not city_data.empty else 100.0
        
        population = st.number_input(
            "Population", 
            min_value=1000, 
            max_value=200000, 
            value=default_pop,
            help="City population"
        )
        crime_rate = st.number_input(
            "Crime Rate per 1000 People", 
            min_value=0.0, 
            max_value=500.0, 
            value=default_crime_rate,
            step=0.1,
            help="Historical crime rate per 1000 residents"
        )

    # Trigger prediction and store in session state
    predict_clicked = st.button("🔮 Predict Crime", type="primary", use_container_width=True)
    if predict_clicked:
        # Generate predictions for each city and cache
        dt = datetime.combine(date, datetime.min.time().replace(hour=hour))
        st.session_state['preds'] = {}
        for city in selected_cities:
            raw = {
                'city': city,
                'location_area': area,
                'year': dt.year,
                'month': dt.month,
                'day': dt.day,
                'hour': dt.hour,
                'dayofweek': dt.weekday(),
                'population': population,
                'crime_rate_per_1000_people': crime_rate
            }
            st.session_state['preds'][city] = predict_crime(raw, model, le, scaler, imputer, ohe_columns, top_k=10)
    # Display cached predictions with adjustable top_k
    if 'preds' in st.session_state:
        tabs = st.tabs(selected_cities)
        for city, tab in zip(selected_cities, tabs):
            with tab:
                city_preds = st.session_state['preds'].get(city, [])
                max_k = len(city_preds)
                n_top = st.slider(
                    "Number of top crime types to display",
                    min_value=1,
                    max_value=max_k,
                    value=min(5, max_k),
                    key=f"n_top_{city}"
                )
                preds = city_preds[:n_top]
                results_df = pd.DataFrame(preds, columns=['Crime Type', 'Probability'])
                st.markdown("---")
                st.header(f"🎯 Predictions for {city}")
                for idx, (crime, prob) in enumerate(preds, start=1):
                    st.write(f"**{idx}.** {crime}: {prob:.2%}")
                fig = px.bar(
                    results_df, x='Probability', y='Crime Type', orientation='h',
                    color='Probability', color_continuous_scale='viridis', text='Probability'
                )
                fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                fig.update_layout(
                    xaxis_title='Probability',
                    yaxis={'categoryorder':'total ascending'}, height=400, showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("Show full predictions"):
                    st.dataframe(
                        results_df.assign(Probability=lambda df: df['Probability'].map(lambda x: f"{x:.2%}")),
                        use_container_width=True
                    )

elif view == "📊 Historical Analysis":
    st.header("Historical Crime Analysis")
    st.markdown("Explore historical crime patterns in Connecticut cities.")
    
    # City selection for historical analysis
    selected_cities = st.multiselect("Select Cities", cities, default=cities[:5] if len(cities) >= 5 else cities)
    
    if selected_cities and not data.empty:
        # Filter data
        filtered_data = data[data['city'].isin(selected_cities)]
        
        # Time period selection
        years = sorted(filtered_data['year'].unique())
        if len(years) > 1:
            year_range = st.slider("Select Year Range", min_value=min(years), max_value=max(years), value=(min(years), max(years)))
            filtered_data = filtered_data[(filtered_data['year'] >= year_range[0]) & (filtered_data['year'] <= year_range[1])]
        
        if not filtered_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Crime counts by city
                city_counts = filtered_data.groupby('city').size().reset_index(name='crime_count')
                fig1 = px.bar(city_counts, x='city', y='crime_count', title="Total Crimes by City")
                fig1.update_xaxes(tickangle=45)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Crime types distribution
                crime_counts = filtered_data.groupby('offense_name').size().reset_index(name='count').head(10)
                fig2 = px.pie(crime_counts, values='count', names='offense_name', title="Top 10 Crime Types")
                st.plotly_chart(fig2, use_container_width=True)
            
            # Time series analysis
            monthly_crimes = filtered_data.groupby(['year', 'month']).size().reset_index(name='crime_count')
            monthly_crimes['date'] = pd.to_datetime(monthly_crimes[['year', 'month']].assign(day=1))
            
            fig3 = px.line(monthly_crimes, x='date', y='crime_count', title="Crime Trends Over Time")
            st.plotly_chart(fig3, use_container_width=True)
            
            # Location analysis
            location_counts = filtered_data.groupby('location_area').size().reset_index(name='count').head(15)
            fig4 = px.bar(location_counts, x='count', y='location_area', orientation='h', title="Crimes by Location Type")
            fig4.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig4, use_container_width=True)
    
    else:
        st.info("Please select at least one city to view historical analysis.")

elif view == "ℹ️ About":
    st.header("About This Application")
    st.markdown("""
    ### 🎯 Purpose
    This Crime Prediction App uses machine learning to forecast the most likely crime types in Connecticut cities based on:
    - **Spatial features**: City and location type
    - **Temporal features**: Date, time, and day of the week  
    - **Environmental features**: Population and crime rate
    
    ### 🧠 Model Information
    - **Dense Neural Network**: DNN model trained on Connecticut crime data (2021-2023)
    - **Features**: 7 core features covering spatial, temporal, and environmental factors
    - **Output**: Probability distribution across 20 crime types
    
    ### 📊 Data Sources
    - Connecticut crime incidents from FBI Crime Database for the state of Connecticut
    - Years covered: 2021, 2022, 2023
    - Cities covered: 95 Connecticut municipalities
    
    ### 🚨 Important Disclaimers
    - This tool is for **informational and research purposes only**
    - Predictions are based on historical patterns and may not reflect future events
    - Should **not be used as the sole basis** for law enforcement decisions
    - Crime is influenced by many factors not captured in this model
    
    ### 🔧 Technical Details
    - Built with Streamlit, TensorFlow, and scikit-learn
    - Real-time predictions using pre-trained models
    - Interactive visualizations with Plotly
    """)
    
    # Model performance metrics (if available)
    st.subheader("📈 Model Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Data Size", "366K+ incidents")
    with col2:
        st.metric("Cities Covered", "95")
    with col3:
        st.metric("Crime Types", "20")
