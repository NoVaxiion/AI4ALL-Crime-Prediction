import pandas as pd
import numpy as np
import joblib
import holidays
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# --- 1. SETUP ---
print("⏳ Loading resources...")

# Load Data
df = pd.read_csv("combined_data.csv")
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

# Load Models
forecaster = joblib.load('crime_forecaster.pkl')
classifier = joblib.load('crime_classifier.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define Holidays
ct_holidays = holidays.US(state='CT')

# ---------------------------------------------------------
# PART A: EVALUATE FORECASTER (Recursive Backtest)
# ---------------------------------------------------------
print("\n" + "="*50)
print("📉 EVALUATING FORECASTER (Volume Prediction)")
print("="*50)

# 1. Prepare Data (Aggregate & Features)
daily_crimes = df.groupby(['date', 'city']).size().reset_index(name='crime_count')
stats_cols = ['population', 'total_officers', 'officers_per_1000_people', 'crime_rate_per_1000_people']
city_stats = df[['city', 'year'] + stats_cols].drop_duplicates(subset=['city', 'year'])

# Reindex to fill missing days
all_dates = pd.date_range(start=daily_crimes['date'].min(), end=daily_crimes['date'].max(), freq='D')
idx = pd.MultiIndex.from_product([all_dates, daily_crimes['city'].unique()], names=['date', 'city'])
daily = daily_crimes.set_index(['date', 'city']).reindex(idx, fill_value=0).reset_index()
daily['year'] = daily['date'].dt.year

# Merge Stats
daily = daily.merge(city_stats, on=['city', 'year'], how='left')
daily[stats_cols] = daily.groupby('city')[stats_cols].ffill().bfill()

# 2. Define Test Period (Last 30 Days)
cutoff_date = daily['date'].max() - timedelta(days=30)
test_df = daily[daily['date'] > cutoff_date].copy()
history_df = daily[daily['date'] <= cutoff_date].copy()

print(f"Test Period: {test_df['date'].min().date()} to {test_df['date'].max().date()}")
print(f"Simulating 30-day forecast for {len(test_df['city'].unique())} cities...")

# 3. Recursive Loop
actuals = []
predictions = []

# Loop through every city to generate independent forecasts
for city in test_df['city'].unique():
    city_history = history_df[history_df['city'] == city].sort_values('date')['crime_count'].tolist()
    city_test = test_df[test_df['city'] == city].sort_values('date')
    
    # Get static stats for this city
    static_stats = city_test.iloc[0][stats_cols]
    
    # Buffer for recursion
    buffer = city_history[-30:] # Start with last 30 days of known history
    
    current_date = cutoff_date
    
    for idx, row in city_test.iterrows():
        current_date += timedelta(days=1)
        
        # Calculate dynamic features from BUFFER (Predictions)
        lag1 = buffer[-1]
        lag7 = buffer[-7]
        roll7 = np.mean(buffer[-7:])
        is_hol = 1 if current_date in ct_holidays else 0
        
        # Build Input
        X_in = pd.DataFrame([{
            'lag_1': lag1, 'lag_7': lag7, 'roll_mean_7': roll7,
            'day_of_week': current_date.weekday(), 'month': current_date.month,
            'city': city, 'is_holiday': is_hol,
            'population': static_stats['population'],
            'total_officers': static_stats['total_officers'],
            'officers_per_1000_people': static_stats['officers_per_1000_people'],
            'crime_rate_per_1000_people': static_stats['crime_rate_per_1000_people']
        }])
        
        # Ensure categories
        X_in['city'] = X_in['city'].astype('category')
        
        # Predict
        pred = max(0, forecaster.predict(X_in)[0])
        
        # Store results
        actuals.append(row['crime_count'])
        predictions.append(pred)
        
        # Update buffer with PREDICTION (not actual) to be honest
        buffer.append(pred)

# 4. Metrics
mae = mean_absolute_error(actuals, predictions)
total_actual = sum(actuals)
total_pred = sum(predictions)
vol_acc = 100 - (abs(total_actual - total_pred) / total_actual * 100)

print("\n" + "-"*30)
print(f"✅ FORECASTER RESULTS")
print("-"*30)
print(f"Mean Absolute Error (MAE): {mae:.2f} (Avg error per city/day)")
print(f"Total Predicted Volume:    {int(total_pred)}")
print(f"Total Actual Volume:       {int(total_actual)}")
print(f"Global Volume Accuracy:    {vol_acc:.2f}%")


# ---------------------------------------------------------
# PART B: EVALUATE CLASSIFIER (Risk Type)
# ---------------------------------------------------------
print("\n" + "="*50)
print("🔍 EVALUATING CLASSIFIER (Crime Type Prediction)")
print("="*50)

# 1. Prepare Test Data (Last 30 Days of raw incidents)
# We use the raw dataframe because classification is per-incident, not aggregated
df_test_clf = df[df['date'] > cutoff_date].copy()

print(f"Evaluating on {len(df_test_clf)} incidents from the last 30 days...")

# 2. Prepare Inputs
X_test = df_test_clf[['city', 'location_area', 'hour', 'dayofweek', 'month']].copy()
y_test = df_test_clf['offense_category_name']

# Ensure Categories match training
for col in X_test.columns:
    X_test[col] = X_test[col].astype('category')

# Encode Target
y_test_encoded = label_encoder.transform(y_test)

# 3. Predict
y_pred = classifier.predict(X_test)

# 4. Metrics
acc = accuracy_score(y_test_encoded, y_pred)
print("\n" + "-"*30)
print(f"✅ CLASSIFIER RESULTS")
print("-"*30)
print(f"Accuracy: {acc:.2%}")
print("\nTop Crime Type Performance:")

# Get readable report
report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_, output_dict=True)
# Convert to dataframe for cleaner view
report_df = pd.DataFrame(report).transpose().sort_values('support', ascending=False)
print(report_df.head(10)[['precision', 'recall', 'f1-score', 'support']])