"""
Monthly Electricity Demand Forecasting
========================================
This script demonstrates how to forecast next month's electricity demand
using the previous month's data.

The model analyzes patterns from the entire previous month (last 30 days of hourly data)
and predicts the average demand for the next month, including expected range and growth.

Usage Example:
    python forecast_monthly.py
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from electricity_demand_model import predict_monthly_demand, DATA_PATH

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: LOAD PREVIOUS MONTH DATA FROM CSV
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 80)
print("  MONTHLY ELECTRICITY DEMAND FORECASTING")
print("  Using Previous Month Data to Forecast Next Month")
print("=" * 80)

# Try to load previous month data from CSV
try:
    previous_month_data = pd.read_csv('data.csv')
    print(f"\n✓ Loaded previous month data from data.csv")
    print(f"  Shape: {previous_month_data.shape}")
    print(f"  Records: {len(previous_month_data)}")
except FileNotFoundError:
    print(f"\n✗ data.csv not found. Generating example previous month data...")
    
    # METHOD 2: CREATE SYNTHETIC PREVIOUS MONTH DATA
    # This creates realistic hourly data for a complete month (720 hours = 30 days)
    
    # Define the month range
    month_start = datetime(2024, 3, 1)  # March 1, 2024
    month_end = datetime(2024, 3, 31, 23)  # March 31, 2024 11 PM
    
    # Create hourly timestamps for entire month
    hours_list = []
    current = month_start
    while current <= month_end:
        hours_list.append(current)
        current += timedelta(hours=1)
    
    # Generate realistic demand patterns
    np.random.seed(42)
    n_hours = len(hours_list)
    
    # Base load + hourly variation + random noise
    base_load = 7800  # MW base demand
    hourly_pattern = 1200 * np.sin(2 * np.pi * np.array([h.hour for h in hours_list]) / 24)
    weekday_bonus = 400 * np.where(np.array([h.weekday() for h in hours_list]) < 5, 1, 0)
    noise = np.random.normal(0, 250, n_hours)
    
    demand = base_load + hourly_pattern + weekday_bonus + noise
    demand = np.maximum(demand, 5000)  # Ensure minimum demand
    
    # Temperature pattern (seasonal + daily variation)
    temp_base = 10  # March average
    seasonal_temp = 8 * np.sin(2 * np.pi * np.array([h.timetuple().tm_yday for h in hours_list]) / 365)
    daily_temp = 6 * np.cos(2 * np.pi * np.array([h.hour for h in hours_list]) / 24)
    temperature = temp_base + seasonal_temp + daily_temp + np.random.normal(0, 1, n_hours)
    
    # Humidity pattern
    humidity_base = 65
    humidity = humidity_base + 15 * np.sin(2 * np.pi * np.array([h.hour for h in hours_list]) / 24) + \
               np.random.normal(0, 3, n_hours)
    humidity = np.clip(humidity, 20, 100)
    
    # Create DataFrame
    previous_month_data = pd.DataFrame({
        'year': [h.year for h in hours_list],
        'month': [h.month for h in hours_list],
        'dayofmonth': [h.day for h in hours_list],
        'dayofweek': [h.weekday() for h in hours_list],
        'dayofyear': [h.timetuple().tm_yday for h in hours_list],
        'hour': [h.hour for h in hours_list],
        'Demand': demand,
        'Temperature': temperature,
        'Humidity': humidity,
    })
    
    print(f"\n✓ Generated synthetic previous month data")
    print(f"  Period: {month_start.date()} to {month_end.date()}")
    print(f"  Records: {len(previous_month_data)} hourly records (1 month = 720-744 hours)")

# Display previous month statistics
print(f"\n[Previous Month Analysis]")
print(f"  Average Daily Demand:    {previous_month_data['Demand'].mean():,.1f} MW")
print(f"  Peak Demand:             {previous_month_data['Demand'].max():,.1f} MW")
print(f"  Minimum Demand:          {previous_month_data['Demand'].min():,.1f} MW")
print(f"  Standard Deviation:      {previous_month_data['Demand'].std():,.1f} MW")
print(f"  Avg Temperature:         {previous_month_data['Temperature'].mean():.1f}°C")
print(f"  Avg Humidity:            {previous_month_data['Humidity'].mean():.1f}%")

# Verify data quality
if len(previous_month_data) < 720:
    print(f"\n⚠ Warning: Previous month has only {len(previous_month_data)} records")
    print(f"  Recommended minimum: 720 records (30 days × 24 hours)")
    print(f"  Forecast may be less accurate with incomplete month data")

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT NEXT MONTH DEMAND
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 80)
print("  GENERATING MONTHLY FORECAST")
print("─" * 80)

# Make monthly forecast using both models
results = predict_monthly_demand(
    previous_month_data,
    model_type="both",
    models_dir="outputs/saved_models"
)

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY AND SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("  MONTHLY FORECAST RESULTS")
print("=" * 80)

# Previous month summary
print(f"\n[Previous Month Summary]")
print(f"  Average Demand: {results['previous_month_avg']:,.1f} MW")
print(f"  Max Demand:     {results['previous_month_max']:,.1f} MW")
print(f"  Min Demand:     {results['previous_month_min']:,.1f} MW")
print(f"  Range:          {results['previous_month_max'] - results['previous_month_min']:,.1f} MW")

# Forecast summary
print(f"\n[Next Month Forecast]")
if 'monthly_demand_forecast' in results:
    forecast_demand = results['monthly_demand_forecast']
    print(f"  Predicted Average: {forecast_demand:,.1f} MW")
    
    if 'monthly_growth_pct' in results:
        growth = results['monthly_growth_pct']
        change_mw = forecast_demand - results['previous_month_avg']
        print(f"  Change from Previous: {change_mw:+,.1f} MW ({growth:+.2f}%)")
        if growth > 0:
            print(f"  Trend: ↑ INCREASING DEMAND")
        elif growth < 0:
            print(f"  Trend: ↓ DECREASING DEMAND")
        else:
            print(f"  Trend: → STABLE DEMAND")
    
    if 'monthly_demand_range' in results:
        min_range, max_range = results['monthly_demand_range']
        print(f"  95% Confidence Range: {min_range:,.1f} - {max_range:,.1f} MW")
        print(f"  Margin of Error: ±{(max_range - min_range) / 2:,.1f} MW")

# Individual model forecasts
if 'xgboost_monthly_avg' in results:
    print(f"\n[XGBoost Model Prediction]")
    print(f"  Average: {results['xgboost_monthly_avg']:,.1f} MW")
    print(f"  Growth:  {results.get('xgboost_growth_pct', 0):+.2f}%")
    print(f"  Std Dev: {results.get('xgboost_monthly_std', 0):,.1f} MW")

if 'lstm_monthly_avg' in results:
    print(f"\n[LSTM Model Prediction]")
    print(f"  Average: {results['lstm_monthly_avg']:,.1f} MW")
    print(f"  Growth:  {results.get('lstm_growth_pct', 0):+.2f}%")
    print(f"  Std Dev: {results.get('lstm_monthly_std', 0):,.1f} MW")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS TO CSV
# ─────────────────────────────────────────────────────────────────────────────

# Create detailed results DataFrame
results_df = pd.DataFrame({
    'Forecast Timestamp': [datetime.now()],
    'Previous Month Avg (MW)': [results.get('previous_month_avg', 0)],
    'Previous Month Max (MW)': [results.get('previous_month_max', 0)],
    'Previous Month Min (MW)': [results.get('previous_month_min', 0)],
    'Predicted Next Month Avg (MW)': [results.get('monthly_demand_forecast', 0)],
    'Expected Growth (%)': [results.get('monthly_growth_pct', 0)],
    'Lower Bound 95% CI (MW)': [results.get('monthly_demand_range', (0, 0))[0]],
    'Upper Bound 95% CI (MW)': [results.get('monthly_demand_range', (0, 0))[1]],
    'XGBoost Prediction (MW)': [results.get('xgboost_monthly_avg', None)],
    'LSTM Prediction (MW)': [results.get('lstm_monthly_avg', None)],
    'Records Analyzed': [results.get('n_records_analyzed', 0)],
})

# Ensure output directory exists
os.makedirs('outputs', exist_ok=True)

# Save to CSV
output_path = 'outputs/monthly_forecast.csv'
results_df.to_csv(output_path, index=False)
print(f"\n✓ Saved detailed forecast to: {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# BUSINESS INSIGHTS & RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("  BUSINESS INSIGHTS")
print("=" * 80)

growth = results.get('monthly_growth_pct', 0)
forecast = results.get('monthly_demand_forecast', 0)
prev_month = results.get('previous_month_avg', 0)

if growth > 5:
    print(f"\n📈 SIGNIFICANT GROWTH EXPECTED")
    print(f"   Next month's demand is expected to increase by {growth:.1f}%")
    print(f"   Recommendation: Ensure adequate generation capacity and supply chain")
    print(f"   Additional load: {forecast - prev_month:,.0f} MW")
elif growth > 0:
    print(f"\n📊 MODERATE GROWTH EXPECTED")
    print(f"   Next month's demand is expected to increase by {growth:.1f}%")
    print(f"   Recommendation: Monitor capacity utilization closely")
elif growth < -5:
    print(f"\n📉 SIGNIFICANT DECLINE EXPECTED")
    print(f"   Next month's demand is expected to decrease by {abs(growth):.1f}%")
    print(f"   Recommendation: Adjust generation schedule and reduce operational costs")
    print(f"   Load reduction: {prev_month - forecast:,.0f} MW")
elif growth < 0:
    print(f"\n📊 MODERATE DECLINE EXPECTED")
    print(f"   Next month's demand is expected to decrease by {abs(growth):.1f}%")
    print(f"   Recommendation: Optimize resource allocation accordingly")
else:
    print(f"\n📊 STABLE DEMAND EXPECTED")
    print(f"   Next month's demand expected to remain similar to previous month")
    print(f"   Recommendation: Maintain current operational schedule")

print("\n" + "=" * 80)
print("  Forecast completed successfully! ✓")
print("=" * 80)
