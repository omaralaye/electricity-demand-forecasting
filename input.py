import pandas as pd
from electricity_demand_model import predict_future_demand, predict_monthly_demand

# Method 1: Load from CSV file
# Your CSV should have the columns: Demand, Temperature, Humidity, hour, dayofweek, month, year, dayofyear
try:
    new_data = pd.read_csv('data.csv')
    print(f"Loaded data from data.csv with shape: {new_data.shape}")
    print(f"Columns: {list(new_data.columns)}")
except FileNotFoundError:
    print("data.csv not found. Using example data instead...")
    # Method 2: Create example DataFrame programmatically
    new_data = pd.DataFrame({
        'Demand': [8500, 8200, 8100, 8000],  # Your actual demand values
        'Temperature': [22.5, 21.8, 23.1, 24.2],  # Your temperature readings
        'Humidity': [65.2, 67.1, 63.8, 62.5],  # Your humidity readings
        'hour': [0, 1, 2, 3],  # 0-23
        'dayofweek': [1, 1, 1, 1],  # 0=Monday, 6=Sunday
        'month': [4, 4, 4, 4],  # 1-12
        'year': [2024, 2024, 2024, 2024],
        'dayofyear': [118, 118, 118, 118]  # Day of year
    })

# ─────────────────────────────────────────────────────────────────────────────
# OPTION 1: HOURLY/DAILY FORECASTING (existing functionality)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  HOURLY/DAILY FORECASTING")
print("=" * 70)

# Make predictions
print("Starting predictions...")
results = predict_future_demand(new_data, model_type="both", models_dir="outputs/saved_models")

# Access predictions
forecast_table = results['data'].copy().reset_index(drop=True)

if 'xgboost_predictions' in results:
    xgb_forecast = results['xgboost_predictions']
    forecast_table['xgboost_forecast'] = xgb_forecast
    print(f"XGBoost predictions: {xgb_forecast}")

if 'lstm_predictions' in results:
    lstm_forecast = results['lstm_predictions']
    pad_length = len(forecast_table) - len(lstm_forecast)
    forecast_table['lstm_forecast'] = [pd.NA] * pad_length + list(lstm_forecast)
    print(f"LSTM predictions: {lstm_forecast}")

if 'ensemble_predictions' in results:
    ensemble_forecast = results['ensemble_predictions']  # Average of both models
    pad_length = len(forecast_table) - len(ensemble_forecast)
    forecast_table['ensemble_forecast'] = [pd.NA] * pad_length + list(ensemble_forecast)
    print(f"Ensemble predictions: {ensemble_forecast}")

print("\nForecast results table:")
if forecast_table.empty:
    print("No forecast results available. Check input data and model feature requirements.")
else:
    print(forecast_table.to_string(index=False))

output_path = 'outputs/forecast_results.csv'
forecast_table.to_csv(output_path, index=False)
print(f"\nSaved forecast table to {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# OPTION 2: MONTHLY FORECASTING (NEW: use previous month to forecast next month)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  MONTHLY FORECASTING (Previous Month → Next Month)")
print("=" * 70)

# For monthly forecasting, you need data from the previous complete month
# This example assumes you have 30 days (720 hourly records) of previous month data
if len(new_data) >= 720:
    print(f"\n[Monthly Mode] Using {len(new_data)} hourly records from previous month...")
    monthly_results = predict_monthly_demand(new_data, model_type="both", models_dir="outputs/saved_models")
    
    print(f"\n[Monthly Forecast Summary]")
    print(f"  Previous Month Average: {monthly_results.get('previous_month_avg', 0):,.1f} MW")
    if 'monthly_demand_forecast' in monthly_results:
        print(f"  Next Month Forecast:    {monthly_results['monthly_demand_forecast']:,.1f} MW")
        print(f"  Expected Growth:        {monthly_results.get('monthly_growth_pct', 0):+.2f}%")
    if 'monthly_demand_range' in monthly_results:
        min_range, max_range = monthly_results['monthly_demand_range']
        print(f"  95% Confidence Range:   {min_range:,.1f} - {max_range:,.1f} MW")
    
    # Save monthly results
    monthly_output = pd.DataFrame([{
        'Forecast Date': pd.Timestamp.now(),
        'Previous Month Avg (MW)': monthly_results.get('previous_month_avg', 0),
        'Next Month Forecast (MW)': monthly_results.get('monthly_demand_forecast', 0),
        'Expected Growth (%)': monthly_results.get('monthly_growth_pct', 0),
        'Range Min (MW)': monthly_results.get('monthly_demand_range', (0, 0))[0],
        'Range Max (MW)': monthly_results.get('monthly_demand_range', (0, 0))[1],
    }])
    monthly_path = 'outputs/monthly_forecast.csv'
    monthly_output.to_csv(monthly_path, index=False)
    print(f"\nSaved monthly forecast to {monthly_path}")
else:
    print(f"\n[Monthly Mode] Requires at least 720 hourly records (30 days)")
    print(f"  Current records: {len(new_data)}")
    print(f"  Please provide a full month of previous data for accurate monthly forecasting")

print("\nPredictions completed successfully!")