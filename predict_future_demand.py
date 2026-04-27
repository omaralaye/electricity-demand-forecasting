"""
Usage Example: Forecast Future Electricity Demand
===================================================
This script demonstrates how to use the trained models to forecast
electricity demand when new actual data is provided.
"""

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta

# ─── LOAD THE MAIN MODULE'S FUNCTIONS ──────────────────────────────────────
# If running as standalone, ensure the model file is in the same directory
import sys
sys.path.insert(0, os.path.dirname(__file__))

# Load the prediction functions from the trained model
from electricity_demand_model import predict_future_demand, load_models_and_scalers, DATA_PATH

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1: Create synthetic future data and forecast
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  EXAMPLE 1: Forecast with Synthetic Future Data")
print("=" * 70)

# Load historical data to understand patterns
df_hist = pd.read_csv(DATA_PATH)
print(f"\nHistorical data shape: {df_hist.shape}")
print(f"Columns loaded: {list(df_hist.columns)}")

# Create synthetic future data (e.g., next 7 days at hourly intervals)
# In real usage, this would be actual measured data
future_dates = []
start_date = datetime(2025, 1, 1)  # Example: January 2025

for hour in range(7 * 24):  # 7 days = 168 hours
    current_date = start_date + timedelta(hours=hour)
    future_dates.append(current_date)

# Create new data with realistic patterns
new_data = pd.DataFrame({
    'year': [d.year for d in future_dates],
    'month': [d.month for d in future_dates],
    'dayofmonth': [d.day for d in future_dates],
    'dayofweek': [d.weekday() for d in future_dates],
    'dayofyear': [d.timetuple().tm_yday for d in future_dates],
    'hour': [d.hour for d in future_dates],
})

# Add realistic demand, temperature, humidity (synthetic but reasonable)
np.random.seed(42)
base_demand = 8000  # MW base load
temperature = 15 + 10 * np.sin(2 * np.pi * new_data['hour'] / 24) + np.random.normal(0, 2, len(new_data))
humidity = 60 + 10 * np.sin(2 * np.pi * new_data['dayofweek'] / 7) + np.random.normal(0, 3, len(new_data))

# Demand varies with hour and day
hourly_pattern = 1000 * np.sin(2 * np.pi * (new_data['hour'] - 8) / 24)  # Peak around 8 AM
weekday_pattern = np.where(new_data['dayofweek'] < 5, 500, 0)  # Weekdays slightly higher

new_data['Demand'] = (base_demand + hourly_pattern + weekday_pattern + 
                      np.random.normal(0, 200, len(new_data))).clip(min=5000)
new_data['Temperature'] = temperature
new_data['Humidity'] = humidity

print("\nNew data created (7 days, hourly):")
print(new_data.head(10))
print(f"  Demand range: {new_data['Demand'].min():.0f} - {new_data['Demand'].max():.0f} MW")
print(f"  Temperature range: {new_data['Temperature'].min():.1f} - {new_data['Temperature'].max():.1f}°C")

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT USING BOTH MODELS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 70)
print("  RUNNING PREDICTIONS")
print("─" * 70)

results = predict_future_demand(new_data, model_type="both", models_dir="outputs/saved_models")

print(f"\n✓ Forecast completed for {results['n_predictions']} hours")

if 'xgboost_predictions' in results:
    pred_xgb = results['xgboost_predictions']
    print(f"\nXGBoost predictions (MW):")
    print(f"  Mean:    {pred_xgb.mean():.1f}")
    print(f"  Min:     {pred_xgb.min():.1f}")
    print(f"  Max:     {pred_xgb.max():.1f}")
    print(f"  Std:     {pred_xgb.std():.1f}")
    print(f"  Model R²: {results['xgboost_perf']['R2']:.4f}")

if 'lstm_predictions' in results:
    pred_lstm = results['lstm_predictions']
    print(f"\nLSTM predictions (MW):")
    print(f"  Mean:    {pred_lstm.mean():.1f}")
    print(f"  Min:     {pred_lstm.min():.1f}")
    print(f"  Max:     {pred_lstm.max():.1f}")
    print(f"  Std:     {pred_lstm.std():.1f}")
    print(f"  Model R²: {results['lstm_perf']['R2']:.4f}")

if 'ensemble_predictions' in results:
    pred_ens = results['ensemble_predictions']
    print(f"\nEnsemble predictions (avg of both models) (MW):")
    print(f"  Mean:    {pred_ens.mean():.1f}")
    print(f"  Min:     {pred_ens.min():.1f}")
    print(f"  Max:     {pred_ens.max():.1f}")
    print(f"  Std:     {pred_ens.std():.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS TO CSV
# ─────────────────────────────────────────────────────────────────────────────

output_results = results['data'].copy()

if 'xgboost_predictions' in results and len(results['xgboost_predictions']) == len(output_results):
    output_results['xgboost_forecast'] = results['xgboost_predictions']

if 'lstm_predictions' in results and len(results['lstm_predictions']) == len(output_results):
    # LSTM has fewer predictions due to lookback, pad with NaN
    lstm_preds = results['lstm_predictions']
    offset = len(output_results) - len(lstm_preds)
    output_results['lstm_forecast'] = [np.nan] * offset + list(lstm_preds)

if 'ensemble_predictions' in results and len(results['ensemble_predictions']) == len(output_results):
    ens_preds = results['ensemble_predictions']
    offset = len(output_results) - len(ens_preds)
    output_results['ensemble_forecast'] = [np.nan] * offset + list(ens_preds)

output_csv = os.path.join("outputs", "forecast_example.csv")
os.makedirs("outputs", exist_ok=True)
output_results.to_csv(output_csv, index=False)
print(f"\n[Results] Saved to → {output_csv}")

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 2: Forecast using only XGBoost (faster)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  EXAMPLE 2: XGBoost-Only Forecast (Faster)")
print("=" * 70)

results_xgb = predict_future_demand(new_data, model_type="xgboost", models_dir="outputs/saved_models")
pred_xgb_only = results_xgb['xgboost_predictions']

print(f"\nXGBoost forecast for {len(pred_xgb_only)} hours:")
print(f"  Mean demand: {pred_xgb_only.mean():.1f} MW")
print(f"  Peak hour:   {pred_xgb_only.max():.1f} MW")
print(f"  Low hour:    {pred_xgb_only.min():.1f} MW")

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 3: Load and inspect model performance
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  EXAMPLE 3: Model Performance Summary")
print("=" * 70)

xgb_m, lstm_m, feat_sc, targ_sc, cfg = load_models_and_scalers("outputs/saved_models")

print(f"\nBest Model: {cfg['best_model']}")
print(f"\nXGBoost Performance on Test Data:")
for metric, value in cfg['xgb_performance'].items():
    print(f"  {metric}: {value:.4f}")

print(f"\nLSTM Performance on Test Data:")
for metric, value in cfg['lstm_performance'].items():
    print(f"  {metric}: {value:.4f}")

print("\n" + "=" * 70)
print("✓ Forecast examples completed successfully!")
print("=" * 70)
