# Monthly Electricity Demand Forecasting Guide

## Overview

Your model has been enhanced to **forecast next month's electricity demand using the previous month's data**. This is in addition to the existing hourly/daily forecasting capabilities.

## What's New

### 1. **Monthly Features Added**
The model now includes monthly-level aggregation features that capture patterns over 30-day periods:
- `demand_roll_mean_720`: Average demand over previous 30 days
- `demand_roll_std_720`: Variability of demand over previous 30 days  
- `demand_roll_max_720`: Peak demand over previous 30 days
- `demand_roll_min_720`: Minimum demand over previous 30 days

### 2. **New Monthly Forecasting Function**
A new function `predict_monthly_demand()` has been added to `electricity_demand_model.py` that:
- Analyzes entire month of historical data (last 30 days)
- Predicts average demand for next month
- Provides confidence intervals (95% CI)
- Estimates growth/decline percentage
- Uses both XGBoost and LSTM models with ensemble averaging

### 3. **Updated Scripts**
- **input.py**: Now includes both hourly and monthly forecasting examples
- **forecast_monthly.py**: Dedicated script for monthly forecasting with comprehensive examples

## How to Use

### Option 1: Quick Monthly Forecast (Recommended)

Run the dedicated monthly forecasting script:

```bash
python forecast_monthly.py
```

This will:
1. Generate or load previous month's data
2. Analyze patterns and statistics
3. Forecast next month's average demand
4. Display confidence intervals and trends
5. Save results to `outputs/monthly_forecast.csv`

### Option 2: Use in Your Code

```python
from electricity_demand_model import predict_monthly_demand
import pandas as pd

# Load your previous month's data (must have 720+ hourly records for accuracy)
previous_month_data = pd.read_csv('march_2024_data.csv')

# Make monthly forecast
results = predict_monthly_demand(
    previous_month_data,
    model_type="both",  # "xgboost", "lstm", or "both"
    models_dir="outputs/saved_models"
)

# Access results
print(f"Previous Month Avg: {results['previous_month_avg']:.1f} MW")
print(f"Next Month Forecast: {results['monthly_demand_forecast']:.1f} MW")
print(f"Expected Growth: {results['monthly_growth_pct']:+.2f}%")
print(f"Confidence Range: {results['monthly_demand_range']}")
```

### Option 3: Use input.py for Both Hourly and Monthly Forecasting

```bash
python input.py
```

This script now supports both:
- **Hourly/Daily Forecasting**: For short-term predictions (hours to days ahead)
- **Monthly Forecasting**: For next month predictions (requires 720+ records)

## Input Data Requirements

### For Monthly Forecasting:
- **Minimum 720 hourly records** (preferably exactly 744 for a full calendar month)
- **Required columns**: `Demand`, `Temperature`, `Humidity`, `hour`, `dayofweek`, `month`, `year`, `dayofyear`
- **Data from**: Complete previous calendar month (or 30 consecutive days)

### Example Data Structure:
```
year,month,dayofmonth,dayofweek,dayofyear,hour,Demand,Temperature,Humidity
2024,3,1,4,61,0,7500.5,12.3,62.1
2024,3,1,4,61,1,7200.2,11.8,64.5
2024,3,1,4,61,2,6950.1,11.2,66.2
...
2024,3,31,6,91,23,8100.3,14.5,61.8
```

## Output Interpretation

The monthly forecast provides:

1. **Previous Month Statistics**
   - Average, max, min, and standard deviation of demand
   - Temperature and humidity patterns

2. **Next Month Forecast**
   - Predicted average demand (MW)
   - Expected growth/decline percentage
   - 95% confidence interval (lower and upper bounds)
   - Margin of error

3. **Model-Specific Predictions**
   - XGBoost model prediction and growth rate
   - LSTM model prediction and growth rate
   - Ensemble average (combined prediction)

4. **Business Insights**
   - Trend indicators (increasing ↑, decreasing ↓, stable →)
   - Recommendations for capacity planning
   - Load change magnitude in MW

## Example Output

```
PREVIOUS MONTH SUMMARY
  Average Demand: 7,856.3 MW
  Max Demand:     9,245.1 MW
  Min Demand:     5,823.7 MW
  Range:          3,421.4 MW

NEXT MONTH FORECAST
  Predicted Average: 8,125.7 MW
  Change from Previous: +269.4 MW (+3.43%)
  Trend: ↑ INCREASING DEMAND
  95% Confidence Range: 7,890.2 - 8,361.2 MW
  Margin of Error: ±235.5 MW
```

## Model Performance

The models have been trained on historical data (2020-2024) with:
- **XGBoost**: Good for capturing complex feature interactions
- **LSTM**: Excellent for temporal patterns and sequences
- **Ensemble**: Combines both models for robust predictions

Performance metrics available in:
- `outputs/saved_models/config.pkl`
- XGBoost R² score: Check console output during training
- LSTM R² score: Check console output during training

## When to Retrain

Retrain the model if:
1. New year of data becomes available (annual retraining recommended)
2. Major changes in electricity demand patterns (infrastructure changes, etc.)
3. Accuracy drifts significantly from baseline

To retrain:
```bash
python electricity_demand_model.py
```

## Troubleshooting

### "Not enough data for monthly forecasting"
**Solution**: Ensure you have at least 720 hourly records (30 days of hourly data). Missing records will affect rolling statistics calculation.

### "Could not prepare monthly features"
**Solution**: Verify all required columns exist in your data: `Demand`, `Temperature`, `Humidity`, `hour`, `dayofweek`, `month`, `year`, `dayofyear`

### "LSTM sequences are empty"
**Solution**: Data has insufficient history. Need at least lookback (24 hours) + 720 monthly hours of context.

### "Predictions seem unrealistic"
**Solution**: Check for data quality issues:
- Remove outliers from demand data
- Verify temperature/humidity readings are realistic
- Ensure no missing values or NaN entries

## API Reference

### predict_monthly_demand()

```python
predict_monthly_demand(
    previous_month_data,      # pd.DataFrame with 720+ rows
    model_type="both",         # "xgboost", "lstm", or "both"
    models_dir="outputs/saved_models"
)

Returns: dict with keys:
  - 'monthly_demand_forecast': Next month avg demand (MW)
  - 'monthly_growth_pct': % change from previous month
  - 'monthly_demand_range': (lower_bound, upper_bound) for 95% CI
  - 'xgboost_monthly_avg': XGBoost model prediction
  - 'lstm_monthly_avg': LSTM model prediction
  - 'previous_month_avg': Previous month actual average
  - And many other detailed metrics
```

### predict_future_demand() [Existing - Still Available]

Used for hourly/daily forecasting when you have recent data:

```python
predict_future_demand(
    new_data,                  # pd.DataFrame with hourly/daily records
    model_type="both",
    models_dir="outputs/saved_models"
)
```

## Examples

See the following files for working examples:
- **forecast_monthly.py** - Complete monthly forecasting example with synthetic data generation
- **input.py** - Both hourly and monthly forecasting in one script
- **predict_future_demand.py** - Original hourly forecasting examples

## Next Steps

1. **Test with your actual data**: Run `forecast_monthly.py` with your previous month's electricity demand data
2. **Integrate into workflow**: Use `predict_monthly_demand()` in your production pipeline
3. **Monitor accuracy**: Compare predictions with actual next-month results
4. **Refine inputs**: Adjust temperature/humidity data quality for better accuracy
5. **Retrain periodically**: Especially when seasonal patterns change

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the example scripts for usage patterns
3. Verify data format matches requirements
4. Check console output for detailed error messages

---

**Version**: 2.0 with Monthly Forecasting  
**Last Updated**: April 27, 2026  
**Models**: XGBoost + LSTM with Ensemble
