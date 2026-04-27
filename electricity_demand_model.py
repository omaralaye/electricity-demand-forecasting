"""
Electricity Demand Forecasting
================================
Models: XGBoost  |  LSTM (Keras / TensorFlow)
Dataset: ~44 K hourly records, 2020-2024
"""

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH = r"D:\electricity demand forecasting\electricity demand dataset.csv"
OUT_DIR     = "outputs"
LOOKBACK    = 24          # LSTM: hours of history used per sample
LSTM_EPOCHS = 60
LSTM_BATCH  = 128
SEED        = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── 1. LOAD & CLEAN ──────────────────────────────────────────────────────────
print("=" * 60)
print("  ELECTRICITY DEMAND FORECASTING — XGBoost + LSTM")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[Data] Loaded {len(df):,} rows × {df.shape[1]} columns")

# Drop rows missing key columns
df.dropna(subset=["Demand", "Temperature", "Humidity", "hour",
                  "dayofweek", "month", "year", "dayofyear"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"[Data] After cleaning: {len(df):,} rows")

# ─── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────
# Cyclic encodings so the model understands circular time
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]      / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]      / 24)
df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"]     / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"]     / 12)
df["doy_sin"]   = np.sin(2 * np.pi * df["dayofyear"] / 365)
df["doy_cos"]   = np.cos(2 * np.pi * df["dayofyear"] / 365)

# Lag features (previous demand hours)
for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
    df[f"demand_lag_{lag}"] = df["Demand"].shift(lag)

# Rolling statistics
df["demand_roll_mean_24"]  = df["Demand"].shift(1).rolling(24).mean()
df["demand_roll_std_24"]   = df["Demand"].shift(1).rolling(24).std()
df["demand_roll_mean_168"] = df["Demand"].shift(1).rolling(168).mean()

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"[Features] After lag/rolling creation: {len(df):,} rows, {df.shape[1]} columns")

FEATURES = [
    "Temperature", "Humidity",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    "demand_lag_1", "demand_lag_2", "demand_lag_3",
    "demand_lag_6", "demand_lag_12", "demand_lag_24",
    "demand_lag_48", "demand_lag_168",
    "demand_roll_mean_24", "demand_roll_std_24", "demand_roll_mean_168",
]
TARGET = "Demand"

# ─── 3. TRAIN / TEST SPLIT  (last year = test) ────────────────────────────────
cutoff = df["year"].max()
train  = df[df["year"] < cutoff].copy()
test   = df[df["year"] == cutoff].copy()
print(f"\n[Split] Train: {len(train):,} rows | Test: {len(test):,} rows (year={cutoff})")

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: XGBoost
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  MODEL 1: XGBoost")
print("─" * 60)

xgb_model = xgb.XGBRegressor(
    n_estimators      = 1000,
    learning_rate     = 0.05,
    max_depth         = 7,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 3,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = SEED,
    n_jobs            = -1,
    early_stopping_rounds = 50,
    eval_metric       = "rmse",
    verbosity         = 0,
)

xgb_model.fit(
    X_train, y_train,
    eval_set = [(X_test, y_test)],
    verbose  = False,
)

xgb_pred  = xgb_model.predict(X_test)
xgb_mae   = mean_absolute_error(y_test, xgb_pred)
xgb_rmse  = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_mape  = np.mean(np.abs((y_test - xgb_pred) / y_test)) * 100
xgb_r2    = r2_score(y_test, xgb_pred)

print(f"  MAE : {xgb_mae:,.1f} MW")
print(f"  RMSE: {xgb_rmse:,.1f} MW")
print(f"  MAPE: {xgb_mape:.2f}%")
print(f"  R²  : {xgb_r2:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: LSTM
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  MODEL 2: LSTM")
print("─" * 60)

# Scale features and target separately
feat_scaler   = MinMaxScaler()
target_scaler = MinMaxScaler()

X_train_s = feat_scaler.fit_transform(X_train)
X_test_s  = feat_scaler.transform(X_test)
y_train_s = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_s  = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

def make_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback: i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_tr_seq, y_tr_seq = make_sequences(X_train_s, y_train_s, LOOKBACK)
X_te_seq, y_te_seq = make_sequences(X_test_s,  y_test_s,  LOOKBACK)
print(f"[LSTM] Sequence shapes — train: {X_tr_seq.shape}, test: {X_te_seq.shape}")

n_features = X_tr_seq.shape[2]

lstm_model = Sequential([
    Input(shape=(LOOKBACK, n_features)),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1),
])
lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
lstm_model.summary(print_fn=lambda x: None)   # suppress verbose output

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(factor=0.5, patience=5, verbose=0),
]

print("[LSTM] Training …\n")
history = lstm_model.fit(
    X_tr_seq, y_tr_seq,
    epochs          = LSTM_EPOCHS,
    batch_size      = LSTM_BATCH,
    validation_split= 0.1,
    callbacks       = callbacks,
    verbose         = 1,
)
print(f"\n[LSTM] Training completed — stopped at epoch {len(history.history['loss'])}/{LSTM_EPOCHS}")

lstm_pred_s = lstm_model.predict(X_te_seq, verbose=0).ravel()
lstm_pred   = target_scaler.inverse_transform(lstm_pred_s.reshape(-1, 1)).ravel()

# Align y_test with the LSTM sequences (first LOOKBACK rows are consumed)
y_test_lstm = y_test.values[LOOKBACK:]

lstm_mae  = mean_absolute_error(y_test_lstm, lstm_pred)
lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm, lstm_pred))
lstm_mape = np.mean(np.abs((y_test_lstm - lstm_pred) / y_test_lstm)) * 100
lstm_r2   = r2_score(y_test_lstm, lstm_pred)

print(f"  MAE : {lstm_mae:,.1f} MW")
print(f"  RMSE: {lstm_rmse:,.1f} MW")
print(f"  MAPE: {lstm_mape:.2f}%")
print(f"  R²  : {lstm_r2:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Plots] Generating …")

SAMPLE = 7 * 24   # one week of hourly predictions for detail plot

fig = plt.figure(figsize=(20, 22), facecolor="#0f0f1a")
gs  = gridspec.GridSpec(4, 2, figure=fig,
                         hspace=0.45, wspace=0.3,
                         left=0.06, right=0.97,
                         top=0.93, bottom=0.05)

TEXT_COLOR = "#e0e0f0"
GRID_COLOR = "#2a2a40"
ACCENT1    = "#00c6ff"   # XGBoost
ACCENT2    = "#f97316"   # LSTM
ACTUAL_C   = "#a78bfa"

def style_ax(ax, title):
    ax.set_facecolor("#131326")
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.5, linestyle="--")
    ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)

# ── (0,0)+(0,1): Metrics comparison bar chart ─────────────────────────────
ax_bar = fig.add_subplot(gs[0, :])
metrics     = ["MAE (MW)", "RMSE (MW)", "MAPE (%)", "R²"]
xgb_vals    = [xgb_mae, xgb_rmse, xgb_mape, xgb_r2]
lstm_vals   = [lstm_mae, lstm_rmse, lstm_mape, lstm_r2]
x = np.arange(len(metrics))
w = 0.35
b1 = ax_bar.bar(x - w/2, xgb_vals,  w, color=ACCENT1, alpha=0.85, label="XGBoost", zorder=3)
b2 = ax_bar.bar(x + w/2, lstm_vals, w, color=ACCENT2, alpha=0.85, label="LSTM",    zorder=3)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, h * 1.02,
                f"{h:.2f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(metrics)
ax_bar.legend(facecolor="#1e1e36", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=10)
style_ax(ax_bar, "Model Performance Comparison")

# ── (1,0): XGBoost — one-week forecast detail ─────────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(y_test.values[:SAMPLE],  color=ACTUAL_C, lw=1.2, label="Actual",   alpha=0.9)
ax1.plot(xgb_pred[:SAMPLE],       color=ACCENT1,  lw=1.2, label="XGBoost",  alpha=0.9)
ax1.set_xlabel("Hours (test set — first week)")
ax1.set_ylabel("Demand (MW)")
ax1.legend(facecolor="#1e1e36", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
style_ax(ax1, "XGBoost — 7-Day Forecast Detail")

# ── (1,1): LSTM — one-week forecast detail ────────────────────────────────
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(y_test_lstm[:SAMPLE], color=ACTUAL_C, lw=1.2, label="Actual", alpha=0.9)
ax2.plot(lstm_pred[:SAMPLE],   color=ACCENT2,  lw=1.2, label="LSTM",   alpha=0.9)
ax2.set_xlabel("Hours (test set — first week)")
ax2.set_ylabel("Demand (MW)")
ax2.legend(facecolor="#1e1e36", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
style_ax(ax2, "LSTM — 7-Day Forecast Detail")

# ── (2,0): XGBoost Scatter ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
lim = [min(y_test.min(), xgb_pred.min()) * 0.98,
       max(y_test.max(), xgb_pred.max()) * 1.02]
ax3.scatter(y_test, xgb_pred, alpha=0.15, s=3, color=ACCENT1)
ax3.plot(lim, lim, "w--", lw=1)
ax3.set_xlim(lim); ax3.set_ylim(lim)
ax3.set_xlabel("Actual Demand (MW)"); ax3.set_ylabel("Predicted Demand (MW)")
style_ax(ax3, f"XGBoost  — Actual vs Predicted  (R²={xgb_r2:.4f})")

# ── (2,1): LSTM Scatter ───────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
lim2 = [min(y_test_lstm.min(), lstm_pred.min()) * 0.98,
        max(y_test_lstm.max(), lstm_pred.max()) * 1.02]
ax4.scatter(y_test_lstm, lstm_pred, alpha=0.15, s=3, color=ACCENT2)
ax4.plot(lim2, lim2, "w--", lw=1)
ax4.set_xlim(lim2); ax4.set_ylim(lim2)
ax4.set_xlabel("Actual Demand (MW)"); ax4.set_ylabel("Predicted Demand (MW)")
style_ax(ax4, f"LSTM  — Actual vs Predicted  (R²={lstm_r2:.4f})")

# ── (3,0): XGBoost feature importance ─────────────────────────────────────
ax5 = fig.add_subplot(gs[3, 0])
imp = pd.Series(xgb_model.feature_importances_, index=FEATURES).nlargest(15)
ax5.barh(imp.index[::-1], imp.values[::-1], color=ACCENT1, alpha=0.85)
ax5.set_xlabel("Feature Importance")
style_ax(ax5, "XGBoost — Top 15 Feature Importances")

# ── (3,1): LSTM training history ──────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, 1])
ax6.plot(history.history["loss"],     color=ACCENT2,  lw=1.5, label="Train Loss")
ax6.plot(history.history["val_loss"], color=ACCENT1,  lw=1.5, linestyle="--", label="Val Loss")
ax6.set_xlabel("Epoch"); ax6.set_ylabel("MSE Loss")
ax6.legend(facecolor="#1e1e36", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
style_ax(ax6, "LSTM — Training History")

# Title
fig.suptitle("Electricity Demand Forecasting  ·  XGBoost vs LSTM",
             fontsize=15, fontweight="bold", color=TEXT_COLOR, y=0.975)

out_plot = os.path.join(OUT_DIR, "electricity_demand_results.png")
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(out_plot, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[Plots] Saved → {out_plot}")

# ─── SUMMARY TABLE ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"{'Metric':<20} {'XGBoost':>12} {'LSTM':>12}")
print("-" * 44)
print(f"{'MAE (MW)':<20} {xgb_mae:>12,.1f} {lstm_mae:>12,.1f}")
print(f"{'RMSE (MW)':<20} {xgb_rmse:>12,.1f} {lstm_rmse:>12,.1f}")
print(f"{'MAPE (%)':<20} {xgb_mape:>12.2f} {lstm_mape:>12.2f}")
print(f"{'R²':<20} {xgb_r2:>12.4f} {lstm_r2:>12.4f}")
print("=" * 60)

winner = "XGBoost" if xgb_r2 >= lstm_r2 else "LSTM"
print(f"\n  Best model by R²: {winner}")

# ─── 4. SAVE MODELS & SCALERS ─────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  SAVING MODELS & SCALERS")
print("─" * 60)

import pickle

models_dir = os.path.join(OUT_DIR, "saved_models")
os.makedirs(models_dir, exist_ok=True)

# Save XGBoost
xgb_path = os.path.join(models_dir, "xgboost_model.pkl")
with open(xgb_path, "wb") as f:
    pickle.dump(xgb_model, f)
print(f"[XGBoost] Model saved → {xgb_path}")

# Save LSTM
lstm_path = os.path.join(models_dir, "lstm_model.h5")
lstm_model.save(lstm_path, save_format='h5')
print(f"[LSTM] Model saved → {lstm_path}")

# Save scalers
feat_scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
target_scaler_path = os.path.join(models_dir, "target_scaler.pkl")
with open(feat_scaler_path, "wb") as f:
    pickle.dump(feat_scaler, f)
with open(target_scaler_path, "wb") as f:
    pickle.dump(target_scaler, f)
print(f"[Scalers] Saved → {feat_scaler_path}, {target_scaler_path}")

# Save config
config = {
    "FEATURES": FEATURES,
    "TARGET": TARGET,
    "LOOKBACK": LOOKBACK,
    "SEED": SEED,
    "best_model": winner,
    "xgb_performance": {"MAE": xgb_mae, "RMSE": xgb_rmse, "MAPE": xgb_mape, "R2": xgb_r2},
    "lstm_performance": {"MAE": lstm_mae, "RMSE": lstm_rmse, "MAPE": lstm_mape, "R2": lstm_r2},
}
config_path = os.path.join(models_dir, "config.pkl")
with open(config_path, "wb") as f:
    pickle.dump(config, f)
print(f"[Config] Saved → {config_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FUNCTIONS FOR FUTURE FORECASTING
# ─────────────────────────────────────────────────────────────────────────────

def load_models_and_scalers(models_dir="outputs/saved_models"):
    """Load trained models and scalers."""
    with open(os.path.join(models_dir, "xgboost_model.pkl"), "rb") as f:
        xgb_m = pickle.load(f)
    lstm_m = tf.keras.models.load_model(os.path.join(models_dir, "lstm_model.h5"))
    with open(os.path.join(models_dir, "feature_scaler.pkl"), "rb") as f:
        feat_sc = pickle.load(f)
    with open(os.path.join(models_dir, "target_scaler.pkl"), "rb") as f:
        targ_sc = pickle.load(f)
    with open(os.path.join(models_dir, "config.pkl"), "rb") as f:
        cfg = pickle.load(f)
    return xgb_m, lstm_m, feat_sc, targ_sc, cfg

def prepare_features_for_prediction(new_data, historical_data, features_list, lookback=24):
    """
    Prepare features for prediction, handling lags and rolling stats.
    
    Parameters:
    -----------
    new_data : pd.DataFrame
        New data rows with: Demand, Temperature, Humidity, hour, dayofweek, month, year, dayofyear
    historical_data : pd.DataFrame
        Full historical data (for lag features and rolling stats)
    features_list : list
        List of required feature names
    lookback : int
        Number of hours for LSTM lookback
        
    Returns:
    --------
    pd.DataFrame with all engineered features
    """
    # Combine historical + new data for proper lag calculation
    combined = pd.concat([historical_data, new_data], ignore_index=True)
    
    # Cyclic encodings
    combined["hour_sin"]  = np.sin(2 * np.pi * combined["hour"]      / 24)
    combined["hour_cos"]  = np.cos(2 * np.pi * combined["hour"]      / 24)
    combined["dow_sin"]   = np.sin(2 * np.pi * combined["dayofweek"] / 7)
    combined["dow_cos"]   = np.cos(2 * np.pi * combined["dayofweek"] / 7)
    combined["month_sin"] = np.sin(2 * np.pi * combined["month"]     / 12)
    combined["month_cos"] = np.cos(2 * np.pi * combined["month"]     / 12)
    combined["doy_sin"]   = np.sin(2 * np.pi * combined["dayofyear"] / 365)
    combined["doy_cos"]   = np.cos(2 * np.pi * combined["dayofyear"] / 365)
    
    # Lag features
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        combined[f"demand_lag_{lag}"] = combined["Demand"].shift(lag)
    
    # Rolling statistics
    combined["demand_roll_mean_24"]  = combined["Demand"].shift(1).rolling(24).mean()
    combined["demand_roll_std_24"]   = combined["Demand"].shift(1).rolling(24).std()
    combined["demand_roll_mean_168"] = combined["Demand"].shift(1).rolling(168).mean()
    
    # Extract only the new data rows (with all features)
    result = combined.iloc[-len(new_data):].copy()
    result = result.dropna()  # Remove rows with NaN (from lag/rolling)
    
    return result[features_list]

def forecast_xgboost(new_data, xgb_model, feat_scaler, features_list):
    """Forecast using XGBoost model."""
    X_new = new_data[features_list].copy()
    predictions = xgb_model.predict(X_new)
    return predictions

def forecast_lstm(new_data_scaled, lookback, lstm_model, target_scaler):
    """Forecast using LSTM model."""
    # Build sequences from scaled data
    X_seq = []
    X_array = new_data_scaled.values
    for i in range(lookback, len(X_array)):
        X_seq.append(X_array[i - lookback: i])
    
    if len(X_seq) == 0:
        print("[Warning] Not enough data for LSTM sequences (need at least lookback={} rows)".format(lookback))
        return np.array([])
    
    X_seq = np.array(X_seq)
    pred_scaled = lstm_model.predict(X_seq, verbose=0).ravel()
    predictions = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    return predictions

def predict_future_demand(new_data, model_type="both", models_dir="outputs/saved_models"):
    """
    Predict future electricity demand.
    
    Parameters:
    -----------
    new_data : pd.DataFrame
        DataFrame with columns: Demand, Temperature, Humidity, hour, dayofweek, month, year, dayofyear
        Must include at least 1 row, preferably 168+ rows for stable rolling stats.
    model_type : str
        "xgboost", "lstm", or "both" (default)
    models_dir : str
        Path to saved models directory
        
    Returns:
    --------
    dict with predictions and metadata
    """
    # Load models
    xgb_m, lstm_m, feat_sc, targ_sc, cfg = load_models_and_scalers(models_dir)
    
    # Load original dataset for context (for lag features)
    df_hist = pd.read_csv(DATA_PATH)
    df_hist.dropna(subset=cfg["TARGET"], inplace=True)
    
    # Prepare features
    X_new = prepare_features_for_prediction(new_data, df_hist, cfg["FEATURES"], cfg["LOOKBACK"])
    
    results = {
        "n_predictions": len(X_new),
        "data": new_data.iloc[-len(X_new):].copy() if len(X_new) > 0 else pd.DataFrame(),
    }
    
    if len(X_new) == 0:
        print("[Error] No valid data for prediction after feature engineering.")
        return results
    
    # XGBoost forecast
    if model_type in ["xgboost", "both"]:
        pred_xgb = forecast_xgboost(X_new, xgb_m, feat_sc, cfg["FEATURES"])
        results["xgboost_predictions"] = pred_xgb
        results["xgboost_perf"] = cfg["xgb_performance"]
    
    # LSTM forecast
    if model_type in ["lstm", "both"]:
        X_new_scaled = feat_sc.transform(X_new)
        pred_lstm = forecast_lstm(pd.DataFrame(X_new_scaled), cfg["LOOKBACK"], lstm_m, targ_sc)
        results["lstm_predictions"] = pred_lstm
        results["lstm_perf"] = cfg["lstm_performance"]
    
    # Ensemble prediction (average)
    if model_type == "both" and len(X_new) > 0:
        if len(pred_xgb) == len(pred_lstm):
            results["ensemble_predictions"] = (pred_xgb + pred_lstm) / 2
        else:
            # Align predictions (LSTM may have fewer due to lookback requirement)
            min_len = min(len(pred_xgb), len(pred_lstm))
            results["ensemble_predictions"] = (pred_xgb[-min_len:] + pred_lstm[-min_len:]) / 2
    
    return results

print("\n[Functions] Prediction functions loaded and ready!")
print("  → Call predict_future_demand(new_data_df) to forecast")
print("  → new_data_df must have: Demand, Temperature, Humidity, hour, dayofweek, month, year, dayofyear")
print("Done ✓")
