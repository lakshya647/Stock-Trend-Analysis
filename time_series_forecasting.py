"""
Module: Time Series Analysis & Forecasting (Lectures 23 - 29, 31, 32)
Implements:
1. Trend, Seasonality, and Noise decomposition (Lecture 24)
2. Exponential Smoothing (Holt-Winters) (Lecture 27)
3. ARIMA Modeling (Lecture 25)
4. SARIMA Modeling (Lecture 26)
5. Evaluation via RMSE & MAE (Lectures 31-32)
Reads directly from stock_data.csv in the root repository.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "stock_data.csv"
FIGURES_PATH = PROJECT_ROOT / "figures"

def run_time_series_pipeline():
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    
    if not DATA_PATH.exists():
        from download_data import main as download_main
        download_main()
        
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    
    # Fill daily gaps with forward fill for time series continuity
    ts_series = df["Close"].asfreq("B").ffill()
    
    print("=" * 60)
    print("STEP 1: TIME SERIES DECOMPOSITION (Lecture 24)")
    print("=" * 60)
    decomposition = seasonal_decompose(ts_series.dropna(), model="additive", period=20)
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.suptitle("Stock Time Series Decomposition: Trend, Seasonality & Noise", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "04_time_series_decomposition.png", dpi=200)
    plt.close()
    print("[+] Decomposition plot saved.")

    # Train-test split (80% train, 20% test for chronological evaluation)
    split_idx = int(len(ts_series) * 0.8)
    train, test = ts_series.iloc[:split_idx], ts_series.iloc[split_idx:]
    print(f"Time Series Split -> Train points: {len(train)}, Test points: {len(test)}")

    results = []

    print("\n" + "=" * 60)
    print("STEP 2: MODEL 1 - EXPONENTIAL SMOOTHING (HOLT-WINTERS) (Lecture 27)")
    print("=" * 60)
    hw_model = ExponentialSmoothing(train, trend="add", seasonal=None, initialization_method="estimated")
    hw_fitted = hw_model.fit()
    hw_pred = hw_fitted.forecast(len(test))
    
    hw_rmse = np.sqrt(mean_squared_error(test, hw_pred))
    hw_mae = mean_absolute_error(test, hw_pred)
    results.append({"Model": "Exponential Smoothing (Holt)", "RMSE": round(hw_rmse, 3), "MAE": round(hw_mae, 3)})
    print(f"Holt-Winters -> RMSE: {hw_rmse:.3f}, MAE: {hw_mae:.3f}")

    print("\n" + "=" * 60)
    print("STEP 3: MODEL 2 - ARIMA FORECASTING (Lecture 25)")
    print("=" * 60)
    arima_model = ARIMA(train, order=(1, 1, 1))
    arima_fitted = arima_model.fit()
    arima_pred = arima_fitted.forecast(steps=len(test))
    
    arima_rmse = np.sqrt(mean_squared_error(test, arima_pred))
    arima_mae = mean_absolute_error(test, arima_pred)
    results.append({"Model": "ARIMA(1, 1, 1)", "RMSE": round(arima_rmse, 3), "MAE": round(arima_mae, 3)})
    print(f"ARIMA(1, 1, 1) -> RMSE: {arima_rmse:.3f}, MAE: {arima_mae:.3f}")

    print("\n" + "=" * 60)
    print("STEP 4: MODEL 3 - SARIMA (SEASONAL ARIMA) (Lecture 26)")
    print("=" * 60)
    sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 5), enforce_stationarity=False)
    sarima_fitted = sarima_model.fit(disp=False)
    sarima_pred = sarima_fitted.forecast(steps=len(test))
    
    sarima_rmse = np.sqrt(mean_squared_error(test, sarima_pred))
    sarima_mae = mean_absolute_error(test, sarima_pred)
    results.append({"Model": "SARIMA(1, 1, 1)(1, 0, 1)5", "RMSE": round(sarima_rmse, 3), "MAE": round(sarima_mae, 3)})
    print(f"SARIMA -> RMSE: {sarima_rmse:.3f}, MAE: {sarima_mae:.3f}")

    # Summary Table
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("TIME SERIES FORECASTING BENCHMARK TABLE (Lectures 31-32)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Forecast comparison plot
    plt.figure(figsize=(14, 6))
    plt.plot(train.index[-100:], train.iloc[-100:], label="Historical Train Price", color="gray", alpha=0.7)
    plt.plot(test.index, test, label="Actual Ground Truth", color="black", linewidth=2)
    plt.plot(test.index, hw_pred, label=f"Exponential Smoothing (RMSE: {hw_rmse:.2f})", linestyle="--", color="blue")
    plt.plot(test.index, arima_pred, label=f"ARIMA (RMSE: {arima_rmse:.2f})", linestyle="--", color="green")
    plt.plot(test.index, sarima_pred, label=f"SARIMA (RMSE: {sarima_rmse:.2f})", linestyle="--", color="red")
    plt.title("Stock Price Forecasting: Comparison of Time-Series Models (Lecture 29)", fontsize=13, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Stock Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "05_forecast_models_comparison.png", dpi=200)
    plt.close()
    print(f"[+] Forecasting plot saved to: {FIGURES_PATH / '05_forecast_models_comparison.png'}")
    
    return results_df

if __name__ == "__main__":
    run_time_series_pipeline()
