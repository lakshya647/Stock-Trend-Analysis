"""
Module: EDA and Preprocessing (Lectures 5 - 11)
Performs exploratory data analysis, statistical summaries (mean, median, variance, std dev),
outlier detection (IQR and Z-score), feature engineering, and visualization saving.
Reads from and writes directly to repository root.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "stock_data.csv"
CLEAN_PATH = PROJECT_ROOT / "stock_features_processed.csv"
FIGURES_PATH = PROJECT_ROOT / "figures"

def run_eda_pipeline():
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    
    if not DATA_PATH.exists():
        from download_data import main as download_main
        download_main()

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    
    print("=" * 60)
    print("STEP 1: DATA EXPLORATION & DESCRIPTIVE STATISTICS (Lectures 7-8)")
    print("=" * 60)
    print(f"Total Rows: {len(df)}, Total Features: {df.shape[1]}")
    print("\nMissing Values Count:")
    print(df.isnull().sum())
    
    # Statistical measures: Mean, Median, Mode, Variance, Standard Deviation
    stats_df = pd.DataFrame({
        "Mean": df[["Open", "High", "Low", "Close", "Volume"]].mean(),
        "Median": df[["Open", "High", "Low", "Close", "Volume"]].median(),
        "Variance": df[["Open", "High", "Low", "Close", "Volume"]].var(),
        "Std_Dev": df[["Open", "High", "Low", "Close", "Volume"]].std()
    })
    print("\nDescriptive Statistical Summary:")
    print(stats_df)
    
    print("\n" + "=" * 60)
    print("STEP 2: OUTLIER DETECTION VIA IQR & Z-SCORE (Lectures 6 & 8)")
    print("=" * 60)
    
    # Z-Score computation for Close price
    mean_close = df["Close"].mean()
    std_close = df["Close"].std()
    df["Close_Z_Score"] = (df["Close"] - mean_close) / std_close
    z_outliers = df[np.abs(df["Close_Z_Score"]) > 3]
    print(f"Number of outliers detected by Z-Score (|Z| > 3): {len(z_outliers)}")
    
    # IQR Method for Daily Volume
    q1 = df["Volume"].quantile(0.25)
    q3 = df["Volume"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    iqr_outliers = df[(df["Volume"] < lower_bound) | (df["Volume"] > upper_bound)]
    print(f"Number of Volume outliers detected by IQR method: {len(iqr_outliers)}")
    
    print("\n" + "=" * 60)
    print("STEP 3: FEATURE ENGINEERING (Lectures 10-11)")
    print("=" * 60)
    
    # Technical Indicators for Predictive Analysis
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility_10"] = df["Daily_Return"].rolling(window=10).std()
    
    # Relative Strength Index (RSI - 14)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    # Target variable: Direction (1 = Uptrend tomorrow, 0 = Downtrend tomorrow)
    df["Tomorrow_Close"] = df["Close"].shift(-1)
    df["Target_Direction"] = (df["Tomorrow_Close"] > df["Close"]).astype(int)
    
    # Drop rows with NaN
    df_clean = df.dropna().copy().reset_index(drop=True)
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"[+] Processed features saved directly to: {CLEAN_PATH}")

    print("\n" + "=" * 60)
    print("STEP 4: GENERATING DATA VISUALIZATIONS (Lecture 9)")
    print("=" * 60)
    
    # Plot 1: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr_cols = ["Open", "High", "Low", "Close", "Volume", "SMA_10", "SMA_50", "RSI_14", "Volatility_10", "Target_Direction"]
    corr_matrix = df_clean[corr_cols].corr()
    if sns is not None:
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    else:
        plt.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
        plt.yticks(range(len(corr_cols)), corr_cols)
    plt.title("Correlation Matrix Heatmap (Lecture 9)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "01_correlation_heatmap.png", dpi=200)
    plt.close()
    
    # Plot 2: Box Plot for Outlier Visualization
    plt.figure(figsize=(8, 5))
    if sns is not None:
        sns.boxplot(y=df["Volume"], color="skyblue")
    else:
        plt.boxplot(df["Volume"].dropna(), patch_artist=True)
    plt.title("Outlier Analysis on Trading Volume (IQR Box Plot)", fontsize=13)
    plt.ylabel("Volume")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "02_volume_boxplot_outliers.png", dpi=200)
    plt.close()

    # Plot 3: Scatter Plot & Histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if sns is not None:
        sns.scatterplot(x="Daily_Return", y="Volume", data=df_clean, ax=axes[0], alpha=0.6, color="purple")
        sns.histplot(df_clean["Daily_Return"], bins=40, kde=True, ax=axes[1], color="teal")
    else:
        axes[0].scatter(df_clean["Daily_Return"], df_clean["Volume"], alpha=0.6, color="purple")
        axes[1].hist(df_clean["Daily_Return"], bins=40, color="teal", edgecolor="black")
    axes[0].set_title("Scatter Plot: Daily Return vs Volume")
    axes[1].set_title("Histogram: Daily Returns Distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "03_scatter_and_histogram.png", dpi=200)
    plt.close()
    
    print(f"[+] Figures saved to: {FIGURES_PATH}")
    return df_clean

if __name__ == "__main__":
    run_eda_pipeline()
