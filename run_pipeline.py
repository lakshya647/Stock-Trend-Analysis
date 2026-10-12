"""
Master Pipeline Runner: Executes end-to-end Predictive Analytics Project.
Runs all components in a flat repository structure:
1. Data Ingestion & Download (download_data.py)
2. EDA, Statistical Summary & Outlier Detection (eda_preprocessing.py)
3. Time Series Forecasting (time_series_forecasting.py)
4. Classification & Discriminant Analysis (trend_classification.py)
"""

import sys
from pathlib import Path

# Add current root directory to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Direct imports (flat directory structure)
from download_data import main as download_main
from eda_preprocessing import run_eda_pipeline
from time_series_forecasting import run_time_series_pipeline
from trend_classification import run_classification_pipeline

def main():
    print("=" * 70)
    print("CSE3141: PREDICTIVE ANALYTICS MINI-PROJECT PIPELINE")
    print("Topic: Comparative Analysis of Machine Learning and Time-Series Techniques")
    print("=" * 70)
    
    print("\n>>> Phase 1: Acquiring Market Data...")
    download_main()
    
    print("\n>>> Phase 2: Running EDA, Preprocessing & Feature Engineering...")
    run_eda_pipeline()
    
    print("\n>>> Phase 3: Executing Time-Series Forecasting Models...")
    ts_results = run_time_series_pipeline()
    
    print("\n>>> Phase 4: Executing Trend Classification Models...")
    clf_results = run_classification_pipeline()
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY REPORT")
    print("=" * 70)
    print("\n[Time Series Forecasting Evaluation (Lectures 31-32)]")
    print(ts_results.to_string(index=False))
    
    print("\n[Market Trend Classification Evaluation (Lectures 33-34)]")
    print(clf_results.to_string(index=False))
    
    print("\n[+] All evaluations complete. Visualizations generated in figures/ directory.")

if __name__ == "__main__":
    main()
