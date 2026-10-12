1. Project Overview

Predicting equity price dynamics is a foundational challenge in quantitative finance. This project implements and benchmarks multiple predictive analytics techniques directly aligned with the **CSE3141 Predictive Analytics** course handout.

The framework formulates two complementary predictive tasks on financial market data:
1. **Time-Series Price Forecasting:** Modeling continuous price trajectories using statistical decomposition, **Exponential Smoothing (Holt-Winters)**, **ARIMA**, and **SARIMA** models.
2. **Directional Trend Classification:** Predicting binary market movements ($\text{Target} = 1$ if $\text{Price}_{t+1} > \text{Price}_t$, else $0$) by benchmarking **Binary Logistic Regression**, **Linear Discriminant Analysis (LDA)**, and **Quadratic Discriminant Analysis (QDA)** with hyperparameter optimization.

---

2. Syllabus & Handout Mapping

Every component implemented in this repository strictly adheres to the official lecture plan:

| Course Phase | Syllabus Lecture Topics | Implemented Techniques | Source File |
| :--- | :--- | :--- | :--- |
| **Data Exploration & Preprocessing** | Lectures 6 – 11 | • IQR Box Plot & Z-Score Outlier Analysis<br>• Mean, Median, Mode, Variance, Std Deviation<br>• Heatmaps, Histograms & Scatter plots | `eda_preprocessing.py` |
| **Classification & Inference** | Lectures 12 – 17 | • Binary Logistic Regression & Coefficient Estimation<br>• Linear Discriminant Analysis (LDA)<br>• Quadratic Discriminant Analysis (QDA) | `trend_classification.py` |
| **Time Series Forecasting** | Lectures 23 – 29 | • Trend, Seasonality & Noise Decomposition<br>• Exponential Smoothing (Holt-Winters)<br>• ARIMA(p,d,q) & SARIMA(p,d,q)(P,D,Q)s | `time_series_forecasting.py` |
| **Validation & Optimization** | Lectures 30 – 36 | • Chronological Train-Test Split<br>• Grid Search CV Hyperparameter Tuning<br>• RMSE, MAE, F1-Score, AUC-ROC | All Modules |

---
 4. Quickstart Guide

### Prerequisites
Make sure Python 3.9+ is installed.

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/Stock-Trend-Predictive-Analysis-CSE3141.git
cd Stock-Trend-Predictive-Analysis-CSE3141
```

### Step 2: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 3: Run the Full End-to-End Pipeline
Execute the master runner to run all analyses, train models, and generate comparison graphs:
```bash
python run_pipeline.py
```

You can also run individual modules independently:
```bash
python eda_preprocessing.py
python time_series_forecasting.py
python trend_classification.py
```

---

## 📊 5. Evaluation Benchmarks

### A. Time Series Price Forecasting Models *(Lectures 31-32)*
| Model | RMSE | MAE | Handout Reference |
| :--- | :---: | :---: | :--- |
| **Exponential Smoothing (Holt)** | 5.21 | 4.12 | Lecture 27 |
| **ARIMA (1, 1, 1)** | 4.88 | 3.76 | Lecture 25 |
| **SARIMA (1, 1, 1)(1, 0, 1)₅** | **4.42** | **3.31** | Lecture 26 |

### B. Market Trend Classification Models *(Lectures 33-34)*
| Model | Accuracy (%) | F1-Score | AUC-ROC | Handout Reference |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression (Tuned)** | 55.45% | 0.548 | 0.572 | Lectures 12–14, 35 |
| **Linear Discriminant Analysis (LDA)** | 56.12% | 0.553 | 0.581 | Lectures 15–16 |
| **Quadratic Discriminant Analysis (QDA)** | **57.30%** | **0.569** | **0.594** | Lecture 17 |

---

## 🔗 6. Dataset Citation & References
- **Dataset Source:** Historical S&P 500 / Equity Price Data via [Kaggle S&P 500 Stock Data](https://www.kaggle.com/datasets/camnugent/sandp500)
- **Primary Reference Textbook:** *Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani, "An Introduction to Statistical Learning with Applications in R" (ISLR), Springer, 2017.*
