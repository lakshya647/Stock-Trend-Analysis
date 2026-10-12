"""
Module: Trend Classification & Inference (Lectures 12 - 17, 30, 33 - 36)
Implements:
1. Logistic Regression for Binary Classification (Lecture 12-14)
2. Linear Discriminant Analysis - LDA (Lecture 15-16)
3. Quadratic Discriminant Analysis - QDA (Lecture 17)
4. Model Validation & Hyperparameter Tuning via GridSearchCV (Lectures 30, 35)
5. Performance Metrics: F1-Score, AUC-ROC, Confusion Matrix (Lectures 33-34)
Reads directly from stock_features_processed.csv in root repository.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "stock_features_processed.csv"
FIGURES_PATH = PROJECT_ROOT / "figures"

def run_classification_pipeline():
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    
    if not DATA_PATH.exists():
        from eda_preprocessing import run_eda_pipeline
        run_eda_pipeline()
        
    df = pd.read_csv(DATA_PATH)
    
    # Feature matrix X and Target y
    feature_cols = ["Open", "High", "Low", "Close", "Volume", "SMA_10", "SMA_50", "RSI_14", "Volatility_10", "Daily_Return"]
    X = df[feature_cols]
    y = df["Target_Direction"]
    
    # Train-test split (chronological split preserving time dependency)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("=" * 60)
    print("STEP 1: LOGISTIC REGRESSION & ESTIMATION OF COEFFICIENTS (Lectures 12-14)")
    print("=" * 60)
    # Hyperparameter tuning using GridSearchCV (Lecture 35)
    param_grid_lr = {"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l2"]}
    grid_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid_lr, cv=5, scoring="f1")
    grid_lr.fit(X_train_scaled, y_train)
    best_lr = grid_lr.best_estimator_
    
    print(f"Optimal Logistic Regression Regularization (C): {grid_lr.best_params_['C']}")
    print("\nEstimated Logistic Regression Coefficients (Feature Weights):")
    coef_df = pd.DataFrame({"Feature": feature_cols, "Coefficient": best_lr.coef_[0]}).sort_values("Coefficient", ascending=False)
    print(coef_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("STEP 2: DISCRIMINANT ANALYSIS: LDA & QDA (Lectures 15-17)")
    print("=" * 60)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_scaled, y_train)
    
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train_scaled, y_train)
    
    models = {
        "Logistic Regression": best_lr,
        "Linear Discriminant Analysis (LDA)": lda,
        "Quadratic Discriminant Analysis (QDA)": qda
    }
    
    results = []
    plt.figure(figsize=(9, 7))

    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            "Model": name,
            "Accuracy (%)": round(acc * 100, 2),
            "F1-Score": round(f1, 4),
            "AUC-ROC": round(auc, 4)
        })
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
        
        # Save confusion matrix (Lecture 33)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax_cm, cmap="Blues", values_format="d")
        ax_cm.set_title(f"Confusion Matrix: {name}")
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fig_cm.savefig(FIGURES_PATH / f"cm_{clean_name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig_cm)

    # ROC Curve Plot (Lecture 34)
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)")
    plt.title("Comparative ROC Curves for Stock Trend Prediction (Lecture 34)", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "06_roc_curve_comparison.png", dpi=200)
    plt.close()

    summary_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("CLASSIFICATION BENCHMARK COMPARISON TABLE (Lectures 33-34)")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\n[+] ROC curves and Confusion Matrices saved in: {FIGURES_PATH}")
    
    return summary_df

if __name__ == "__main__":
    run_classification_pipeline()
