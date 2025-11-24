# 9417-project

**Preprocessing**

Implementation: ./preprocessing.py

Output: ./air+quality/*

**Anomaly Results**

./anomaly_results

**Regression**

Random Forest: ./rfr_regressor.py

**Classification**

Random Forest: ./rfr_classifier.py

SVM & Logistic Regression: ./svm_logistic_classifier.py


Air Quality Time Series Forecasting (XGBoost)

This document provides a clear guide to setting up and running the regression.py script.
The script trains multiple XGBoost regression models to forecast air pollutant concentrations across different time horizons.

1. Prerequisites
1.1 Dependencies

You need Python 3.7+.

Install the required libraries:
```
pip install pandas numpy scikit-learn xgboost matplotlib
```
1.2 Data Setup

The script requires the pre-processed Air Quality UCI dataset.

File Name: AirQualityUCI_knn_imputed.csv

Required Path: Place the file in a folder named air+quality/ in the project root.

2. Execution

Run the script from your terminal:

```python regression.py```

What the Script Does

The script performs multi-step forecasting for 5 pollutants across 4 time horizons (1h, 6h, 12h, 24h) — a total of 20 models.

Preparation

Loads the dataset

Creates lag, rolling mean, and rolling std features

Defines target columns (e.g., CO(GT)_tplus6)

Training

Trains 20 XGBoost models using data from 2004

Evaluation

Tests all models on data from 2005

Compares XGBoost RMSE against a Naïve Baseline

Output

Prints a detailed RMSE results table to the console

Displays a heatmap showing percent RMSE improvement over baseline

Saves 20 forecast plots (Actual vs XGBoost vs Baseline) as PNG files in:

xgb_plots/
