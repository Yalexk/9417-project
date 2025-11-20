import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import os

# make directory for plots
plot_dir = "rf_regression_plots"
os.makedirs(plot_dir, exist_ok=True)

df = pd.read_csv("AirQualityUCI_with_timestamp.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

pollutants = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
horizons = [1, 6, 12, 24]

# interpolate missing pollutant values
df[pollutants] = df[pollutants].interpolate().bfill().ffill()

# create lag features and future targets
for pollutant in pollutants:
    for h in horizons:
        df[f"{pollutant}_lag_{h}"] = df[pollutant].shift(h)
        df[f"{pollutant}_future_{h}"] = df[pollutant].shift(-h)

# drop rows that don't have full lag/future data
required_cols = []
for pollutant in pollutants:
    for h in horizons:
        required_cols += [
            f"{pollutant}_lag_{h}",
            f"{pollutant}_future_{h}",
        ]

df = df.dropna(subset=required_cols).reset_index(drop=True)

train = df[df["Timestamp"].dt.year == 2004].copy()
test  = df[df["Timestamp"].dt.year == 2005].copy()

future_cols = [f"{p}_future_{h}" for p in pollutants for h in horizons]

feature_cols = [
    col for col in df.columns
    if col not in ["Timestamp"] and col not in future_cols
]

X_train = train[feature_cols]
X_test = test[feature_cols]

# fill any remaining nans in features
X_train = X_train.ffill().bfill().fillna(0)
X_test = X_test.ffill().bfill().fillna(0)

for p in pollutants:
    test[p] = test[p].ffill().bfill().fillna(0)


# train ranbdom forest for each pollutant + horizon
results = {}

for pollutant in pollutants:
    results[pollutant] = {}

    for h in horizons:
        y_train = train[f"{pollutant}_future_{h}"]
        y_test = test[f"{pollutant}_future_{h}"]

        # n_estimators = 300  stable predictions
        # random_state = 42  for consistent reseults
        # n_jobs = -1  use all CPU cores
        rf = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        preds = rf.predict(X_test)

        ## rmse for model and naive baseline
        rmse = mean_squared_error(y_test, preds) ** 0.5
        baseline_rmse = mean_squared_error(y_test, test[pollutant]) ** 0.5

        results[pollutant][h] = (rmse, baseline_rmse, preds, y_test, rf)

# print summary table of RMSEs
print("\nForecast Results\n")
print(f"{'Pollutant':<12} {'Horizon':<8} {'RF_RMSE':<10} {'Baseline':<10}")
print("-" * 45)

for pollutant in pollutants:
    for h in horizons:
        rmse, baseline_rmse, _, _, _ = results[pollutant][h]
        print(f"{pollutant:<12} {str(h)+'h':<8} {rmse:<10.4f} {baseline_rmse:<10.4f}")
    print()

# save prediction plots 
for pollutant in pollutants:
    for h in horizons:
        rmse, base_rmse, preds, y_test, _ = results[pollutant][h]

        plt.figure(figsize=(12,4))
        #first 300 for readability
        plt.plot(y_test.values[:300], label="Actual")
        plt.plot(preds[:300], label="Predicted")
        plt.title(f"{pollutant} — {h}-Hour Forecast")
        plt.xlabel("Index")
        plt.ylabel(pollutant)
        plt.legend()

        plt.savefig(
            os.path.join(plot_dir, f"{pollutant}_{h}h_forecast.png"),

        )
        plt.close()

# print top 15 feature importances for 1 hour models
print("\nFeature importances (1 hour horizon):")
for pollutant in pollutants:
    rf = results[pollutant][1][4]
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    print(f"\n{pollutant}:")
    print(importances.sort_values(ascending=False).head(15))
