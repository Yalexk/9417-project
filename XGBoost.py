import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


RANDOM_STATE = 42

BASE_FEATURE_COLS = [
    "CO(GT)",
    "NMHC(GT)", 
    "C6H6(GT)",
    "NOx(GT)",
    "NO2(GT)",
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]

TARGET_COLS = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
HORIZONS = [1, 6, 12, 24]


# feature engineering funcs

def add_time_features(
    df,
    base_cols,
    lags=(1, 2, 3, 6, 12, 24),
    roll_windows=(3, 6, 12, 24)
):
    df = df.copy().sort_values("Timestamp")
    new_cols = {} 
    for col in base_cols:
        if col not in df.columns:
            continue

        s = df[col]

        for h in lags:
            new_cols[f"{col}_lag{h}h"] = s.shift(h)

        for w in roll_windows:
            shifted = s.shift(1)
            new_cols[f"{col}_roll{w}h_mean"] = shifted.rolling(w).mean()
            new_cols[f"{col}_roll{w}h_std"]  = shifted.rolling(w).std()

    df_feat = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df_feat


def add_target_horizons(df, target_cols, horizons):
    df = df.copy()
    for col in target_cols:
        for h in horizons:
            df[f"{col}_tplus{h}"] = df[col].shift(-h)
    return df


# model and evaluation functions

def make_gbm():
    return XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        objective="reg:squarederror",
        reg_lambda=10.0,
        min_child_weight=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )

def get_test_data(df_full, pollutant, H, feature_cols):
    target_col = f"{pollutant}_tplus{H}"
    cols_needed = feature_cols + [target_col, "Timestamp", pollutant] 
    df_task = df_full.dropna(subset=[target_col]).loc[:, cols_needed].copy()
    test_df = df_task[df_task["Timestamp"].dt.year == 2005].copy()
    
    X_test = test_df[feature_cols].values
    y_true = test_df[target_col].values
    timestamps = test_df["Timestamp"].values
    baseline_test = test_df[pollutant].values
    
    return X_test, y_true, timestamps, baseline_test


def plot_single_forecast(
    timestamps, y_true, y_pred, baseline_pred, pollutant, H, save_dir=None, show=True
):
    """Plots the actual, XGBoost prediction, and naive baseline for a forecast."""
    plt.figure(figsize=(12, 4))

    plt.plot(timestamps, y_true, label="Actual", alpha=0.7)
    plt.plot(timestamps, y_pred, label="Predicted (XGB)", alpha=0.7)
    plt.plot(timestamps, baseline_pred, label="Baseline (Naive)", linestyle='--', alpha=0.7)
    
    plt.legend()
    plt.title(f"Forecast for {pollutant} at +{H} hours")
    plt.xlabel("Timestamp")
    plt.ylabel(pollutant)
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{pollutant}_tplus{H}.png"))
    
    if show:
        plt.show()
    else:
        plt.close()


# helper to generate heatmap
def plot_improvement_heatmap(results_df):
    """Plots a heatmap of the RMSE improvement of XGB over the naive baseline."""
    pivot = results_df.pivot_table(
        index="pollutant",
        columns="horizon_h",
        values=["rmse_baseline_test", "rmse_gbm_test"]
    )

    base = pivot["rmse_baseline_test"]
    gbm = pivot["rmse_gbm_test"]

    improvement = (base - gbm) / base * 100

    plt.figure(figsize=(6, 4))
    im = plt.imshow(improvement.values, aspect="auto", cmap='viridis')
    plt.colorbar(im, label="% RMSE improvement (baseline → XGB)")
    plt.xticks(range(len(improvement.columns)), improvement.columns)
    plt.yticks(range(len(improvement.index)), improvement.index)
    plt.xlabel("Horizon (hours)")
    plt.ylabel("Pollutant")
    plt.title("XGB vs baseline: % RMSE improvement")
    plt.tight_layout()
    plt.show()

# additional helper to generate plots for evaluation
def generate_all_plots(df_full, feature_cols, models, target_cols, horizons, save_dir="xgb_plots", show=False, n_points=None):
    print(f"Generating plots in directory: {save_dir}")
    for pollutant in target_cols:
        for H in horizons:
            print(f"Plots for {pollutant} +{H}h...")
            
            X_test, y_true, timestamps, baseline_pred = get_test_data(df_full, pollutant, H, feature_cols)
            
            model = models[(pollutant, H)]
            y_pred = model.predict(X_test)
            
            if n_points:
                y_true, timestamps, baseline_pred, y_pred = \
                    y_true[:n_points], timestamps[:n_points], baseline_pred[:n_points], y_pred[:n_points]
            
            plot_single_forecast(
                timestamps, 
                y_true, 
                y_pred, 
                baseline_pred,
                pollutant, 
                H, 
                save_dir=save_dir, 
                show=show
            )
    print("All plots generated.")



# 1. data Loading and cleaning
try:
    df = pd.read_csv("air+quality/AirQualityUCI_knn_imputed.csv")
except FileNotFoundError:
    print("file not found")
    exit()

df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df = df.dropna(subset=["Timestamp"]).reset_index(drop=True)

print("--- Initial Data Year Counts ---")
print(df["Timestamp"].dt.year.value_counts())
print("-" * 30)

# 2. add time-based Features
df_feat = add_time_features(df, BASE_FEATURE_COLS)

time_feature_cols = [
    c for c in df_feat.columns
    if any(tag in c for tag in ["_lag", "_roll"])
]

df_feat_clean = df_feat.dropna(subset=time_feature_cols).reset_index(drop=True)

# 3. add Target Horizons
df_full = add_target_horizons(df_feat_clean, TARGET_COLS, HORIZONS)

print("--- Data Year Counts after Feature/Target Cleaning ---")
print(df_full["Timestamp"].dt.year.value_counts())
print("-" * 30)


FEATURE_COLS = [
    c for c in df_full.columns
    if c != "Timestamp"
    and not any(c.endswith(f"_tplus{h}") for h in HORIZONS)
]

print("Num features:", len(FEATURE_COLS))
print("-" * 30)


# model Training and Evaluation Loop
results = []
models = {}

print("Model Training and Evaluation")
for pollutant in TARGET_COLS:
    for H in HORIZONS:
        target_col = f"{pollutant}_tplus{H}"

        cols_needed = FEATURE_COLS + [target_col, "Timestamp"]
        df_task = df_full.dropna(subset=[target_col]).loc[:, cols_needed].copy()

        years = df_task["Timestamp"].dt.year

        train_df = df_task[years == 2004]
        test_df  = df_task[years == 2005]

        print(f"{pollutant} +{H}h: ")

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df[target_col].values

        X_test = test_df[FEATURE_COLS].values
        y_test = test_df[target_col].values

        baseline_train = train_df[pollutant].values
        baseline_test  = test_df[pollutant].values

        rmse_baseline_train = float(np.sqrt(mean_squared_error(y_train, baseline_train)))
        rmse_baseline_test  = float(np.sqrt(mean_squared_error(y_test,  baseline_test)))

        model = make_gbm()
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)

        rmse_gbm_train = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
        rmse_gbm_test  = float(np.sqrt(mean_squared_error(y_test,  y_pred_test)))

        results.append({
            "pollutant": pollutant,
            "horizon_h": H,
            "rmse_baseline_train": rmse_baseline_train,
            "rmse_baseline_test":  rmse_baseline_test,
            "rmse_gbm_train":      rmse_gbm_train,
            "rmse_gbm_test":       rmse_gbm_test,
        })

        models[(pollutant, H)] = model

        print(
            f"Done: {pollutant}, +{H}h | "
            f"Test RMSE baseline={rmse_baseline_test:.3f}, "
            f"GBM={rmse_gbm_test:.3f}"
        )

results_df = pd.DataFrame(results)

print("\nResults DataFrame (RMSE Comparison)")
print(results_df)
print("-" * 30)


print("Generating Plots")

plot_improvement_heatmap(results_df)
