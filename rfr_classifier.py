import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Hyperparameters
N_ESTIMATORS=300 # number of trees
RANDOM_STATE=42 # randomness
N_JOBS=-1 # number of cores -1 is maximum

# Load data
df = pd.read_csv("./air+quality/AirQualityUCI_imputed_robust_scaled.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Discretise CO(GT)
def discretise_co(value):
    if pd.isna(value):  # Add this check
        return np.nan
    if value < 1.5:
        return "low"
    elif value < 2.5:
        return "mid"
    else:
        return "high"

df["CO_class"] = df["CO(GT)"].apply(discretise_co)

# Creat shifted targets for prediction
shift_hours = [1, 6, 12, 24]
for k in shift_hours:
    df[f"CO_class_t+{k}"] = df["CO_class"].shift(-k)

# Split 2004 → train, 2005 → test
train_df = df[df["Timestamp"].dt.year == 2004]
test_df  = df[df["Timestamp"].dt.year == 2005]

# Feature Columns
feature_cols = [col for col in df.columns if col not in ["CO_class", "Timestamp"] + [f"CO_class_t+{k}" for k in shift_hours]]

# Filter out NaN targets BEFORE creating X_train, y_train
train_valid = train_df.dropna(subset=["CO_class"])
test_valid = test_df.dropna(subset=["CO_class"])

X_train = train_valid[feature_cols]
y_train = train_valid["CO_class"]

X_test = test_valid[feature_cols]
y_test = test_valid["CO_class"]

# Random forest classifier for the current time
rf_current = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=N_JOBS)
rf_current.fit(X_train, y_train)

y_pred_current = rf_current.predict(X_test)
current_acc = accuracy_score(y_test, y_pred_current)
print("=== Random Forest Results ===")
print(f"t+0 (current): {current_acc:.4f}")

# random forest classifier for future time steps
rf_results = {"t+0": current_acc}

for k in shift_hours:
    target_col = f"CO_class_t+{k}"
    
    # Filter out rows where future target is missing eg. near the end
    train_valid = train_df.dropna(subset=[target_col])
    test_valid = test_df.dropna(subset=[target_col])
    
    if len(train_valid) > 0 and len(test_valid) > 0:
        X_train_k = train_valid[feature_cols]
        y_train_k = train_valid[target_col]
        X_test_k = test_valid[feature_cols]
        y_test_k = test_valid[target_col]
        
        # Train model for this prediction horizon
        rf_k = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=N_JOBS)
        rf_k.fit(X_train_k, y_train_k)
        
        # Predict
        y_pred_k = rf_k.predict(X_test_k)
        acc_k = accuracy_score(y_test_k, y_pred_k)
        
        rf_results[f"t+{k}"] = acc_k
        print(f"t+{k}: {acc_k:.4f}")

# Naive Baseline
baseline_results = {}

for k in shift_hours:
    test_temp = test_df.copy()
    test_temp["target_shifted"] = test_temp["CO_class"].shift(-k)
    test_temp = test_temp.dropna(subset=["target_shifted", "CO_class"])
    
    if len(test_temp) > 0:
        acc = accuracy_score(test_temp["target_shifted"], test_temp["CO_class"])
        baseline_results[f"t+{k}"] = acc

print("\n=== Naive Baseline Results ===")
for k, v in baseline_results.items():
    print(f"t+{k}: {v:.4f}")

# Comparison 
print("\n=== Comparison ===")
print("Horizon | Random Forest | Naive Baseline | Improvement")
print("--------|---------------|----------------|------------")
for k in [f"t+{h}" for h in shift_hours]:
    rf_acc = rf_results.get(k, np.nan)
    naive_acc = baseline_results.get(k, np.nan)
    improvement = rf_acc - naive_acc if not np.isnan(rf_acc) and not np.isnan(naive_acc) else np.nan
    print(f"{k:7} | {rf_acc:13.4f} | {naive_acc:14.4f} | {improvement:+11.4f}")