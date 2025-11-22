import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# Suppress warnings 
# logistic regression unable to converge regardless of solver/params
warnings.filterwarnings("ignore")

HORIZONS = [1, 6, 12, 24]

# Hyperparameters
C = 1         
RANDOM_STATE = 42

# Load data
df = pd.read_csv("air+quality/AirQualityUCI_standard_scaled.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Create shifted targets
for k in HORIZONS:
    df[f"CO_class_t+{k}"] = df["CO_class"].shift(-k)

# Split 2004 -> train, 2005 -> test
train_df = df[df["Timestamp"].dt.year == 2004].copy()
test_df  = df[df["Timestamp"].dt.year == 2005].copy()

# Feature Columns 
feature_cols = [col for col in df.columns if col not in ["CO_class", "Timestamp"] + [f"CO_class_t+{k}" for k in HORIZONS]]

lr_results = {}
svm_results = {}
baseline_results = {}

for k in HORIZONS:
    target_col = f"CO_class_t+{k}"
    
    # Filter out NaN targets (Teammate's logic)
    train_valid = train_df.dropna(subset=[target_col])
    test_valid = test_df.dropna(subset=[target_col])

    if len(train_valid) > 0 and len(test_valid) > 0:
        X_train = train_valid[feature_cols]
        y_train = train_valid[target_col]
        
        X_test = test_valid[feature_cols]
        y_test = test_valid[target_col]
        
        # naive baseline
        naive_acc = accuracy_score(test_valid[target_col], test_valid["CO_class"])
        baseline_results[f"t+{k}"] = naive_acc
        
        # logistic regression
        lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(X_train, y_train)
        lr_acc = accuracy_score(y_test, lr.predict(X_test))
        lr_results[f"t+{k}"] = lr_acc
        
        # SVM
        svm = SVC(kernel='rbf', C=C, gamma='scale', random_state=RANDOM_STATE)
        svm.fit(X_train, y_train)
        svm_acc = accuracy_score(y_test, svm.predict(X_test))
        svm_results[f"t+{k}"] = svm_acc
        
        print(f"t+{k} done")

# Logistic Regression Comparison
print("\n=== Logistic Regression Comparison ===")
print("Horizon | Logistic Reg |  Naive Baseline | Improvement")
print("--------|--------------|-----------------|------------")
for k in [f"t+{h}" for h in HORIZONS]:
    naive = baseline_results.get(k, np.nan)
    lr = lr_results.get(k, np.nan)
    improvement = lr - naive if not np.isnan(lr) and not np.isnan(naive) else np.nan
    print(f"{k:7} | {lr:13.4f} | {naive:14.4f} | {improvement:+11.4f}")

# SVM Comparison
print("\n=== SVM Comparison ===")
print("Horizon | SVM           | Naive Baseline | Improvement")
print("--------|---------------|----------------|------------")
for k in [f"t+{h}" for h in HORIZONS]:
    naive = baseline_results.get(k, np.nan)
    svm = svm_results.get(k, np.nan)
    improvement = svm - naive if not np.isnan(svm) and not np.isnan(naive) else np.nan
    print(f"{k:7} | {svm:13.4f} | {naive:14.4f} | {improvement:+11.4f}")