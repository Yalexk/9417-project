import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# Suppress warnings
# logistic regression unable to converge regardless of solver/params
warnings.filterwarnings("ignore")

# Hyperparameters
SVM_C = 100         
RANDOM_STATE = 42
HORIZONS = [1, 6, 12, 24]

# Load data
train_df = pd.read_csv('air+quality/AirQualityUCI_standard_scaled_v2_train.csv')
test_df = pd.read_csv('air+quality/AirQualityUCI_standard_scaled_v2_test.csv')

train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])

raw_target_cols = [c for c in train_df.columns if c.endswith('_raw')]

lr_results = {}
svm_results = {}
baseline_results = {}

for h in HORIZONS:
    target_col = f'CO_class_t+{h}'
    
    # create shifted targets
    train_df[target_col] = train_df['CO_class'].shift(-h)
    test_df[target_col] = test_df['CO_class'].shift(-h)
    
    # Filter out NaN targets BEFORE creating X_train, y_train
    train_valid = train_df.dropna(subset=[target_col])
    test_valid = test_df.dropna(subset=[target_col])
    
    # exclude timestamp, discrete class, raw target values, future targets
    cols_to_exclude = ['Timestamp', 'CO_class'] + raw_target_cols + [c for c in train_df.columns if 'CO_class_t+' in c]
    feature_cols = [c for c in train_df.columns if c not in cols_to_exclude]
    
    X_train = train_valid[feature_cols]
    y_train = train_valid[target_col]
    
    X_test = test_valid[feature_cols]
    y_test = test_valid[target_col]
    
    # naive baseline
    y_pred_naive = test_valid['CO_class']
    naive_acc = accuracy_score(y_test, y_pred_naive)
    baseline_results[f"t+{h}"] = naive_acc
    
    # logistic regression
    lr = LogisticRegression(solver='lbfgs', max_iter=1000, C=1)
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    lr_results[f"t+{h}"] = lr_acc

    # svm
    svm = SVC(kernel='rbf', C=SVM_C, gamma='scale', random_state=RANDOM_STATE)
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))
    svm_results[f"t+{h}"] = svm_acc

    print(f"t+{h} done")

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