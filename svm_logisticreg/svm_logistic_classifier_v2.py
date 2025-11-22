import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
import matplotlib.pyplot as plt

# Suppress warnings
# logistic regression unable to converge regardless of solver/params
warnings.filterwarnings("ignore")

# Hyperparameters
SVM_C_CANDIDATES = [0.1, 1, 10, 100, 1000]
RANDOM_STATE = 42
HORIZONS = [1, 6, 12, 24]

# Load data
train_df = pd.read_csv('AirQualityUCI_standard_scaled_v2_train.csv')
test_df = pd.read_csv('AirQualityUCI_standard_scaled_v2_test.csv')

train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])

raw_target_cols = [c for c in train_df.columns if c.endswith('_raw')]

lr_results = {}
svm_results = {}
baseline_results = {}
svm_best_c = {}

results = []

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

    # split test (2004) data to find best C value for SVM
    train_split = int(len(train_valid) * 0.8)
    x_SVM_train = X_train.iloc[:train_split]
    y_SVM_train = y_train.iloc[:train_split]
    x_SVM_test = X_train.iloc[train_split:]
    y_SVM_test = y_train.iloc[train_split:]

    best_C = None
    best_acc = -np.inf

    # find best C value that gives highest accuracy within 2004 (80/20 split)
    for c_value in SVM_C_CANDIDATES:
        svm = SVC(kernel='rbf', C=c_value, gamma='scale', random_state=RANDOM_STATE)
        svm.fit(x_SVM_train, y_SVM_train)
        curr_acc = accuracy_score(y_SVM_test, svm.predict(x_SVM_test))
        if curr_acc > best_acc:
            best_acc = curr_acc
            best_c = c_value
    
    # using best C value, set SVM accuracy
    svm_best = SVC(kernel='rbf', C=best_c, gamma='scale', random_state=RANDOM_STATE)
    svm_best.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm_best.predict(X_test))
    svm_results[f"t+{h}"] = svm_acc
    svm_best_c[f"t+{h}"] = best_c

    print(f"t+{h}: | Naive={naive_acc:.4f} | LR={lr_acc:.4f} | SVM(C={best_c})={svm_acc:.4f}")

    results.append({
        'Horizon': f't+{h}',
        'Naive': naive_acc,
        'LogisticRegression': lr_acc,
        'SVM': svm_acc,
        'SVM_best_c': best_c,
    })

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
print("Horizon | SVM       | SVM Best C | Naive Baseline | Improvement")
print("--------|-----------|------------|----------------|------------")
for k in [f"t+{h}" for h in HORIZONS]:
    naive = baseline_results.get(k, np.nan)
    svm = svm_results.get(k, np.nan)
    improvement = svm - naive if not np.isnan(svm) and not np.isnan(naive) else np.nan
    print(f"{k:7} | {svm:9.4f} | {best_c:10} | {naive:14.4f} | {improvement:+11.4f}")

df_results = pd.DataFrame(results)

# generate comparison bar graph
horizons = df_results['Horizon']
x = np.arange(len(horizons))
width = 0.25  

fig, ax = plt.subplots(figsize=(10, 6))

naive_bars = ax.bar(x - width, df_results['Naive'], width, label='Naive Baseline', color='#828282')
LR_bars = ax.bar(x, df_results['LogisticRegression'], width, label='Logistic Regression', color='#0041C2')
SVM_bars = ax.bar(x + width, df_results['SVM'], width, label='SVM (RBF, best C)', color='#008000')

# Formatting
ax.set_ylabel('Accuracy')
ax.set_title('Classification Model Comparison - Naive/Logistic Regression/SVM')
ax.set_xticks(x)
ax.set_xticklabels(horizons)
ax.legend(loc='lower right')
ax.grid(axis='y', linestyle='--', alpha=0.6)

# set bottom of graph
lowest_score = df_results[['Naive', 'LogisticRegression', 'SVM']].min().min()
bottom = max(0, lowest_score - 0.1)
ax.set_ylim(bottom=bottom, top=1.0)

# add accuracy on top of each bar
def label(models):
    for model in models:
        height = model.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(model.get_x() + model.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

label(naive_bars)
label(LR_bars)
label(SVM_bars)

plt.tight_layout()
plt.savefig('svm_logistic_comparison_v2.png', dpi=300)