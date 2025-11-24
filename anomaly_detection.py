import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import os

# basic config 
INPUT_CSV = "./air+quality/AirQualityUCI_with_timestamp.csv"
OUT_DIR = "anomaly_results"
POLLUTANTS = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)']

PREDICTORS = [
    'T','RH','AH','Hour','IsWeekend','Hour_sin','Hour_cos','Month_sin','Month_cos',
    'Weekday_sin','Weekday_cos',
    'PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)'
]

TRAIN_FRACTION = 0.7
TOP_ANOMS = 50

os.makedirs(OUT_DIR, exist_ok=True)

# load and align timestamps
df = pd.read_csv(INPUT_CSV, engine='python')
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.set_index('Timestamp').sort_index()

pollutants = [p for p in POLLUTANTS if p in df.columns]
predictors = [col for col in PREDICTORS if col in df.columns]

results = []
all_rows = []

for pollutant in pollutants:
    # filter rows with target available
    tmp = df.dropna(subset=[pollutant])
    if len(tmp) < 50:
        continue

    X = tmp[predictors].fillna(tmp[predictors].median())
    y = tmp[pollutant]

    # basic train/test split based on time
    split = int(TRAIN_FRACTION * len(X))
    X_train, y_train = X.iloc[:split], y.iloc[:split]

    # train simple RF model for residual-based anomaly detection
    rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = pd.Series(rf.predict(X), index=X.index)
    residuals = y - preds

    # MAD threshold
    mad = np.median(np.abs(residuals - residuals.median()))
    thr = 3 * mad
    res_flag = residuals.abs() > thr

    # isolation forest as second detector
    iso = IsolationForest(contamination=0.02, random_state=0)
    iso.fit(X_train)
    iso_score = iso.decision_function(X)
    iso_flag = iso.predict(X) == -1

    # rough label for evaluating detectors (top 1% true values)
    spike_thr = y.quantile(0.99)
    spikes = y > spike_thr

    # metrics helper
    def safe_metrics(pred, true):
        if true.sum() == 0:
            return np.nan, np.nan
        return precision_score(true, pred), recall_score(true, pred)

    # evaluate residual vs spike labels
    pr_res, rc_res = safe_metrics(res_flag.astype(int), spikes.astype(int))
    pr_iso, rc_iso = safe_metrics(iso_flag.astype(int), spikes.astype(int))

    ap_res = average_precision_score(spikes.astype(int), residuals.abs()) if spikes.sum() else np.nan
    ap_iso = average_precision_score(spikes.astype(int), -iso_score) if spikes.sum() else np.nan

    # build summary table for exporting
    summary = pd.DataFrame({
        'pollutant': pollutant,
        'value': y,
        'pred': preds,
        'residual': residuals,
        'residual_anom': res_flag,
        'iso_anom': iso_flag,
        'spike_label': spikes,
        'T': X['T'] if 'T' in X.columns else np.nan,
        'RH': X['RH'] if 'RH' in X.columns else np.nan
    }, index=X.index)

    # top anomalies (union of detectors)
    top = summary[(summary['residual_anom']) | (summary['iso_anom'])]
    top = top.sort_values('residual', ascending=False).head(TOP_ANOMS)
    top = top.reset_index().rename(columns={'index': 'Timestamp'})
    all_rows.append(top)
    top.to_csv(os.path.join(OUT_DIR, f"top_anomalies_{pollutant.replace('/','_')}.csv"), index=False)

    # record performance
    results.append({
        'pollutant': pollutant,
        'n_rows': len(y),
        'spike_threshold': float(spike_thr),
        'mad_threshold': float(thr),
        'res_precision': None if np.isnan(pr_res) else float(pr_res),
        'res_recall': None if np.isnan(rc_res) else float(rc_res),
        'res_avg_precision': None if np.isnan(ap_res) else float(ap_res),
        'iso_precision': None if np.isnan(pr_iso) else float(pr_iso),
        'iso_recall': None if np.isnan(rc_iso) else float(rc_iso),
        'iso_avg_precision': None if np.isnan(ap_iso) else float(ap_iso)
    })

    # plotting
    plt.figure(figsize=(12,4))
    plt.plot(y.index, y.values)
    plt.scatter(y.index[res_flag], y[res_flag])
    plt.title(f"{pollutant}: detected anomalies")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{pollutant}_timeseries_anoms.png"))
    plt.close()

    plt.figure(figsize=(12,4))
    plt.plot(residuals.index, residuals.values)
    plt.axhline(thr, ls='--')
    plt.axhline(-thr, ls='--')
    plt.title(f"{pollutant}: residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{pollutant}_residuals.png"))
    plt.close()

    prec_r, rec_r, _ = precision_recall_curve(spikes.astype(int), residuals.abs()) if spikes.sum() else ([],[],[])
    prec_i, rec_i, _ = precision_recall_curve(spikes.astype(int), -iso_score) if spikes.sum() else ([],[],[])

    plt.figure(figsize=(5,5))
    if len(rec_r): plt.plot(rec_r, prec_r, label="Residual")
    if len(rec_i): plt.plot(rec_i, prec_i, label="IsolationForest")
    plt.title(f"{pollutant}: precision–recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{pollutant}_prcurve.png"))
    plt.close()

# save combined metrics
pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "detection_metrics_summary.csv"), index=False)

if all_rows:
    pd.concat(all_rows, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "top_anomalies_combined.csv"), index=False
    )

print("Completed. Outputs saved in:", OUT_DIR)

