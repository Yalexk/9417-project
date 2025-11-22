import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

in_file = 'air+quality/AirQualityUCI.csv'

# read csv (semicolon sep, comma decimals, treat -200 variants as NaN)
df = pd.read_csv(
    in_file,
    sep=';',
    decimal=',',
    na_values=['-200', -200, '-200,0', 'NaN', 'NA', ''],
    engine='python'
)

# drop empty columns (those last two)
df = df.dropna(axis=1, how='all')

# make timestamp and drop original date/time
df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
df = df.drop(columns=['Date', 'Time'])

# time parts
df['Hour'] = df['Timestamp'].dt.hour
df['Weekday'] = df['Timestamp'].dt.dayofweek # monday = 0, Sunday = 6
df['Month'] = df['Timestamp'].dt.month
df['DayOfMonth'] = df['Timestamp'].dt.day
df['DayOfYear'] = df['Timestamp'].dt.dayofyear
df['Week'] = df['Timestamp'].dt.isocalendar().week
df['IsWeekend'] = (df['Weekday'] >= 5).astype(int) # weekend = 1, weekday = 0 - for traffic/ travel

# cyclical encodings (sin/cos) so that 11pm is close to 1am,  
# december is close to january, sunday close to monday
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['Weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
df['Weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)

cyclical_time_cols = [
    'Hour_sin',
    'Hour_cos',
    'Month_sin',
    'Month_cos',
    'Weekday_sin',
    'Weekday_cos'
]

raw_time_cols = [
    'Hour',
    'Weekday',
    'Month',
    'DayOfMonth',
    'DayOfYear',
    'Week'
]

regression_pollutants = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
for col in regression_pollutants:
    df[f'{col}_raw'] = df[col]
raw_target_cols = [f'{col}_raw' for col in regression_pollutants]
target_cols = raw_target_cols + ['CO_class']

# forward fill for feature creation
sensor_cols = [
    'CO(GT)',
    'PT08.S1(CO)',
    'NMHC(GT)',
    'C6H6(GT)',
    'PT08.S2(NMHC)',
    'NOx(GT)',
    'PT08.S3(NOx)',
    'NO2(GT)',
    'PT08.S4(NO2)',
    'PT08.S5(O3)',
    'T',
    'RH',
    'AH'
]
df[sensor_cols] = df[sensor_cols].ffill()
# df[sensor_cols] = df[sensor_cols].bfill()

lag_intervals = [1, 6, 12, 24] 

lag_cols = []
for col in sensor_cols:
    for lag in lag_intervals:
        new_col_name = f'{col}_lag{lag}'
        df[new_col_name] = df[col].shift(lag)
        lag_cols.append(new_col_name)

# drop first 24 hours (to account for 24 hour lag)
df = df.dropna(subset=lag_cols)

# Discretise co class with a new feature column
def discretise_co(value):
    if pd.isna(value):
        return np.nan
    if value < 1.5:
        return "low"
    elif value < 2.5:
        return "mid"
    else:
        return "high"

df['CO_class'] = df['CO(GT)'].apply(discretise_co)

# drop rows where any raw values is missing
df = df.dropna(subset=['CO(GT)_raw'])

# split into 2004, 2005 before scaling to prevent data leakage
train_df = df[df['Timestamp'].dt.year == 2004].copy()
test_df = df[df['Timestamp'].dt.year == 2005].copy()

# scale first then impute as we're using distance based imputer (kNN)

impute_cols = sensor_cols + lag_cols + cyclical_time_cols
scale_cols = impute_cols + raw_time_cols

sc_std = StandardScaler()
train_df[scale_cols] = sc_std.fit_transform(train_df[scale_cols])
test_df[scale_cols] = sc_std.transform(test_df[scale_cols])

imputer = KNNImputer(n_neighbors=5, weights="uniform")
train_df[impute_cols] = imputer.fit_transform(train_df[impute_cols])
test_df[impute_cols] = imputer.transform(test_df[impute_cols])

# reorder cols (date, CO(GT), the rest)
date_cols = [
    'Timestamp', 'Month', 'DayOfMonth', 'DayOfYear', 'Week', 'Weekday', 'IsWeekend', 'Hour', 
    'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'Weekday_sin', 'Weekday_cos'
]
rest_col = [c for c in train_df.columns if c not in date_cols + target_cols]
final_order = date_cols + target_cols + rest_col
train_df = train_df[final_order]
test_df = test_df[final_order]

train_df.to_csv('air+quality/AirQualityUCI_standard_scaled_v2_train.csv', index=False)
test_df.to_csv('air+quality/AirQualityUCI_standard_scaled_v2_test.csv', index=False)

print("preprocessing complete")
print("regression: pollutant_raw (unscaled)")
print("classification: use CO_class (low/med/high)")
