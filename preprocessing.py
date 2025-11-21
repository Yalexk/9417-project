import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer

in_file = 'air+quality/AirQualityUCI.csv'
out_file = 'air+quality/AirQualityUCI_with_timestamp.csv'

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

# put time stuff first
time_cols = [
    'Timestamp', 'Hour', 'Weekday', 'Month', 'DayOfMonth', 'DayOfYear',
    'Week', 'IsWeekend', 'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos',
    'Weekday_sin', 'Weekday_cos'
]
df = df[time_cols + [c for c in df.columns if c not in time_cols]]

# columns to scale (numeric but not the discrete/cyclical time features)
dont_scale = {
    'Hour','Weekday','Month','DayOfMonth','DayOfYear','Week','IsWeekend',
    'Hour_sin','Hour_cos','Month_sin','Month_cos','Weekday_sin','Weekday_cos'
}

# save unscaled
df.to_csv(out_file, index=False)

# impute missing values using knn
impute_cols = [
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

imputer = KNNImputer(n_neighbors=5, weights="uniform")
df[impute_cols] = pd.DataFrame(imputer.fit_transform(df[impute_cols]), columns=impute_cols)

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

df["CO_class"] = df["CO(GT)"].apply(discretise_co)

# scaled variants
df_std = df.copy()
df_min = df.copy()
df_rob = df.copy()

# can use different scaler based on model needs
sc_std = StandardScaler() # good for if data is normally distributed
sc_min = MinMaxScaler() # good for getting bounded values
sc_rob = RobustScaler() # good if data has outliers

df_std[impute_cols] = sc_std.fit_transform(df[impute_cols])
df_min[impute_cols] = sc_min.fit_transform(df[impute_cols])
df_rob[impute_cols] = sc_rob.fit_transform(df[impute_cols])

df_std.to_csv('air+quality/AirQualityUCI_standard_scaled.csv', index=False)
df_min.to_csv('air+quality/AirQualityUCI_minmax_scaled.csv', index=False)
df_rob.to_csv('air+quality/AirQualityUCI_robust_scaled.csv', index=False)

print("preprocessing complete")
