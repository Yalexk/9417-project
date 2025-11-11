import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

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
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
num_cols = [c for c in num_cols if c not in dont_scale]

# save unscaled
df.to_csv(out_file, index=False)

# scaled variants
df_std = df.copy()
df_min = df.copy()
df_rob = df.copy()

# can use different scaler based on model needs
sc_std = StandardScaler() # good for if data is normally distributed
sc_min = MinMaxScaler() # good for getting bounded values
sc_rob = RobustScaler() # good if data has outliers

df_std[num_cols] = sc_std.fit_transform(df[num_cols])
df_min[num_cols] = sc_min.fit_transform(df[num_cols])
df_rob[num_cols] = sc_rob.fit_transform(df[num_cols])

df_std.to_csv('air+quality/AirQualityUCI_standard_scaled.csv', index=False)
df_min.to_csv('air+quality/AirQualityUCI_minmax_scaled.csv', index=False)
df_rob.to_csv('air+quality/AirQualityUCI_robust_scaled.csv', index=False)

print("preprocessing complete")
