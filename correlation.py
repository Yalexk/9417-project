import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./air+quality/AirQualityUCI_with_timestamp.csv")
df = df[
    [
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
]
corr = df.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")

