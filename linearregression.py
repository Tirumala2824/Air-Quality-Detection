import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

csv_filename = "AirQualityUCI.csv"
df = pd.read_csv(csv_filename, sep=";", parse_dates=['Date', 'Time'])
df.dropna(how="all", axis=1, inplace=True)
df.dropna(how="all", axis=0, inplace=True)
cols = list(df.columns[2:])
for col in cols:
    if df[col].dtype != 'float64':
        str_x = pd.Series(df[col]).str.replace(',', '.')
        float_X = []
        for value in str_x.values:
            fv = float(value)
            float_X.append(fv)
            df[col] = pd.DataFrame(float_X)

features = list(df.columns)
features.remove('Date')
features.remove('Time')
features.remove('PT08.S4(NO2)')

X = df[['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)',
        'PT08.S5(O3)', 'T', 'RH', 'AH']].copy()
y = df[['C6H6(GT)']].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
print(f"X_train = {X_train.shape}")
print(f"X_test = {X_test.shape}")
print(f"y_train = {y_train.shape}")
print(f"y_test = {y_test.shape}")

print(f"\nUsing inbuilt function:")
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"reg.coef_ = {reg.coef_}")
print(f"reg.intercept_ = {reg.intercept_}")
print(f"reg.R2 = {r2_score(y_test, y_pred)}")
plt.scatter(y_test, y_pred, color="b")
plt.plot(y_test, y_pred, color="r")
plt.show()
