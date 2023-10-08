import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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


class lr1:

    def __init__(self):
        self.coef = None
        self.intercept = None

    def fit(self, X_train1, y_train1):
        X_train1["intercept"] = 1
        X_train1 = X_train1[['intercept', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'PT08.S2(NMHC)', 'NOx(GT)',
                             'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S5(O3)', 'T', 'RH', 'AH']]
        betas = np.linalg.inv(np.dot(X_train1.T, X_train1)).dot(X_train1.T).dot(y_train1)
        self.intercept = betas[0]
        self.coef = betas[1:]

    def predict(self, X_test1):
        y_pred1 = np.dot(X_test1, self.coef) + self.intercept
        return y_pred1


if __name__ == "__main__":
    lreg = lr1()
    lreg.fit(X_train, y_train)
    print(f"X_train shape = {X_train.shape}")
    y_pred = lreg.predict(X_test)
    print(f"co-efficients = {lreg.coef}")
    print(f"intercept = {lreg.intercept}")
    SSR = ((y_test - y_pred) ** 2).sum()
    SST = ((y_test - y_test.mean()) ** 2).sum()
    print(f"SSR = {SSR}")
    print(f"SST = {SST}")
    R2 = 1 - (SSR / SST)
    print(f"R2 = {R2}")
    plt.scatter(y_test, y_pred, color="b")
    plt.plot(y_test, y_pred, color="r")
    plt.show()





