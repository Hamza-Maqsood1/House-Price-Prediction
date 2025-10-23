import pandas as pd

data = pd.read_csv(r"C:\Users\Hamza\Downloads\Cloud_credits\Predicting House Prices\HousingData.csv")

print(data.shape)
print(data.head())
print(data.info())

print(data.isnull().sum())

data.fillna(data.mean(), inplace=True)
print(data.isnull().sum())

data = pd.get_dummies(data, drop_first=True)


from sklearn.preprocessing import StandardScaler

X = data.drop("MEDV", axis=1)
y = data["MEDV"]

scaler = StandardScaler()
X_Scaled = scaler.fit_transform(X)


print("Feature Shape : ", X_Scaled.shape)
print("Target Shape : ", y.shape)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X_Scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Square Error : ", mse)
print("R2 Score : ", r2)


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()