import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

df_weather = pd.read_csv("weather-data.csv")
df_weather["date"] = pd.to_datetime(df_weather["date"])
df_weather = df_weather.sort_values("date")
df_weather["wspd_3d"] = df_weather["wspd"].rolling(3).mean()
df_weather["pres_change"]  = df_weather["pres"].diff()

df_air = pd.read_csv("new_pm2_data.csv", skipinitialspace=True)
df_air["date"] = pd.to_datetime(df_air["date"])
df_air = df_air[["date", "pm25"]].sort_values("date")

df = pd.merge(df_weather, df_air, on="date", how="inner")  
df.to_csv("data.csv", index=False)

df = df.drop(columns=['snow', 'wdir', 'wpgt', 'tsun'], errors='ignore')
df = df.dropna(subset=['pm25'])

df['prcp'] = df['prcp'].fillna(0)
df["month"] = df["date"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df['trange'] = df['tmax'] - df['tmin']

features = ['tmin', 'tmax','wspd', 'pres', 'month_sin', 'month_cos']

df = df.dropna(subset=features)

X = df[features]
y = df['pm25']

train = df[df["date"] < "2024-01-01"]
test = df[df["date"] >= "2024-01-01"]

# print(df.groupby([df["date"].dt.year, df["date"].dt.month]).size())

X_train, y_train = train[features], train["pm25"]
X_test, y_test = test[features], test["pm25"]

forest = RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=5, random_state=42)

forest.fit(X_train, y_train)
#print(pd.Series(forest.feature_importances_, index=features).sort_values(ascending=False))

forest_prediction = forest.predict(X_test)

cv = TimeSeriesSplit(n_splits=5)
fcross = cross_val_score(forest, X, y, cv=cv, scoring="r2")
print("Cross-val scores:")
print(fcross, " ", fcross.mean())

fscore = r2_score(y_test, forest_prediction)
print("R2 scores:")
print(fscore)

baseline = np.full(len(y_test), y_train.mean())
print("baseline", mean_absolute_error(y_test, baseline))
print(mean_absolute_error(y_test, forest_prediction))
#print(df.describe())

param_grid = {
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 9],
    "min_samples_leaf": [1, 5, 10, 20],
    "n_estimators": [100, 200, 300, 400, 500],
}
#grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=cv, scoring="r2", n_jobs=-1)
#grid.fit(X_train, y_train)
#print(grid.best_params_)
#{'max_depth': 5, 'min_samples_leaf': 5, 'n_estimators': 300}
