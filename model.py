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

df_weather = pd.read_csv("weather-data.csv")
df_weather["date"] = pd.to_datetime(df_weather["date"])

df_air = pd.read_csv("air_quality_data.csv")
df_air["date"] = pd.to_datetime(df_air["date"])

df = pd.merge(df_weather, df_air, on="date", how="inner")  
df.to_csv("data.csv", index=False)

df = df.drop(columns=['snow', 'wdir', 'wpgt', 'tsun'], errors='ignore')
df = df.dropna(subset=['pm25'])

df['prcp'] = df['prcp'].fillna(0)
df["month"] = df["date"].dt.month

features = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres', 'month']

X = df[features]
y = df['pm25']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeRegressor(max_depth=3, random_state=42)
forest = RandomForestRegressor(n_estimators=200, max_depth=3, random_state=42)
linear = LinearRegression()

tree.fit(X_train, y_train)
forest.fit(X_train, y_train)
linear.fit(X_train, y_train)

tree_prediction = tree.predict(X_test)
forest_prediction = forest.predict(X_test)
linear_prediction = linear.predict(X_test)

tcross = cross_val_score(tree, X, y, cv=5, scoring="r2")
fcross = cross_val_score(forest, X, y, cv=5, scoring="r2")
lcross = cross_val_score(linear, X, y, cv=5, scoring="r2")
print("Cross-val scores:")
print(tcross, " ", tcross.mean())
print(fcross, " ", fcross.mean())
print(lcross, " ", lcross.mean())

tscore = r2_score(y_test, tree_prediction)
fscore = r2_score(y_test, forest_prediction)
lscore = r2_score(y_test, linear_prediction)
print("R2 scores:")
print(tscore)
print(fscore)
print(lscore)

#print(df.describe())

