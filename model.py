import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

df_weather = pd.read_csv("weather-data.csv")
df_weather["date"] = pd.to_datetime(df_weather["date"])

df_air = pd.read_csv("air_quality_data.csv")
df_air["date"] = pd.to_datetime(df_air["date"])

df = pd.merge(df_weather, df_air, on="date", how="inner")  
df.to_csv("data.csv", index=False)

df = df.drop(columns=['snow', 'wdir', 'wpgt', 'tsun'], errors='ignore')
df = df.dropna(subset=['pm25'])



print(df.describe())

