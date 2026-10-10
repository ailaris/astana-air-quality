# Predicting Astana air quality from weather

It's a regression model that estimates the daily air quality index (AQI, based on PM2.5) in Astana from daily weather. The question was how much of the day-to-day variation weather alone can explain. On a test year the model never saw (2024), the best model explains about 42% of the variation and cuts the typical error by about a quarter compared with guessing the average.

## Results

Test set: all of 2024 (327 days). Training set: 2019 to 2023 (1587 days).

| Model | R2 on 2024 | Mean cross-validated R2 | MAE on 2024 |
|---|---|---|---|
| Random forest (500 trees, max_depth 5, min_samples_leaf 5) | 0.420 | 0.241 | 15.74 AQI points |
| Linear regression | 0.395 | 0.223 | not measured |
| Baseline: predict the training mean every day | n/a | n/a | 21.07 AQI points |

MAE is the average absolute error. The forest misses by about 16 AQI points on a typical day, which is a third of the width of one US AQI category (the first four categories span 50 points each).

Cross-validation uses `TimeSeriesSplit` with 5 folds. The first fold trains on a small early slice of the data and scores negative for every model, and the mean includes it.

## Data

**Weather.** Meteostat, daily values for Astana, 2019-01-01 to 2024-12-31. The final model uses minimum and maximum temperature (°C), wind speed (km/h) and pressure (hPa). Snow depth, wind direction, peak gust and sunshine duration were empty in the export, so I dropped them. Precipitation had many empty cells. I filled them with 0 for the experiments (an assumption: no record means no rain) and left the column out of the final model.

**Air quality.** Daily average PM2.5 for the US Embassy monitor in Nur-Sultan (Astana), from the historical data platform of the World Air Quality Index project (aqicn.org). The file covers June 2018 to March 2025, and 2135 of those days have a PM2.5 value. The values are AQI on the US EPA scale, not µg/m³.

**Merged dataset.** Joining weather and air quality on date for 2019 to 2024 leaves 1914 days. I dropped days without a PM2.5 value and did not fill them. The source has gaps: January to April 2023 is missing, and other months have scattered missing days (for example, September 2024 has 7 days).

**First attempt.** My first source was an OpenAQ export for an Astana reference-grade monitor. It had 895 usable days and nothing for 2020 and 2021, so I switched to the aqicn data. That export stays in the repo as `air_quality_data.csv` for reference. `model.py` does not read it.

## Method

- **Features:** `tmin`, `tmax`, `wspd`, `pres`, plus the month encoded as a sine and a cosine, so December and January sit next to each other.
- **Split:** chronological. Neighbouring days look alike, so a random split lets the model see near-copies of its test days. I train on the past and test on a later year.
- **Models:** random forest and linear regression. I also trained a single decision tree and dropped it (see below).
- **Tuning:** `GridSearchCV` over `max_depth`, `min_samples_leaf` and `n_estimators`, scored with `TimeSeriesSplit`. The best setting was depth 5, leaf size 5, 300 trees. I use 500 trees, which scored the same (cross-validated R2 0.2406 vs 0.2404).

## Experiment log

R² on the test set, with the mean cross-validated R² in brackets where I recorded it.

| Step | Tree | Forest | Linear |
|---|---|---|---|
| Random 80/20 split, OpenAQ data (895 days) | 0.334 | 0.417 | 0.343 |
| aqicn data, chronological split, month as a number | 0.238 | 0.450 | 0.388 |
| Month as sine and cosine | 0.262 (0.031) | 0.437 (0.224) | 0.399 (0.225) |
| Added temperature range (`tmax - tmin`) | 0.207 (0.022) | 0.453 (0.232) | 0.399 (0.225) |
| Added 3-day average wind and daily pressure change | 0.336 (0.070) | 0.434 (0.254) | 0.403 (0.251) |
| Reduced to six features (`tmin`, `tmax`, `wspd`, `pres`, month sine and cosine) | 0.342 (0.097) | 0.462 (0.234) | 0.395 (0.223) |
| Dropped the tree, forest with 500 trees at depth 3 | dropped | 0.469 (0.235) | 0.395 (0.223) |
| Final: grid-search settings (depth 5, leaf 5, 500 trees) | dropped | 0.420 (0.241) | 0.395 (0.223) |

The depth-3 forest scored higher on 2024 (0.469) than the final model (0.420). I chose that depth while looking at the 2024 score, so I report the grid-search model, which I selected by cross-validation.

## What was learned

- **Temperature range carries information.** When I replaced `tmin` and `tmax` with the daily average temperature, the forest fell from 0.44 to 0.30 on 2024. A large gap between the daily low and high may reflect calm, clear nights that trap pollution near the ground. I have not tested that explanation.
- **Temperature and wind drive the forest.** In an 11-feature run, the temperature columns together held about half of the feature importance and wind speed about a quarter.
- **Day of the week does not matter here.** Average AQI by weekday ranged from 56.8 to 58.9, so I added no weekday feature.
- **Extra features did not help.** The 3-day average wind and the daily pressure change left the scores within noise (about ±0.02), and precipitation had near-zero importance. I removed all three.
- **The forest edges out linear regression.** It leads on 2024 (0.420 vs 0.395), and the two are close in cross-validation (0.241 vs 0.223). The relationship between these weather inputs and AQI looks simple. The single decision tree overfit and scored between 0.21 and 0.34 depending on the features.

## Limitations

- Weather explains part of the variation. Heating, traffic and industry are not in the features, which is why R² stays well below 1.
- The test set is one year from one monitor. January to April 2023 is missing, so one winter is absent from training and cross-validation.
- I chose features and early settings while looking at the 2024 score, so 0.42 may be higher than the model would score on a new year.
- AQI is a nonlinear scale. Compare these numbers with other AQI models only.
- The aqicn data is not validated. This project is a demo and not a health tool.

## Run it

```bash
git clone <repo-url>
cd <repo-folder>
uv sync
```

Download the daily history for "Nur-Sultan, US Embassy" from the aqicn.org historical data platform (https://aqicn.org/data-platform/register/) and save it in the project folder as `new_pm2_data.csv`. Then run:

```bash
uv run model.py
```

The script merges the two files, trains the models and prints the scores.

## Data credits

- Weather: [Meteostat](https://meteostat.net).
- Air quality: US Embassy monitor data from the originating EPA/AirNow network, distributed by the [World Air Quality Index project](https://aqicn.org). Their terms state that the data is not fully verified or validated and is not official. The terms also forbid redistributing it as an archive, so the CSV is not in this repo.
- First-attempt air quality export: [OpenAQ](https://openaq.org).

## Status

The Streamlit app is in progress. It takes minimum and maximum temperature, wind speed, pressure and the month, and returns an estimated AQI with the typical error shown next to it.