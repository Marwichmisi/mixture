import pandas as pd
import numpy as np
import os

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT = os.path.join(os.path.dirname(__file__), '..', 'data')

def add_temporal_features(df):
    df = df.copy()
    hour = df['ts'].dt.hour
    weekday = df['ts'].dt.weekday
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
    return df

def aggregate_6h(df):
    df = df.copy()
    df['ts_6h'] = df['ts'].dt.floor('6h')

    # Group by line + 6h window
    grouped = df.groupby(['line', 'ts_6h'])

    rows = []
    for (line_id, ts_window), grp in grouped:
        grp = grp.sort_values('ts')
        n = len(grp)
        if n < 2:
            continue

        row = {'ts': ts_window, 'line': line_id}

        row['air_temp'] = grp['air_temp'].mean()
        row['humidity'] = grp['air_humidity'].mean()
        row['pressure'] = grp['pressure'].mean()
        row['soil_moisture'] = grp['soil_moisture'].mean()

        # Soil moisture trend over the window
        x_secs = (grp['ts'] - grp['ts'].iloc[0]).dt.total_seconds().values
        y_moist = grp['soil_moisture'].values.astype(np.float64)
        if n >= 3 and np.std(x_secs) > 1e-6:
            slope = np.polyfit(x_secs, y_moist, 1)[0] * 3600
        else:
            slope = 0.0
        row['soil_moisture_trend'] = float(slope)

        # Water volume increase
        vol_start = grp['current_volume'].iloc[0]
        vol_end = grp['current_volume'].iloc[-1]
        row['volume_increase'] = vol_end - vol_start

        # Weather
        if 'rain' in grp.columns and grp['rain'].notna().any():
            row['rain_6h'] = grp['rain'].sum()
            row['wind_speed'] = grp['wind_speed'].mean()
            row['et0'] = grp['et0'].sum()
            row['solar_radiation'] = grp['solar_radiation'].mean()
            mode = 'A'
        else:
            mode = 'B'

        # Temporal features at midpoint
        mid = ts_window + pd.Timedelta(hours=3)
        h = mid.hour
        wd = mid.weekday()
        row['hour_sin'] = np.sin(2 * np.pi * h / 24)
        row['hour_cos'] = np.cos(2 * np.pi * h / 24)
        row['weekday_sin'] = np.sin(2 * np.pi * wd / 7)
        row['weekday_cos'] = np.cos(2 * np.pi * wd / 7)

        row['mode'] = mode
        rows.append(row)

    return pd.DataFrame(rows)

def create_target(df):
    df = df.copy()
    for line_id in df['line'].unique():
        mask = df['line'] == line_id
        vals = df.loc[mask, 'volume_increase']
        # Class 0: no irrigation (volume_increase <= small threshold)
        # Class 1/2: split non-zero values by median
        nonzero = vals[vals > 1.0]
        if len(nonzero) < 5:
            # Not enough irrigation events: use soil moisture instead
            moist = df.loc[mask, 'soil_moisture']
            lo = moist.quantile(0.33)
            hi = moist.quantile(0.67)
            if abs(hi - lo) < 1e-6:
                df.loc[mask, 'class'] = 0
            else:
                df.loc[mask, 'class'] = pd.cut(
                    moist, bins=[-np.inf, lo, hi, np.inf],
                    labels=[0, 1, 2]
                )
        else:
            med = nonzero.median()
            cond = vals.values
            cls = np.zeros(len(cond), dtype=int)
            cls[cond > 1.0] = 1
            cls[cond > med] = 2
            df.loc[mask, 'class'] = cls
    df['class'] = df['class'].astype(int)
    return df

def main():
    merged = pd.read_csv(os.path.join(DATA, 'merged_dataset.csv'), parse_dates=['ts'])
    merged = add_temporal_features(merged)

    weather_path = os.path.join(DATA, 'weather_parma_2023.csv')
    if os.path.exists(weather_path):
        print("Weather data found")
        weather = pd.read_csv(weather_path, parse_dates=['ts'])
        weather.rename(columns={'ts': 'ts_weather'}, inplace=True)
        merged['ts_hour'] = merged['ts'].dt.floor('h')
        weather['ts_hour'] = weather['ts_weather'].dt.floor('h')
        merged = merged.merge(weather, on='ts_hour', how='left')
        merged.drop(columns=['ts_hour', 'ts_weather'], inplace=True)
    else:
        print("No weather data")
        for col in ['rain', 'wind_speed', 'et0', 'solar_radiation']:
            merged[col] = np.nan

    agg = aggregate_6h(merged)
    agg = create_target(agg)

    print(f"Total 6h windows: {len(agg)}")
    print(f"Class distribution:\n{agg['class'].value_counts().sort_index()}")
    print(f"Per line volume_increase stats:\n{agg.groupby('line')['volume_increase'].describe()}")

    agg.to_csv(os.path.join(OUT, 'aggregated_6h.csv'), index=False)
    print(f"Saved to {OUT}/aggregated_6h.csv")

if __name__ == '__main__':
    main()
