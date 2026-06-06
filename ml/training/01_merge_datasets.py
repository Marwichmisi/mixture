import pandas as pd
import numpy as np
import os

RAW = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
OUT = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_csv(name):
    path = os.path.join(RAW, name)
    df = pd.read_csv(path, low_memory=False)
    # Remove duplicate header rows
    df = df[df['device_identifier'] != 'device_identifier'].copy()
    return df

def to_datetime(df):
    df['ts'] = pd.to_datetime(df['ts_generation'].astype(np.int64), unit='ms')
    df.drop(columns=['ts_generation'], inplace=True)
    return df

def main():
    env = load_csv('stuard_environmental_data.csv')
    soil = load_csv('stuard_soil_data.csv')
    water = load_csv('stuard_water_meter_data.csv')

    to_datetime(env)
    to_datetime(soil)
    to_datetime(water)

    # Rename for clarity
    env.rename(columns={
        'temperature': 'air_temp',
        'humidity': 'air_humidity'
    }, inplace=True)
    soil.rename(columns={
        'humidity': 'soil_moisture',
        'temperature': 'soil_temp'
    }, inplace=True)

    # Keep only columns matching available sensors
    env = env[['ts', 'air_temp', 'air_humidity', 'pressure']].copy()
    soil = soil[['ts', 'line', 'soil_moisture']].copy()
    water = water[['ts', 'line', 'current_volume']].copy()

    soil['line'] = soil['line'].astype(int)
    water['line'] = water['line'].astype(int)

    # Merge soil + water by line + timestamp proximity (5 min window)
    # Using merge_asof for nearest timestamp within each line group
    soil_sorted = soil.sort_values('ts')
    water_sorted = water.sort_values('ts')

    merged = pd.merge_asof(
        soil_sorted,
        water_sorted,
        on='ts',
        by='line',
        direction='nearest',
        tolerance=pd.Timedelta('5min')
    )

    # Drop rows without matching water meter reading
    merged.dropna(subset=['current_volume'], inplace=True)

    # Merge with environmental by timestamp proximity (nearest, no line filter)
    env_sorted = env.sort_values('ts')

    final = pd.merge_asof(
        merged.sort_values('ts'),
        env_sorted,
        on='ts',
        direction='nearest',
        tolerance=pd.Timedelta('5min')
    )

    final.dropna(subset=['air_temp', 'air_humidity', 'pressure'], inplace=True)

    # Reorder columns
    final = final[['ts', 'line', 'soil_moisture', 'current_volume', 'air_temp', 'air_humidity', 'pressure']]
    final.sort_values('ts', inplace=True)
    final.reset_index(drop=True, inplace=True)

    out_path = os.path.join(OUT, 'merged_dataset.csv')
    final.to_csv(out_path, index=False)
    print(f"Saved {len(final)} rows to {OUT}/merged_dataset.csv")
    print(f"Columns: {list(final.columns)}")
    print(f"Date range: {final['ts'].min()} → {final['ts'].max()}")
    print(f"Samples per line:\n{final['line'].value_counts().sort_index()}")
    print(f"Memory: {final.memory_usage(deep=True).sum() / 1e6:.1f} MB")

if __name__ == '__main__':
    main()
