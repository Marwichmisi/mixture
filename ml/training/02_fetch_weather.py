import requests
import pandas as pd
import os
from datetime import datetime, timedelta

OUT = os.path.join(os.path.dirname(__file__), '..', 'data')

def fetch_open_meteo(lat, lon, start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": [
            "precipitation",
            "wind_speed_10m",
            "et0_fao_evapotranspiration",
            "shortwave_radiation"
        ],
        "timezone": "UTC"
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    hourly = data['hourly']
    df = pd.DataFrame({
        'ts': pd.to_datetime(hourly['time']),
        'rain': hourly['precipitation'],
        'wind_speed': hourly['wind_speed_10m'],
        'et0': hourly['et0_fao_evapotranspiration'],
        'solar_radiation': hourly['shortwave_radiation']
    })
    return df

def main():
    lat, lon = 44.8, 10.3  # Parma, Italy
    start = "2023-06-28"
    end = "2023-09-14"

    print(f"Fetching Open-Meteo data for Parma ({lat}, {lon})...")
    print(f"Period: {start} → {end}")

    df = fetch_open_meteo(lat, lon, start, end)
    print(f"Received {len(df)} hourly rows")

    out_path = os.path.join(OUT, 'weather_parma_2023.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    print(f"Rain days: {(df['rain'] > 0).sum()} / {len(df)}")
    print(f"Rain total: {df['rain'].sum():.1f} mm")

if __name__ == '__main__':
    main()
