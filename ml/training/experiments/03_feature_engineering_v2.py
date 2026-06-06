import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
import tensorflow as tf
from tensorflow import keras
import json
import pickle

DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
MODELS = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
EXPERIMENTS = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'experiments_v2')
os.makedirs(EXPERIMENTS, exist_ok=True)

CLASS_LABELS = ['No watering', 'Short watering', 'Long watering']


def add_temporal_features(df):
    df = df.copy()
    hour = df['ts'].dt.hour
    weekday = df['ts'].dt.weekday
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
    return df


def add_interaction_features(df):
    df = df.copy()
    # Vapor Pressure Deficit (kPa)
    es = 0.6108 * np.exp(17.27 * df['air_temp'] / (df['air_temp'] + 237.3))
    ea = es * df['humidity'] / 100.0
    df['vapor_pressure_deficit'] = es - ea
    # Heat stress index
    df['heat_stress'] = df['air_temp'] * (1 - df['humidity'] / 100)
    # Moisture-temperature ratio
    df['moisture_temp_ratio'] = df['soil_moisture'] / (df['air_temp'] + 1)
    return df


def aggregate_6h_v2(df):
    df = df.copy()
    df['ts_6h'] = df['ts'].dt.floor('6h')

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

        # Interaction features (computed on raw means before VPD calc)
        # VPD calculated per-window from aggregated means
        es = 0.6108 * np.exp(17.27 * row['air_temp'] / (row['air_temp'] + 237.3))
        ea = es * row['humidity'] / 100.0
        row['vapor_pressure_deficit'] = es - ea
        row['heat_stress'] = row['air_temp'] * (1 - row['humidity'] / 100)
        row['moisture_temp_ratio'] = row['soil_moisture'] / (row['air_temp'] + 1)

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

    agg = pd.DataFrame(rows)
    agg = agg.sort_values(['line', 'ts']).reset_index(drop=True)

    # --- Enhanced features ---
    # 1. Lag features (previous 6h window)
    for col in ['soil_moisture', 'air_temp', 'humidity']:
        agg[f'{col}_lag1'] = agg.groupby('line')[col].shift(1)

    # 2. Rate of change
    for col in ['soil_moisture', 'air_temp', 'humidity']:
        agg[f'{col}_delta'] = agg[col] - agg[f'{col}_lag1']

    # 3. Rolling statistics (24h = 4 windows)
    for line_id in agg['line'].unique():
        mask = agg['line'] == line_id
        idx = agg.index[mask]
        for col, stats in [('soil_moisture', ['mean', 'max', 'min']), ('air_temp', ['max'])]:
            for stat in stats:
                col_name = f'{col}_{stat}_24h'
                if stat == 'mean':
                    agg.loc[idx, col_name] = (
                        agg.loc[idx, col].rolling(window=4, min_periods=1).mean()
                    )
                elif stat == 'max':
                    agg.loc[idx, col_name] = (
                        agg.loc[idx, col].rolling(window=4, min_periods=1).max()
                    )
                elif stat == 'min':
                    agg.loc[idx, col_name] = (
                        agg.loc[idx, col].rolling(window=4, min_periods=1).min()
                    )

    # Fill NaN from lag/rolling with forward fill then 0
    agg = agg.ffill().fillna(0)

    return agg


def create_target(df):
    df = df.copy()
    for line_id in df['line'].unique():
        mask = df['line'] == line_id
        vals = df.loc[mask, 'volume_increase']
        nonzero = vals[vals > 1.0]
        if len(nonzero) < 5:
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


def build_model(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_LABELS,
                                   zero_division=0, output_dict=True)
    per_class = {}
    for i, label in enumerate(CLASS_LABELS):
        key = label
        per_class[key] = {
            'precision': report[key]['precision'] if key in report else 0.0,
            'recall': report[key]['recall'] if key in report else 0.0,
            'f1': report[key]['f1-score'] if key in report else 0.0,
            'support': int(report[key]['support']) if key in report else 0,
        }
    return {
        'accuracy': float(acc),
        'precision_weighted': float(prec),
        'recall_weighted': float(rec),
        'f1_weighted': float(f1),
        'confusion_matrix': cm.tolist(),
        'per_class': per_class,
        'pred_dist': np.bincount(y_pred, minlength=3).tolist(),
        'true_dist': np.bincount(y_true, minlength=3).tolist(),
    }


def main():
    print("=" * 60)
    print("ENHANCED FEATURE ENGINEERING v2")
    print("=" * 60)

    merged = pd.read_csv(os.path.join(DATA, 'merged_dataset.csv'), parse_dates=['ts'])
    merged = add_temporal_features(merged)

    weather_path = os.path.join(DATA, 'weather_parma_2023.csv')
    if os.path.exists(weather_path):
        print("Weather data found — merging.")
        weather = pd.read_csv(weather_path, parse_dates=['ts'])
        weather.rename(columns={'ts': 'ts_weather'}, inplace=True)
        merged['ts_hour'] = merged['ts'].dt.floor('h')
        weather['ts_hour'] = weather['ts_weather'].dt.floor('h')
        merged = merged.merge(weather, on='ts_hour', how='left')
        merged.drop(columns=['ts_hour', 'ts_weather'], inplace=True)
    else:
        print("No weather data.")
        for col in ['rain', 'wind_speed', 'et0', 'solar_radiation']:
            merged[col] = np.nan

    agg = aggregate_6h_v2(merged)
    agg = create_target(agg)

    print(f"\nTotal 6h windows (v2): {len(agg)}")
    print(f"Class distribution:\n{agg['class'].value_counts().sort_index()}")
    print(f"Columns:\n{list(agg.columns)}")

    agg.to_csv(os.path.join(DATA, 'aggregated_6h_v2.csv'), index=False)
    print(f"Saved → {DATA}/aggregated_6h_v2.csv")

    # === Features ===
    base_features = [
        'air_temp', 'humidity', 'pressure', 'soil_moisture',
        'soil_moisture_trend',
        'vapor_pressure_deficit', 'heat_stress', 'moisture_temp_ratio',
        'soil_moisture_lag1', 'air_temp_lag1', 'humidity_lag1',
        'soil_moisture_delta', 'air_temp_delta', 'humidity_delta',
        'soil_moisture_mean_24h', 'soil_moisture_max_24h', 'soil_moisture_min_24h',
        'air_temp_max_24h',
        'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
    ]
    # V2 Mode B: 21 features (base without weather)
    features_b = base_features
    # V2 Mode A: 21 + 4 weather = 25
    features_a = features_b + ['rain_6h', 'wind_speed', 'et0', 'solar_radiation']
    target = 'class'

    y = agg[target].values.astype(np.int32)

    print(f"\nFeature count (Mode B): {len(features_b)}")
    print(f"Feature count (Mode A): {len(features_a)}")
    print(f"Features B: {features_b}")

    # === Compare with baseline class distribution (+ accuracy from earlier run) ===
    baseline_path = os.path.join(DATA, 'aggregated_6h.csv')
    if os.path.exists(baseline_path):
        baseline = pd.read_csv(baseline_path)
        print(f"\nBaseline size: {len(baseline)}, V2 size: {len(agg)}")
        print(f"Baseline class dist:\n{baseline['class'].value_counts().sort_index()}")
        print(f"V2 class dist:\n{agg['class'].value_counts().sort_index()}")

    # ============= MODE B (sensors only) =============
    print("\n" + "=" * 60)
    print("MODE B — Enhanced features (sensors only)")
    print("=" * 60)

    X = agg[features_b].values.astype(np.float32)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Train class dist: {np.bincount(y_train, minlength=3)}")
    print(f"Val   class dist: {np.bincount(y_val, minlength=3)}")
    print(f"Test  class dist: {np.bincount(y_test, minlength=3)}")

    classes = np.bincount(y_train, minlength=3)
    total = len(y_train)
    class_weight = {i: total / (3 * count) if count > 0 else 1.0
                    for i, count in enumerate(classes)}
    print(f"Class weights: {class_weight}")

    scaler_b = MinMaxScaler()
    X_train_b_s = scaler_b.fit_transform(X_train)
    X_val_b_s = scaler_b.transform(X_val)
    X_test_b_s = scaler_b.transform(X_test)

    model_b = build_model(len(features_b))
    cb = [
        keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True,
                                      monitor='val_accuracy'),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10,
                                          monitor='val_accuracy'),
    ]
    model_b.fit(
        X_train_b_s, y_train,
        validation_data=(X_val_b_s, y_val),
        epochs=200, batch_size=32,
        class_weight=class_weight,
        callbacks=cb, verbose=0
    )

    val_loss_b, val_acc_b = model_b.evaluate(X_val_b_s, y_val, verbose=0)
    test_loss_b, test_acc_b = model_b.evaluate(X_test_b_s, y_test, verbose=0)
    y_pred_b = np.argmax(model_b.predict(X_test_b_s, verbose=0), axis=1)

    results_b = compute_metrics(y_test, y_pred_b)
    results_b['val_accuracy'] = float(val_acc_b)
    results_b['test_accuracy'] = float(test_acc_b)

    model_b.save(os.path.join(EXPERIMENTS, 'mlp_v2_model_b.keras'))
    with open(os.path.join(EXPERIMENTS, 'scaler_v2_b.pkl'), 'wb') as f:
        pickle.dump({'scaler': scaler_b, 'features': features_b}, f)

    # ============= MODE A (sensors + weather) =============
    print("\n" + "=" * 60)
    print("MODE A — Enhanced features (sensors + weather)")
    print("=" * 60)

    # Filter to Mode A rows (have weather data)
    mask_a = agg['mode'] == 'A'
    print(f"Mode A rows: {mask_a.sum()} / {len(agg)}")

    X_a_full = agg[features_a].values.astype(np.float32)
    y_a_full = agg[target].values.astype(np.int32)

    # Use only Mode A samples
    X_a = X_a_full[mask_a.values]
    y_a = y_a_full[mask_a.values]

    print(f"Mode A samples after filter: {len(X_a)}")
    print(f"Mode A class dist: {np.bincount(y_a, minlength=3)}")

    results_a = None
    if len(X_a) >= 50:
        X_train_a, X_temp_a, y_train_a, y_temp_a = train_test_split(
            X_a, y_a, test_size=0.4, stratify=y_a, random_state=42
        )
        X_val_a, X_test_a, y_val_a, y_test_a = train_test_split(
            X_temp_a, y_temp_a, test_size=0.5, stratify=y_temp_a, random_state=42
        )

        scaler_a = MinMaxScaler()
        X_train_a_s = scaler_a.fit_transform(X_train_a)
        X_val_a_s = scaler_a.transform(X_val_a)
        X_test_a_s = scaler_a.transform(X_test_a)

        model_a = build_model(len(features_a))
        model_a.fit(
            X_train_a_s, y_train_a,
            validation_data=(X_val_a_s, y_val_a),
            epochs=200, batch_size=32,
            class_weight=class_weight,
            callbacks=cb, verbose=0
        )

        val_loss_a, val_acc_a = model_a.evaluate(X_val_a_s, y_val_a, verbose=0)
        test_loss_a, test_acc_a = model_a.evaluate(X_test_a_s, y_test_a, verbose=0)
        y_pred_a = np.argmax(model_a.predict(X_test_a_s, verbose=0), axis=1)

        results_a = compute_metrics(y_test_a, y_pred_a)
        results_a['val_accuracy'] = float(val_acc_a)
        results_a['test_accuracy'] = float(test_acc_a)

        model_a.save(os.path.join(EXPERIMENTS, 'mlp_v2_model_a.keras'))
        with open(os.path.join(EXPERIMENTS, 'scaler_v2_a.pkl'), 'wb') as f:
            pickle.dump({'scaler': scaler_a, 'features': features_a}, f)

    # ============= PRINT COMPARISON =============
    print("\n\n" + "=" * 60)
    print("COMPARISON: BASELINE vs V2 (Enhanced)")
    print("=" * 60)

    print(f"\n{'Metric':<30} {'Mode B Base':<15} {'Mode B V2':<15} {'Mode A Base':<15} {'Mode A V2':<15}")
    print("-" * 90)

    # These are approximate baseline values from the existing report/run
    baseline_b = {'accuracy': 0.422, 'precision_weighted': 0.178, 'recall_weighted': 0.422, 'f1_weighted': 0.251}
    baseline_a = {'accuracy': 0.475, 'precision_weighted': 0.226, 'recall_weighted': 0.475, 'f1_weighted': 0.307}

    for metric in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
        b_base = f"{baseline_b[metric]:.4f}"
        b_v2 = f"{results_b[metric]:.4f}" if results_b else "N/A"
        a_base = f"{baseline_a[metric]:.4f}" if results_a else "N/A"
        a_v2 = f"{results_a[metric]:.4f}" if results_a else "N/A"
        print(f"{metric:<30} {b_base:<15} {b_v2:<15} {a_base:<15} {a_v2:<15}")

    print("\n" + "-" * 60)
    print("MODE B V2 — Per-class metrics (test set)")
    print("-" * 60)
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("-" * 60)
    for label in CLASS_LABELS:
        p = results_b['per_class'][label]
        print(f"{label:<20} {p['precision']:<10.3f} {p['recall']:<10.3f} {p['f1']:<10.3f} {p['support']:<10}")
    print(f"\nPred dist: {results_b['pred_dist']}")
    print(f"True  dist: {results_b['true_dist']}")

    if results_a:
        print("\n" + "-" * 60)
        print("MODE A V2 — Per-class metrics (test set)")
        print("-" * 60)
        print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
        print("-" * 60)
        for label in CLASS_LABELS:
            p = results_a['per_class'][label]
            print(f"{label:<20} {p['precision']:<10.3f} {p['recall']:<10.3f} {p['f1']:<10.3f} {p['support']:<10}")
        print(f"\nPred dist: {results_a['pred_dist']}")
        print(f"True  dist: {results_a['true_dist']}")

    # Comparison conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    b_improvement = results_b['accuracy'] - baseline_b['accuracy']
    print(f"\nMode B:  Baseline={baseline_b['accuracy']:.4f} → V2={results_b['accuracy']:.4f} (Δ={b_improvement:+.4f})")
    if results_a:
        a_improvement = results_a['accuracy'] - baseline_a['accuracy']
        print(f"Mode A:  Baseline={baseline_a['accuracy']:.4f} → V2={results_a['accuracy']:.4f} (Δ={a_improvement:+.4f})")

    outperforms = b_improvement > 0
    print(f"\n→ V2 {'OUTPERFORMS' if outperforms else 'does NOT outperform'} baseline on Mode B accuracy.")
    if results_a:
        outperforms_a = a_improvement > 0
        print(f"→ V2 {'OUTPERFORMS' if outperforms_a else 'does NOT outperform'} baseline on Mode A accuracy.")

    # Save results summary
    summary = {
        'features_b': features_b,
        'features_a': features_a,
        'n_features_b': len(features_b),
        'n_features_a': len(features_a),
        'n_samples': len(agg),
        'class_distribution': agg['class'].value_counts().sort_index().to_dict(),
        'mode_b': results_b,
        'mode_a': results_a,
        'comparison': {
            'baseline_b_accuracy': baseline_b['accuracy'],
            'v2_b_accuracy': results_b['accuracy'],
            'baseline_a_accuracy': baseline_a['accuracy'],
            'v2_a_accuracy': results_a['accuracy'] if results_a else None,
            'b_improvement': b_improvement,
            'a_improvement': a_improvement if results_a else None,
            'v2_outperforms_baseline_b': bool(outperforms),
            'v2_outperforms_baseline_a': bool(results_a and a_improvement > 0) if results_a else None,
        },
        'class_labels': CLASS_LABELS,
    }
    with open(os.path.join(EXPERIMENTS, 'v2_training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ All results saved to {EXPERIMENTS}/")
    print("✅ aggregated_6h_v2.csv saved to data/")


if __name__ == '__main__':
    main()
