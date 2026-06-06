import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import json
import pickle

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS, exist_ok=True)

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

def main():
    agg = pd.read_csv(os.path.join(DATA, 'aggregated_6h.csv'), parse_dates=['ts'])

    print(f"Total samples: {len(agg)}")
    print(f"Overall class distribution:\n{agg['class'].value_counts().sort_index()}")

    features_b = [
        'air_temp', 'humidity', 'pressure', 'soil_moisture',
        'soil_moisture_trend',
        'hour_sin', 'hour_cos',
        'weekday_sin', 'weekday_cos'
    ]
    features_a = features_b + ['rain_6h', 'wind_speed', 'et0', 'solar_radiation']
    target = 'class'

    # Stratified random split (POC: temporal structure noted as limitation)
    X = agg[features_b].values.astype(np.float32)
    y = agg[target].values.astype(np.int32)

    X_train_b, X_temp, y_train_b, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val_b, X_test_b, y_val_b, y_test_b = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"\nTrain: {len(X_train_b)}, Val: {len(X_val_b)}, Test: {len(X_test_b)}")
    print(f"Train  class dist: {np.bincount(y_train_b, minlength=3)}")
    print(f"Val    class dist: {np.bincount(y_val_b, minlength=3)}")
    print(f"Test   class dist: {np.bincount(y_test_b, minlength=3)}")

    # Class weights
    classes = np.bincount(y_train_b, minlength=3)
    total = len(y_train_b)
    class_weight = {i: total / (3 * count) if count > 0 else 1.0 for i, count in enumerate(classes)}
    print(f"Class weights: {class_weight}")

    # --- Mode B ---
    print("\n" + "="*50)
    print("TRAINING MODE B (sensors only)")
    print("="*50)

    scaler_b = MinMaxScaler()
    X_train_b_s = scaler_b.fit_transform(X_train_b)
    X_val_b_s = scaler_b.transform(X_val_b)
    X_test_b_s = scaler_b.transform(X_test_b)

    model_b = build_model(len(features_b))
    cb = [
        keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True, monitor='val_accuracy'),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, monitor='val_accuracy')
    ]
    history_b = model_b.fit(
        X_train_b_s, y_train_b,
        validation_data=(X_val_b_s, y_val_b),
        epochs=200, batch_size=32,
        class_weight=class_weight,
        callbacks=cb, verbose=2
    )

    val_loss_b, val_acc_b = model_b.evaluate(X_val_b_s, y_val_b, verbose=0)
    test_loss_b, test_acc_b = model_b.evaluate(X_test_b_s, y_test_b, verbose=0)
    print(f"\nMode B - Val acc: {val_acc_b:.4f}, Test acc: {test_acc_b:.4f}")

    y_pred_b = np.argmax(model_b.predict(X_test_b_s, verbose=0), axis=1)
    print(f"Predicted: {np.bincount(y_pred_b, minlength=3)}")
    print(f"True:      {np.bincount(y_test_b, minlength=3)}")

    model_b.save(os.path.join(MODELS, 'mlp_model_b.keras'))
    with open(os.path.join(MODELS, 'scaler_b.pkl'), 'wb') as f:
        pickle.dump({'scaler': scaler_b, 'features': features_b}, f)

    # --- Mode A ---
    print("\n" + "="*50)
    print("TRAINING MODE A (sensors + weather)")
    print("="*50)

    X_a = agg[features_a].values.astype(np.float32)
    X_train_a, X_temp_a, y_train_a, y_temp_a = train_test_split(
        X_a, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val_a, X_test_a, y_val_a, y_test_a = train_test_split(
        X_temp_a, y_temp_a, test_size=0.5, stratify=y_temp_a, random_state=42
    )

    scaler_a = MinMaxScaler()
    X_train_a_s = scaler_a.fit_transform(X_train_a)
    X_val_a_s = scaler_a.transform(X_val_a)
    X_test_a_s = scaler_a.transform(X_test_a)

    model_a = build_model(len(features_a))
    history_a = model_a.fit(
        X_train_a_s, y_train_a,
        validation_data=(X_val_a_s, y_val_a),
        epochs=200, batch_size=32,
        class_weight=class_weight,
        callbacks=cb, verbose=2
    )

    val_loss_a, val_acc_a = model_a.evaluate(X_val_a_s, y_val_a, verbose=0)
    test_loss_a, test_acc_a = model_a.evaluate(X_test_a_s, y_test_a, verbose=0)
    print(f"\nMode A - Val acc: {val_acc_a:.4f}, Test acc: {test_acc_a:.4f}")

    y_pred_a = np.argmax(model_a.predict(X_test_a_s, verbose=0), axis=1)
    print(f"Predicted: {np.bincount(y_pred_a, minlength=3)}")
    print(f"True:      {np.bincount(y_test_a, minlength=3)}")

    model_a.save(os.path.join(MODELS, 'mlp_model_a.keras'))
    with open(os.path.join(MODELS, 'scaler_a.pkl'), 'wb') as f:
        pickle.dump({'scaler': scaler_a, 'features': features_a}, f)

    # Summary
    summary = {
        'mode_b': {
            'val_accuracy': float(val_acc_b),
            'test_accuracy': float(test_acc_b),
            'features': features_b,
            'n_features': len(features_b),
            'n_train': len(X_train_b),
            'n_val': len(X_val_b),
            'n_test': len(X_test_b),
            'pred_test_dist': np.bincount(y_pred_b, minlength=3).tolist(),
            'true_test_dist': np.bincount(y_test_b, minlength=3).tolist(),
        },
        'mode_a': {
            'val_accuracy': float(val_acc_a),
            'test_accuracy': float(test_acc_a),
            'features': features_a,
            'n_features': len(features_a),
            'n_train': len(X_train_a),
            'n_val': len(X_val_a),
            'n_test': len(X_test_a),
            'pred_test_dist': np.bincount(y_pred_a, minlength=3).tolist(),
            'true_test_dist': np.bincount(y_test_a, minlength=3).tolist(),
        },
        'class_weights': class_weight,
    }
    with open(os.path.join(MODELS, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Training complete.")

if __name__ == '__main__':
    main()
