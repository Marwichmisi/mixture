import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS, exist_ok=True)

CLASS_LABELS = ['No watering', 'Short watering', 'Long watering']
SEED = 42

def agronomic_target(row):
    """Expert rule-based target for irrigation decisions."""
    sm = row['soil_moisture']
    at = row['air_temp']
    hu = row['humidity']
    if sm < 20:
        return 2
    elif sm < 30 and (at > 30 or hu < 40):
        return 1
    elif sm < 25:
        return 1
    return 0

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

def train_eval_mode(features, target, feature_names, mode_name, X_raw_for_report=None):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED)
    train_idx, temp_idx = next(sss.split(features, target))
    X_train, X_temp = features[train_idx], features[temp_idx]
    y_train, y_temp = target[train_idx], target[temp_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_idx, test_idx = next(sss2.split(X_temp, y_temp))
    X_val, X_test = X_temp[val_idx], X_temp[test_idx]
    y_val, y_test = y_temp[val_idx], y_temp[test_idx]

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    cls_counts = np.bincount(y_train, minlength=3)
    total = len(y_train)
    class_weight = {i: total / (3 * count) if count > 0 else 1.0 for i, count in enumerate(cls_counts)}

    model = build_model(X_train.shape[1])
    cb = [
        keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True, monitor='val_accuracy'),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, monitor='val_accuracy')
    ]
    model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val),
              epochs=200, batch_size=32, class_weight=class_weight,
              callbacks=cb, verbose=0)

    y_pred = np.argmax(model.predict(X_test_s, verbose=0), axis=1)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    per_class = {}
    for c in range(3):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[CLASS_LABELS[c]] = {
            'precision': float(prec), 'recall': float(rec), 'f1': float(f1),
            'support': int(cm[c, :].sum())
        }

    return {
        'model': model,
        'scaler': scaler,
        'accuracy': float(acc),
        'precision_weighted': float(prec_w),
        'recall_weighted': float(rec_w),
        'f1_weighted': float(f1_w),
        'confusion_matrix': cm.tolist(),
        'per_class': per_class,
        'pred_dist': np.bincount(y_pred, minlength=3).tolist(),
        'true_dist': np.bincount(y_test, minlength=3).tolist(),
        'features': feature_names,
        'n_features': len(feature_names),
        'n_train': len(X_train), 'n_val': len(X_val), 'n_test': len(X_test),
        'class_weight': class_weight,
    }

def plot_confusion_matrix(cm, labels, title, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    print("=" * 60)
    print("  FINAL TRAINING — Agronomic Target Approach")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATA, 'aggregated_6h.csv'), parse_dates=['ts'])
    y = df.apply(agronomic_target, axis=1).values.astype(np.int32)

    print(f"\nSamples: {len(df)}")
    print(f"\nTarget distribution (agronomic rules):")
    for c in [0, 1, 2]:
        pct = (y == c).mean() * 100
        print(f"  {CLASS_LABELS[c]:20s}: {(y == c).sum():4d} ({pct:.1f}%)")

    features_b = ['air_temp', 'humidity', 'pressure', 'soil_moisture',
                  'soil_moisture_trend', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']
    features_a = features_b + ['rain_6h', 'wind_speed', 'et0', 'solar_radiation']

    print(f"\nMode B features ({len(features_b)}): {features_b}")
    X_b = df[features_b].values.astype(np.float32)
    res_b = train_eval_mode(X_b, y, features_b, 'Mode B')

    print(f"\nMode A features ({len(features_a)}): {features_a}")
    X_a = df[features_a].values.astype(np.float32)
    res_a = train_eval_mode(X_a, y, features_a, 'Mode A')

    # Save models
    res_b['model'].save(os.path.join(MODELS, 'final_model_b.keras'))
    res_a['model'].save(os.path.join(MODELS, 'final_model_a.keras'))
    with open(os.path.join(MODELS, 'final_scaler_b.pkl'), 'wb') as f:
        pickle.dump({'scaler': res_b['scaler'], 'features': features_b}, f)
    with open(os.path.join(MODELS, 'final_scaler_a.pkl'), 'wb') as f:
        pickle.dump({'scaler': res_a['scaler'], 'features': features_a}, f)

    # Confusion matrices
    for mode, res, label in [('b', res_b, 'Mode B'), ('a', res_a, 'Mode A')]:
        cm_path = os.path.join(MODELS, f'final_confusion_{mode}.png')
        plot_confusion_matrix(
            res['confusion_matrix'], CLASS_LABELS,
            f'{label} — Matrice de confusion', cm_path
        )

    # Save metrics
    summary = {
        'dataset': {
            'source': 'Stuard IoT Tomato Cultivation (Parma, Italy)',
            'samples': len(df),
            'period': f"{df['ts'].min()} → {df['ts'].max()}",
            'target': 'Agronomic rules: soil_moisture thresholds + stress conditions',
        },
        'mode_b': {
            'accuracy': res_b['accuracy'],
            'precision_weighted': res_b['precision_weighted'],
            'recall_weighted': res_b['recall_weighted'],
            'f1_weighted': res_b['f1_weighted'],
            'per_class': res_b['per_class'],
            'confusion_matrix': res_b['confusion_matrix'],
            'pred_dist': res_b['pred_dist'],
            'true_dist': res_b['true_dist'],
            'features': features_b,
            'n_features': len(features_b),
        },
        'mode_a': {
            'accuracy': res_a['accuracy'],
            'precision_weighted': res_a['precision_weighted'],
            'recall_weighted': res_a['recall_weighted'],
            'f1_weighted': res_a['f1_weighted'],
            'per_class': res_a['per_class'],
            'confusion_matrix': res_a['confusion_matrix'],
            'pred_dist': res_a['pred_dist'],
            'true_dist': res_a['true_dist'],
            'features': features_a,
            'n_features': len(features_a),
        },
    }
    with open(os.path.join(MODELS, 'final_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    for mode_name, res in [('Mode B (9 features)', res_b), ('Mode A (13 features)', res_a)]:
        print(f"\n{mode_name}:")
        print(f"  Accuracy:  {res['accuracy']:.4f} ({res['accuracy']*100:.1f}%)")
        print(f"  F1 (weighted): {res['f1_weighted']:.4f}")
        print(f"  Pred dist: {res['pred_dist']}")
        print(f"  True dist: {res['true_dist']}")
        print(f"\n  Per-class:")
        for label in CLASS_LABELS:
            p = res['per_class'][label]
            print(f"    {label:20s}  P={p['precision']:.3f}  R={p['recall']:.3f}  F1={p['f1']:.3f}  n={p['support']}")

    print(f"\n✅ Models saved to {MODELS}/")
    print(f"   final_model_b.keras (Mode B)")
    print(f"   final_model_a.keras (Mode A)")

if __name__ == '__main__':
    main()
