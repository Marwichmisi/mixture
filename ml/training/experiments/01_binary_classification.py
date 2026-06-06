import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

df = pd.read_csv('/home/marwane/Documents/mixture/ml/data/aggregated_6h.csv')
df['is_irrigation'] = (df['class'] > 0).astype(int)

mode_b_features = [
    'air_temp', 'humidity', 'pressure', 'soil_moisture',
    'soil_moisture_trend', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos'
]
mode_a_features = mode_b_features + ['rain_6h', 'wind_speed', 'et0', 'solar_radiation']

results = {}

for mode_name, features in [('Mode B (9 sensor features)', mode_b_features),
                             ('Mode A (13 features + weather)', mode_a_features)]:

    X = df[features].values
    y = df['is_irrigation'].values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED)
    train_idx, temp_idx = next(sss.split(X, y))

    X_train, X_temp = X[train_idx], X[temp_idx]
    y_train, y_temp = y[train_idx], y[temp_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_idx, test_idx = next(sss2.split(X_temp, y_temp))

    X_val, X_test = X_temp[val_idx], X_temp[test_idx]
    y_val, y_test = y_temp[val_idx], y_temp[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    class_weight = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}

    model = keras.Sequential([
        layers.Input(shape=(len(features),)),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(2, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, batch_size=16, class_weight=class_weight, verbose=0)

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision_0 = precision_score(y_test, y_pred, pos_label=0)
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    f1_0 = f1_score(y_test, y_pred, pos_label=0)
    precision_1 = precision_score(y_test, y_pred, pos_label=1)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    f1_1 = f1_score(y_test, y_pred, pos_label=1)

    unique_true, counts_true = np.unique(y_test, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    dist_true = dict(zip(unique_true, counts_true))
    dist_pred = dict(zip(unique_pred, counts_pred))

    results[mode_name] = {
        'accuracy': acc, 'confusion_matrix': cm,
        'precision_0': precision_0, 'recall_0': recall_0, 'f1_0': f1_0,
        'precision_1': precision_1, 'recall_1': recall_1, 'f1_1': f1_1,
        'true_dist': dist_true, 'pred_dist': dist_pred
    }

    print(f"{'='*70}")
    print(f"{mode_name}")
    print(f"{'='*70}")
    print(f"\nTrue distribution (test):  class 0: {dist_true.get(0,0)}, class 1: {dist_true.get(1,0)}")
    print(f"Pred distribution (test): class 0: {dist_pred.get(0,0)}, class 1: {dist_pred.get(1,0)}")
    print(f"\nAccuracy:  {acc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Pred 0    Pred 1")
    print(f"  Actual 0    {cm[0,0]:5d}      {cm[0,1]:5d}")
    print(f"  Actual 1    {cm[1,0]:5d}      {cm[1,1]:5d}")
    print(f"\nPer-class metrics:")
    print(f"  Class 0 (no irrigation):  Precision={precision_0:.4f}  Recall={recall_0:.4f}  F1={f1_0:.4f}")
    print(f"  Class 1 (irrigation):     Precision={precision_1:.4f}  Recall={recall_1:.4f}  F1={f1_1:.4f}")
    print(f"\nMacro avg F1: {(f1_0 + f1_1) / 2:.4f}")
    print()

print(f"{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for mode_name, r in results.items():
    print(f"\n{mode_name}:")
    print(f"  Accuracy:     {r['accuracy']:.4f}")
    print(f"  F1 class 0:   {r['f1_0']:.4f}")
    print(f"  F1 class 1:   {r['f1_1']:.4f}")
    print(f"  Macro F1:     {(r['f1_0']+r['f1_1'])/2:.4f}")
    print(f"  Pred dist:    {r['pred_dist']}")
    print(f"  True dist:    {r['true_dist']}")
    print(f"  Confusion:")
    print(r['confusion_matrix'])
