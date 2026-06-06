import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import tensorflow as tf
from tensorflow import keras

DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
EXPERIMENTS = os.path.join(os.path.dirname(__file__))

CLASS_LABELS = ['No watering', 'Short watering', 'Long watering']
SEEDS = [42, 123, 456, 789, 1111]

features_b = [
    'air_temp', 'humidity', 'pressure', 'soil_moisture',
    'soil_moisture_trend',
    'hour_sin', 'hour_cos',
    'weekday_sin', 'weekday_cos'
]
features_a = features_b + ['rain_6h', 'wind_speed', 'et0', 'solar_radiation']


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


def focal_loss(gamma=2.0, alpha=None):
    def loss(y_true, y_pred):
        y_true_c = tf.cast(y_true, tf.int32)
        y_true_o = tf.one_hot(y_true_c, depth=3)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true_o * tf.math.log(y_pred)
        pt = tf.where(tf.equal(y_true_o, 1.0), y_pred, 1.0 - y_pred)
        focal_weight = tf.pow(1.0 - pt, gamma)
        loss = focal_weight * cross_entropy
        if alpha is not None:
            alpha_t = tf.constant([alpha.get(i, 1.0) for i in range(3)], dtype=tf.float32)
            alpha_w = tf.reduce_sum(y_true_o * alpha_t, axis=-1)
            loss = loss * tf.expand_dims(alpha_w, -1)
        return tf.reduce_sum(loss, axis=-1)
    return loss


def evaluate(model, X_test, y_test):
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_LABELS, zero_division=0, output_dict=True)
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
        'true_dist': np.bincount(y_test, minlength=3).tolist(),
    }


def train_and_eval(X_train, y_train, X_val, y_val, X_test, y_test,
                   input_dim, approach_name, mode_name, focal=False,
                   alpha_weights=None):
    callbacks = [
        keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True, monitor='val_accuracy'),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, monitor='val_accuracy'),
    ]

    if focal:
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])
        loss_fn = focal_loss(gamma=2.0, alpha=alpha_weights)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss_fn,
            metrics=['accuracy']
        )
    else:
        model = build_model(input_dim)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200, batch_size=32,
        callbacks=callbacks, verbose=0
    )

    val_results = evaluate(model, X_val, y_val)
    test_results = evaluate(model, X_test, y_test)
    return test_results


def main():
    agg = pd.read_csv(os.path.join(DATA, 'aggregated_6h.csv'), parse_dates=['ts'])
    y_all = agg['class'].values.astype(np.int32)

    results = {}

    for mode_name, features in [('B', features_b), ('A', features_a)]:
        X_all = agg[features].values.astype(np.float32)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
        )

        scaler = MinMaxScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        print(f"\n{'='*60}")
        print(f"Mode {mode_name} — {len(features)} features")
        print(f"{'='*60}")
        print(f"Train dist: {np.bincount(y_train, minlength=3)}")
        print(f"Val dist:   {np.bincount(y_val, minlength=3)}")
        print(f"Test dist:  {np.bincount(y_test, minlength=3)}")

        mode_results = {}

        # ── Approach 1: SMOTE ──
        print("\n  [1/4] SMOTE oversampling...")
        smote = SMOTE(random_state=42, sampling_strategy='auto')
        X_tr_sm, y_tr_sm = smote.fit_resample(X_train_s, y_train)
        print(f"    After SMOTE: {np.bincount(y_tr_sm, minlength=3)}")
        mode_results['smote'] = train_and_eval(
            X_tr_sm, y_tr_sm, X_val_s, y_val, X_test_s, y_test,
            len(features), 'SMOTE', mode_name
        )
        print(f"    Test acc: {mode_results['smote']['accuracy']:.4f}  "
              f"Pred: {mode_results['smote']['pred_dist']}")

        # ── Approach 2: SMOTETomek ──
        print("\n  [2/4] SMOTE + Tomek links...")
        smt = SMOTETomek(random_state=42, sampling_strategy='auto')
        X_tr_smt, y_tr_smt = smt.fit_resample(X_train_s, y_train)
        print(f"    After SMOTETomek: {np.bincount(y_tr_smt, minlength=3)}")
        mode_results['smote_tomek'] = train_and_eval(
            X_tr_smt, y_tr_smt, X_val_s, y_val, X_test_s, y_test,
            len(features), 'SMOTETomek', mode_name
        )
        print(f"    Test acc: {mode_results['smote_tomek']['accuracy']:.4f}  "
              f"Pred: {mode_results['smote_tomek']['pred_dist']}")

        # ── Approach 3: Ensemble of 5 ──
        print("\n  [3/4] Ensemble (5 models, bootstrap, soft voting)...")
        all_probs = []
        for i, seed in enumerate(SEEDS):
            n = len(X_train_s)
            idx = np.random.RandomState(seed).choice(n, size=int(n * 0.8), replace=True)
            X_boot = X_train_s[idx]
            y_boot = y_train[idx]

            model_i = build_model(len(features))
            cb_i = [
                keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='val_accuracy'),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, monitor='val_accuracy'),
            ]
            model_i.fit(
                X_boot, y_boot,
                validation_data=(X_val_s, y_val),
                epochs=150, batch_size=32,
                callbacks=cb_i, verbose=0
            )
            probs = model_i.predict(X_test_s, verbose=0)
            all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0)
        y_pred_ens = np.argmax(avg_probs, axis=1)

        acc = accuracy_score(y_test, y_pred_ens)
        prec = precision_score(y_test, y_pred_ens, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred_ens, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_ens, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred_ens)
        report = classification_report(y_test, y_pred_ens, target_names=CLASS_LABELS, zero_division=0, output_dict=True)
        per_class = {}
        for i, label in enumerate(CLASS_LABELS):
            key = label
            per_class[key] = {
                'precision': report[key]['precision'] if key in report else 0.0,
                'recall': report[key]['recall'] if key in report else 0.0,
                'f1': report[key]['f1-score'] if key in report else 0.0,
                'support': int(report[key]['support']) if key in report else 0,
            }
        mode_results['ensemble'] = {
            'accuracy': float(acc),
            'precision_weighted': float(prec),
            'recall_weighted': float(rec),
            'f1_weighted': float(f1),
            'confusion_matrix': cm.tolist(),
            'per_class': per_class,
            'pred_dist': np.bincount(y_pred_ens, minlength=3).tolist(),
            'true_dist': np.bincount(y_test, minlength=3).tolist(),
        }
        print(f"    Test acc: {mode_results['ensemble']['accuracy']:.4f}  "
              f"Pred: {mode_results['ensemble']['pred_dist']}")

        # ── Approach 4: Focal Loss ──
        print("\n  [4/4] Focal Loss (gamma=2.0)...")
        classes = np.bincount(y_train, minlength=3)
        total = len(y_train)
        alpha_weights = {i: total / (3 * count) if count > 0 else 1.0
                         for i, count in enumerate(classes)}

        mode_results['focal_loss'] = train_and_eval(
            X_train_s, y_train, X_val_s, y_val, X_test_s, y_test,
            len(features), 'FocalLoss', mode_name,
            focal=True, alpha_weights=alpha_weights
        )
        print(f"    Test acc: {mode_results['focal_loss']['accuracy']:.4f}  "
              f"Pred: {mode_results['focal_loss']['pred_dist']}")

        results[mode_name] = mode_results

    # ── Print comparison table ──
    print("\n\n" + "="*80)
    print("COMPARISON TABLE — All Approaches")
    print("="*80)

    header = f"{'Approach':<16} {'Mode':<6} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'Pred dist':<20} {'True dist':<20}"
    print(header)
    print("-" * 80)

    for mode_name in ['B', 'A']:
        for app_name, app_label in [('smote', 'SMOTE'), ('smote_tomek', 'SMOTETomek'),
                                     ('ensemble', 'Ensemble'), ('focal_loss', 'FocalLoss')]:
            r = results[mode_name][app_name]
            pd_str = str(r['pred_dist'])
            td_str = str(r['true_dist'])
            print(f"{app_label:<16} {mode_name:<6} {r['accuracy']:<8.4f} {r['precision_weighted']:<8.4f} "
                  f"{r['recall_weighted']:<8.4f} {r['f1_weighted']:<8.4f} "
                  f"{pd_str:<20} {td_str:<20}")

    # ── Per-class detail ──
    print("\n\n" + "="*80)
    print("PER-CLASS METRICS — Best approach per mode")
    print("="*80)

    for mode_name in ['B', 'A']:
        best_app = max(results[mode_name].items(),
                       key=lambda x: (x[1]['per_class']['Short watering']['recall'] +
                                      x[1]['per_class']['Long watering']['recall']))
        best_name, best_res = best_app
        print(f"\nMode {mode_name} — Best: {best_name}")
        print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
        print("-" * 60)
        for label in CLASS_LABELS:
            p = best_res['per_class'][label]
            print(f"{label:<20} {p['precision']:<10.3f} {p['recall']:<10.3f} {p['f1']:<10.3f} {p['support']:<10}")
        print(f"\nAccuracy:  {best_res['accuracy']:.4f}")
        print(f"Pred dist: {best_res['pred_dist']}")
        print(f"True dist: {best_res['true_dist']}")

    # ── Save comparison JSON ──
    save_path = os.path.join(EXPERIMENTS, '05_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {save_path}")


if __name__ == '__main__':
    main()
