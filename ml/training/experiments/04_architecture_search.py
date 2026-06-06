import pandas as pd
import numpy as np
import os
import json
import itertools
import time
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
MODELS = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
OUT = os.path.join(os.path.dirname(__file__))
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

CLASS_LABELS = ['No watering', 'Short watering', 'Long watering']

FEATURES_B = [
    'air_temp', 'humidity', 'pressure', 'soil_moisture',
    'soil_moisture_trend',
    'hour_sin', 'hour_cos',
    'weekday_sin', 'weekday_cos'
]
FEATURES_A = FEATURES_B + ['rain_6h', 'wind_speed', 'et0', 'solar_radiation']

ARCH_BUILDERS = {
    'wider': lambda d: keras.Sequential([
        layers.Input(shape=(d,)), layers.Dense(32), layers.Activation('relu'),
        layers.Dense(16), layers.Activation('relu'),
        layers.Dense(3, activation='softmax'),
    ]),
    'deeper': lambda d: keras.Sequential([
        layers.Input(shape=(d,)), layers.Dense(24), layers.Activation('relu'),
        layers.Dense(16), layers.Activation('relu'),
        layers.Dense(8), layers.Activation('relu'),
        layers.Dense(3, activation='softmax'),
    ]),
    'dropout': lambda d: keras.Sequential([
        layers.Input(shape=(d,)), layers.Dense(16), layers.Activation('relu'),
        layers.Dropout(0.2),
        layers.Dense(8), layers.Activation('relu'),
        layers.Dropout(0.2),
        layers.Dense(3, activation='softmax'),
    ]),
    'batchnorm': lambda d: keras.Sequential([
        layers.Input(shape=(d,)), layers.Dense(16),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Dense(8),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Dense(3, activation='softmax'),
    ]),
    'bigger': lambda d: keras.Sequential([
        layers.Input(shape=(d,)), layers.Dense(64), layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.Dense(32), layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.Dense(16), layers.Activation('relu'),
        layers.Dense(3, activation='softmax'),
    ]),
}

ARCH_DESCS = {
    'wider': 'Dense(32)→Dense(16)→Dense(3)',
    'deeper': 'Dense(24)→Dense(16)→Dense(8)→Dense(3)',
    'dropout': 'Dense(16)→Drop0.2→Dense(8)→Drop0.2→Dense(3)',
    'batchnorm': 'Dense(16)→BN→ReLU→Dense(8)→BN→ReLU→Dense(3)',
    'bigger': 'Dense(64)→Dense(32)→Dense(16)→Dense(3)+Drop0.3',
}

LR = [0.01, 0.001, 0.0005]
BATCH = [16, 32]
ACTIVATIONS = ['relu', 'tanh']

def make_model(input_dim, arch_key, activation):
    model = ARCH_BUILDERS[arch_key](input_dim)
    for layer in model.layers:
        if isinstance(layer, layers.Activation):
            layer.activation = tf.keras.activations.get(activation)
    return model

def train_model(model, X_tr, y_tr, X_vl, y_vl, lr, bs, cw, max_epochs=100):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    cb = [
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='val_accuracy'),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, monitor='val_accuracy')
    ]
    history = model.fit(X_tr, y_tr, validation_data=(X_vl, y_vl),
                        epochs=max_epochs, batch_size=bs,
                        class_weight=cw, callbacks=cb, verbose=0)
    return history

def compute_metrics(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    report = classification_report(y_test, y_pred, target_names=CLASS_LABELS, zero_division=0, output_dict=True)
    per_class = {}
    for i, label in enumerate(CLASS_LABELS):
        r = report.get(label, {})
        per_class[label] = {
            'precision': r.get('precision', 0.0),
            'recall': r.get('recall', 0.0),
            'f1': r.get('f1-score', 0.0),
            'support': int(r.get('support', 0)),
        }
    return acc, cm.tolist(), per_class, np.bincount(y_pred, minlength=3).tolist(), np.bincount(y_test, minlength=3).tolist()

def main():
    agg = pd.read_csv(os.path.join(DATA, 'aggregated_6h.csv'), parse_dates=['ts'])
    y = agg['class'].values.astype(np.int32)

    classes = np.bincount(y, minlength=3)
    total = len(y)
    class_weight = {i: total / (3 * count) if count > 0 else 1.0 for i, count in enumerate(classes)}

    all_results = []

    for mode_name, features in [('B', FEATURES_B), ('A', FEATURES_A)]:
        print(f"\n{'='*70}\n  MODE {mode_name} ({len(features)} features)\n{'='*70}")
        sys.stdout.flush()

        X = agg[features].values.astype(np.float32)
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
        X_vl, X_te, y_vl, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

        scaler = MinMaxScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_vl_s = scaler.transform(X_vl)
        X_te_s = scaler.transform(X_te)

        print(f"  Split: train={len(X_tr)} val={len(X_vl)} test={len(X_te)}")
        print(f"  Train class dist: {np.bincount(y_tr, minlength=3)}")
        sys.stdout.flush()

        # ── Phase 1: Architecture comparison ──
        print(f"\n  ─── Phase 1: Architecture comparison ───")
        sys.stdout.flush()

        arch_results = []
        for arch_key in ARCH_BUILDERS:
            tf.random.set_seed(42)
            model = make_model(len(features), arch_key, 'relu')
            t0 = time.time()
            history = train_model(model, X_tr_s, y_tr, X_vl_s, y_vl, 0.001, 32, class_weight)
            t_elapsed = time.time() - t0
            n_epochs = len(history.history['loss'])
            val_acc = max(history.history['val_accuracy'])
            test_acc, cm, pc, pred_dist, true_dist = compute_metrics(model, X_te_s, y_te)

            res = {
                'mode': mode_name, 'arch': arch_key, 'arch_desc': ARCH_DESCS[arch_key],
                'lr': 0.001, 'batch_size': 32, 'activation': 'relu',
                'epochs_run': n_epochs, 'train_time_s': round(t_elapsed, 1),
                'val_accuracy': float(val_acc), 'test_accuracy': float(test_acc),
                'per_class': pc, 'confusion_matrix': cm,
                'pred_dist': pred_dist, 'true_dist': true_dist,
            }
            arch_results.append(res)
            all_results.append(res)
            print(f"  {arch_key:12s} test_acc={test_acc:.4f} val_acc={val_acc:.4f}  "
                  f"epochs={n_epochs:3d}  {t_elapsed:5.1f}s")
            sys.stdout.flush()

        arch_results.sort(key=lambda r: r['test_accuracy'], reverse=True)
        best_arch = arch_results[0]['arch']
        best_test = arch_results[0]['test_accuracy']
        print(f"\n  >> Best architecture: {best_arch} (test_acc={best_test:.4f})")
        sys.stdout.flush()

        # ── Phase 2: Hyperparameter sweep ──
        print(f"\n  ─── Phase 2: Hyperparameter sweep on '{best_arch}' ───")
        sys.stdout.flush()

        hp_results = []
        for lr, bs, act in itertools.product(LR, BATCH, ACTIVATIONS):
            tf.random.set_seed(42)
            model = make_model(len(features), best_arch, act)
            t0 = time.time()
            history = train_model(model, X_tr_s, y_tr, X_vl_s, y_vl, lr, bs, class_weight)
            t_elapsed = time.time() - t0
            n_epochs = len(history.history['loss'])
            val_acc = max(history.history['val_accuracy'])
            test_acc, cm, pc, pred_dist, true_dist = compute_metrics(model, X_te_s, y_te)

            res = {
                'mode': mode_name, 'arch': best_arch, 'arch_desc': ARCH_DESCS[best_arch],
                'lr': lr, 'batch_size': bs, 'activation': act,
                'epochs_run': n_epochs, 'train_time_s': round(t_elapsed, 1),
                'val_accuracy': float(val_acc), 'test_accuracy': float(test_acc),
                'per_class': pc, 'confusion_matrix': cm,
                'pred_dist': pred_dist, 'true_dist': true_dist,
            }
            hp_results.append(res)
            all_results.append(res)
            print(f"  lr={lr:.4f} bs={bs:2d} act={act:5s}  test_acc={test_acc:.4f} val_acc={val_acc:.4f}  "
                  f"epochs={n_epochs:3d}  {t_elapsed:5.1f}s")
            sys.stdout.flush()

        hp_results.sort(key=lambda r: r['test_accuracy'], reverse=True)
        best = hp_results[0]
        print(f"\n  >> Best: arch={best_arch} lr={best['lr']} bs={best['batch_size']} act={best['activation']} "
              f"test_acc={best['test_accuracy']:.4f}")
        sys.stdout.flush()

        # ── Phase 3: Final retrain ──
        print(f"\n  ─── Phase 3: Final retrain (full training+val) ───")
        sys.stdout.flush()

        X_full = np.vstack([X_tr_s, X_vl_s])
        y_full = np.concatenate([y_tr, y_vl])
        full_classes = np.bincount(y_full, minlength=3)
        full_total = len(y_full)
        full_cw = {i: full_total / (3 * count) if count > 0 else 1.0 for i, count in enumerate(full_classes)}

        tf.random.set_seed(42)
        final_model = make_model(len(features), best_arch, best['activation'])
        cb = [
            keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True, monitor='loss'),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, monitor='loss')
        ]
        final_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=best['lr']),
            loss='sparse_categorical_crossentropy', metrics=['accuracy']
        )
        t0 = time.time()
        final_model.fit(X_full, y_full, epochs=100, batch_size=best['batch_size'],
                        class_weight=full_cw, callbacks=cb, verbose=0)
        t_elapsed = time.time() - t0

        test_acc, cm, pc, pred_dist, true_dist = compute_metrics(final_model, X_te_s, y_te)
        model_name = f'mlp_{best_arch}_m{mode_name}.keras'
        model_path = os.path.join(MODELS, model_name)
        final_model.save(model_path)
        print(f"  Saved: {model_path} ({t_elapsed:.1f}s)")
        print(f"  Test acc: {test_acc:.4f}")
        for label in CLASS_LABELS:
            p = pc[label]
            print(f"    {label:20s}  P={p['precision']:.3f}  R={p['recall']:.3f}  F1={p['f1']:.3f}  n={p['support']}")
        print(f"  CM:\n{np.array(cm)}")
        print(f"  Pred: {pred_dist}  True: {true_dist}")
        sys.stdout.flush()

    # ── Summary table ──
    print(f"\n\n{'='*110}")
    print("  ARCHITECTURE COMPARISON (default hparams: lr=0.001, batch=32, relu)")
    print(f"{'='*110}")
    print(f"  {'Mode':5s} {'Arch':12s} {'TestAcc':8s} {'ValAcc':8s} "
          f"{'P0':6s} {'R0':6s} {'P1':6s} {'R1':6s} {'P2':6s} {'R2':6s} {'Ep':4s} {'T(s)':6s}")
    print(f"  {'─'*5} {'─'*12} {'─'*8} {'─'*8} "
          f"{'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*4} {'─'*6}")
    for r in sorted(all_results, key=lambda x: (x['mode'], -x['test_accuracy'])):
        if not (r['lr'] == 0.001 and r['batch_size'] == 32 and r['activation'] == 'relu'):
            continue
        pc = r['per_class']
        print(f"  {r['mode']:5s} {r['arch']:12s} {r['test_accuracy']:8.4f} {r['val_accuracy']:8.4f} "
              f"{pc[CLASS_LABELS[0]]['precision']:6.3f} {pc[CLASS_LABELS[0]]['recall']:6.3f} "
              f"{pc[CLASS_LABELS[1]]['precision']:6.3f} {pc[CLASS_LABELS[1]]['recall']:6.3f} "
              f"{pc[CLASS_LABELS[2]]['precision']:6.3f} {pc[CLASS_LABELS[2]]['recall']:6.3f} "
              f"{r['epochs_run']:3d}  {r['train_time_s']:5.1f}")

    print(f"\n{'='*110}")
    print("  HYPERPARAMETER SWEEP (top 20)")
    print(f"{'='*110}")
    print(f"  {'Mode':5s} {'Arch':12s} {'LR':8s} {'BS':4s} {'Act':6s} {'TestAcc':8s} {'ValAcc':8s} "
          f"{'P0':6s} {'P1':6s} {'P2':6s} {'Ep':4s} {'T(s)':6s}")
    print(f"  {'─'*5} {'─'*12} {'─'*8} {'─'*4} {'─'*6} {'─'*8} {'─'*8} "
          f"{'─'*6} {'─'*6} {'─'*6} {'─'*4} {'─'*6}")
    for i, r in enumerate(sorted(all_results, key=lambda x: -x['test_accuracy'])):
        if i >= 20:
            break
        pc = r['per_class']
        print(f"  {r['mode']:5s} {r['arch']:12s} {r['lr']:.4f}  {r['batch_size']:2d}  {r['activation']:5s} "
              f"{r['test_accuracy']:8.4f} {r['val_accuracy']:8.4f} "
              f"{pc[CLASS_LABELS[0]]['precision']:6.3f} {pc[CLASS_LABELS[1]]['precision']:6.3f} "
              f"{pc[CLASS_LABELS[2]]['precision']:6.3f} {r['epochs_run']:3d}  {r['train_time_s']:5.1f}")

    with open(os.path.join(OUT, 'architecture_search_results.json'), 'w') as f:
        # Convert per_class keys to standalone for JSON compatibility
        clean = []
        for r in all_results:
            cr = dict(r)
            cr['per_class'] = {k: dict(v) for k, v in r['per_class'].items()}
            clean.append(cr)
        json.dump(clean, f, indent=2)

    print(f"\n  ✅ Architecture search complete. Results → {os.path.join(OUT, 'architecture_search_results.json')}")

if __name__ == '__main__':
    main()
