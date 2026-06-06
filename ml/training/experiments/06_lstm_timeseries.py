import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

MAX_STEPS = 36
STEP_FEATURES = ['air_temp', 'humidity', 'pressure', 'soil_moisture', 'hour_sin', 'hour_cos']

# ─── 1. Load and prepare data ───────────────────────────────────────────────
df = pd.read_csv('/home/marwane/Documents/mixture/ml/data/merged_dataset.csv', parse_dates=['ts'])
df.columns = df.columns.str.strip()

# rename for consistency
df = df.rename(columns={'air_humidity': 'humidity'})

# time features per reading
ts = pd.to_datetime(df['ts'])
hour = ts.dt.hour + ts.dt.minute / 60.0
df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)

# ─── 2. Create 6h calendar-aligned windows ──────────────────────────────────
def create_sequences(df_line, max_steps=MAX_STEPS):
    df_line = df_line.sort_values('ts').copy()
    ts = pd.to_datetime(df_line['ts'])
    window_start = ts.dt.floor('6h')
    df_line['window_start'] = window_start

    sequences = []
    labels = []
    seq_info = []

    for ws, group in df_line.groupby('window_start', sort=True):
        group = group.sort_values('ts')
        seq = group[STEP_FEATURES].values.astype(np.float64)

        # pad or truncate
        if len(seq) > max_steps:
            seq = seq[:max_steps]
        elif len(seq) < max_steps:
            pad = np.zeros((max_steps - len(seq), len(STEP_FEATURES)), dtype=np.float64)
            seq = np.vstack([seq, pad])

        volume_increase = float(group['current_volume'].max()) - float(group['current_volume'].min())

        if volume_increase <= 1.0:
            cls = 0
        elif volume_increase <= 530.0:
            cls = 1
        else:
            cls = 2

        sequences.append(seq)
        labels.append(cls)
        seq_info.append((ws, len(group), volume_increase, cls))

    return np.array(sequences), np.array(labels), seq_info


all_sequences = []
all_labels = []
all_info = []

for line in sorted(df['line'].unique()):
    df_line = df[df['line'] == line].copy()
    seqs, lbls, info = create_sequences(df_line)
    all_sequences.append(seqs)
    all_labels.append(lbls)
    all_info.extend(info)
    dist = np.bincount(lbls, minlength=3)
    print(f"Line {line}: {len(seqs)} sequences, class dist = [{dist[0]}, {dist[1]}, {dist[2]}]")

X = np.vstack(all_sequences)
y = np.concatenate(all_labels)

print(f"\nTotal sequences: {len(X)}")
print(f"Sequence shape: {X.shape}")
print(f"Steps per sequence: {X.shape[1]}, Features per step: {X.shape[2]}")
dist = np.bincount(y, minlength=3)
print(f"Class distribution: 0={dist[0]}, 1={dist[1]}, 2={dist[2]}")
print(f"  (%): 0={dist[0]/len(y)*100:.1f}%, 1={dist[1]/len(y)*100:.1f}%, 2={dist[2]/len(y)*100:.1f}%")

# ─── 3. Stratified split (preserve sequence-level class distribution) ──────
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED)
train_idx, temp_idx = next(sss.split(np.zeros(len(y)), y))

X_train, X_temp = X[train_idx], X[temp_idx]
y_train, y_temp = y[train_idx], y[temp_idx]

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
val_idx, test_idx = next(sss2.split(np.zeros(len(y_temp)), y_temp))

X_val, X_test = X_temp[val_idx], X_temp[test_idx]
y_val, y_test = y_temp[val_idx], y_temp[test_idx]

print(f"\nSplit sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

# ─── 4. Scale features per-step ────────────────────────────────────────────
# Reshape to (n_samples * n_steps, n_features), scale, reshape back
n_train, n_steps, n_feats = X_train.shape
X_train_2d = X_train.reshape(-1, n_feats)
X_val_2d = X_val.reshape(-1, n_feats)
X_test_2d = X_test.reshape(-1, n_feats)

scaler = StandardScaler()
X_train_2d = scaler.fit_transform(X_train_2d)
X_val_2d = scaler.transform(X_val_2d)
X_test_2d = scaler.transform(X_test_2d)

X_train = X_train_2d.reshape(-1, n_steps, n_feats)
X_val = X_val_2d.reshape(-1, n_steps, n_feats)
X_test = X_test_2d.reshape(-1, n_steps, n_feats)

# ─── 5. Class weights ──────────────────────────────────────────────────────
n_total = len(y_train)
class_counts = np.bincount(y_train, minlength=3)
class_weight = {i: n_total / (len(np.unique(y_train)) * c) for i, c in enumerate(class_counts) if c > 0}
print(f"\nClass weights: {class_weight}")

# ─── 6. Model definitions ──────────────────────────────────────────────────
def build_lstm():
    model = keras.Sequential([
        layers.Input(shape=(MAX_STEPS, len(STEP_FEATURES))),
        layers.Masking(mask_value=0.0),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(16),
        layers.Dense(8, activation='relu'),
        layers.Dense(3, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_gru():
    model = keras.Sequential([
        layers.Input(shape=(MAX_STEPS, len(STEP_FEATURES))),
        layers.Masking(mask_value=0.0),
        layers.GRU(32, return_sequences=True),
        layers.GRU(16),
        layers.Dense(8, activation='relu'),
        layers.Dense(3, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_bilstm():
    model = keras.Sequential([
        layers.Input(shape=(MAX_STEPS, len(STEP_FEATURES))),
        layers.Masking(mask_value=0.0),
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(16)),
        layers.Dense(8, activation='relu'),
        layers.Dense(3, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_small_lstm():
    model = keras.Sequential([
        layers.Input(shape=(MAX_STEPS, len(STEP_FEATURES))),
        layers.Masking(mask_value=0.0),
        layers.LSTM(16, return_sequences=True),
        layers.LSTM(8),
        layers.Dense(3, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

models = {
    'LSTM(32→16)': build_lstm,
    'GRU(32→16)': build_gru,
    'BiLSTM(32→16)': build_bilstm,
    'LSTM(16→8)': build_small_lstm,
}

# ─── 7. Train and evaluate each model ──────────────────────────────────────
results = {}

for model_name, build_fn in models.items():
    print(f"\n{'='*70}")
    print(f"  Training: {model_name}")
    print(f"{'='*70}")

    model = build_fn()
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=0
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100, batch_size=16,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=0
    )

    best_epoch = early_stop.stopped_epoch - early_stop.patience + 1 if early_stop.stopped_epoch else 100
    val_loss = min(history.history['val_loss'])

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # per-class metrics
    per_class = {}
    for c in sorted(np.unique(y_test)):
        per_class[c] = {
            'precision': precision_score(y_test, y_pred, labels=[c], average='macro', zero_division=0),
            'recall': recall_score(y_test, y_pred, labels=[c], average='macro', zero_division=0),
            'f1': f1_score(y_test, y_pred, labels=[c], average='macro', zero_division=0),
        }

    unique_true, counts_true = np.unique(y_test, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    dist_true = dict(zip(unique_true, counts_true))
    dist_pred = dict(zip(unique_pred, counts_pred))

    results[model_name] = {
        'accuracy': acc,
        'confusion_matrix': cm,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'per_class': per_class,
        'true_dist': dist_true,
        'pred_dist': dist_pred,
        'best_epoch': best_epoch,
        'val_loss': val_loss,
    }

    print(f"  Best epoch: {best_epoch} | Val loss: {val_loss:.4f}")
    print(f"  Test accuracy: {acc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                Pred 0    Pred 1    Pred 2")
    print(f"    Actual 0    {cm[0,0]:5d}      {cm[0,1]:5d}      {cm[0,2]:5d}")
    if cm.shape[0] > 1:
        print(f"    Actual 1    {cm[1,0]:5d}      {cm[1,1]:5d}      {cm[1,2]:5d}")
    if cm.shape[0] > 2:
        print(f"    Actual 2    {cm[2,0]:5d}      {cm[2,1]:5d}      {cm[2,2]:5d}")
    print(f"\n  Per-class metrics:")
    for c in sorted(per_class.keys()):
        print(f"    Class {c}: P={per_class[c]['precision']:.4f}  R={per_class[c]['recall']:.4f}  F1={per_class[c]['f1']:.4f}")
    print(f"\n  Macro avg: P={precision_macro:.4f}  R={recall_macro:.4f}  F1={f1_macro:.4f}")
    print(f"  True dist: {dist_true}")
    print(f"  Pred dist: {dist_pred}")

# ─── 8. Summary comparison ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  MODEL COMPARISON SUMMARY")
print(f"{'='*70}")
print(f"\n{'Model':<20} {'Acc':<8} {'F1(macro)':<12} {'P(macro)':<12} {'R(macro)':<12} {'Epochs':<8}")
print(f"{'-'*72}")
for model_name in models:
    r = results[model_name]
    print(f"{model_name:<20} {r['accuracy']:<8.4f} {r['f1_macro']:<12.4f} {r['precision_macro']:<12.4f} {r['recall_macro']:<12.4f} {r['best_epoch']:<8}")

print(f"\n{'='*70}")
print("  PER-CLASS F1 COMPARISON")
print(f"{'='*70}")
print(f"\n{'Model':<20} {'F1 class 0':<12} {'F1 class 1':<12} {'F1 class 2':<12}")
print(f"{'-'*56}")
for model_name in models:
    r = results[model_name]
    f1s = [r['per_class'][c]['f1'] for c in sorted(r['per_class'].keys())]
    print(f"{model_name:<20} {f1s[0]:<12.4f} {f1s[1]:<12.4f} {f1s[2]:<12.4f}")

print(f"\n{'='*70}")
print("  PREDICTED vs TRUE DISTRIBUTION")
print(f"{'='*70}")
print(f"{'Model':<20} {'True dist':<20} {'Pred dist':<20}")
print(f"{'-'*60}")
for model_name in models:
    r = results[model_name]
    td = str(r['true_dist'])
    pd_str = str(r['pred_dist'])
    print(f"{model_name:<20} {td:<20} {pd_str:<20}")

# ─── 9. Confusion matrices ──────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  CONFUSION MATRICES")
print(f"{'='*70}")
for model_name in models:
    cm = results[model_name]['confusion_matrix']
    print(f"\n  {model_name}:")
    print(f"      Pred 0  Pred 1  Pred 2")
    for i, row in enumerate(cm):
        print(f"  Act {i}  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}")

# ─── 10. Sequence stats ─────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  SEQUENCE STATISTICS")
print(f"{'='*70}")
print(f"  Sequence length (steps): {MAX_STEPS}")
print(f"  Features per step:        {len(STEP_FEATURES)}")
print(f"  Total sequences:          {len(X)}")
window_lengths = [info[1] for info in all_info]
print(f"  Avg readings/window:      {np.mean(window_lengths):.1f}")
print(f"  Min readings/window:      {np.min(window_lengths)}")
print(f"  Max readings/window:      {np.max(window_lengths)}")
print(f"  Padded sequences (len<{MAX_STEPS}): {sum(1 for wl in window_lengths if wl < MAX_STEPS)}/{len(window_lengths)}")
print(f"  Truncated sequences:      {sum(1 for wl in window_lengths if wl > MAX_STEPS)}/{len(window_lengths)}")

print(f"\n{'='*70}")
print("  COMPARISON TO BASELINE MLP (01_binary_classification.py)")
print(f"{'='*70}")
print("""
  NOTE: Direct comparison is approximate because:
  - MLP used 2 classes (binary: irrigation vs no irrigation)
  - LSTM uses 3 classes (none, small, large irrigation)
  - MLP used 6h-aggregated features + weather + engineered features
  - LSTM uses only raw per-step sensor readings (6 features)
  - MLP used Mode B (9 features) and Mode A (13 features + weather)
  - LSTM uses only: air_temp, humidity, pressure, soil_moisture, hour_sin, hour_cos

  Key difference: LSTM learns temporal patterns within the 6h window,
  while MLP sees only aggregated statistics (mean, trend).

  LSTM macro F1 vs MLP binary F1 (conceptual improvement expected
  from modeling temporal structure).
""")
