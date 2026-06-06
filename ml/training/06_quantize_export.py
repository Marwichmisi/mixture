import numpy as np
import pandas as pd
import os
import pickle
import json
import tensorflow as tf

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')
FW = os.path.join(os.path.dirname(__file__), '..', 'firmware')
os.makedirs(FW, exist_ok=True)

CLASS_LABELS = ['No watering', 'Short watering', 'Long watering']

def agronomic_target(row):
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

def quantize_model(model, scaler, features, model_name):
    print(f"\n{'='*60}")
    print(f"  Quantizing {model_name}")
    print(f"{'='*60}")

    # Representative dataset for calibration — must be callable
    df_calib = pd.read_csv(os.path.join(DATA, 'aggregated_6h.csv'))
    X_calib = df_calib[features].values.astype(np.float32)
    X_calib_s = scaler.transform(X_calib)
    n_calib = min(200, len(X_calib))
    indices = np.random.choice(len(X_calib_s), n_calib, replace=False)
    calib_samples = [X_calib_s[i:i+1] for i in indices]

    def representative_dataset():
        for sample in calib_samples:
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path = os.path.join(MODELS, f'{model_name}_int8.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"  Saved: {tflite_path}")
    print(f"  Size:  {len(tflite_model)} bytes ({len(tflite_model)/1024:.1f} KB)")

    return tflite_model

def tflite_to_c_array(tflite_model, var_name):
    hex_bytes = ', '.join(f'0x{b:02x}' for b in tflite_model)
    return f'const unsigned char {var_name}[] = {{\n  {hex_bytes}\n}};\nconst unsigned int {var_name}_len = {len(tflite_model)};'

def scaler_to_c(scaler, features, var_name):
    """Export scaler params as C header."""
    lines = [f'// Scaler parameters for {var_name}']
    lines.append(f'#ifndef SCALER_{var_name.upper()}_H')
    lines.append(f'#define SCALER_{var_name.upper()}_H')
    lines.append('')
    lines.append(f'#define N_FEATURES_{var_name.upper()} {len(features)}')
    lines.append('')

    lines.append(f'const float {var_name}_min[{len(features)}] = {{')
    lines.append('  ' + ', '.join(f'{v:.6f}f' for v in scaler.data_min_))
    lines.append('};')
    lines.append('')

    lines.append(f'const float {var_name}_max[{len(features)}] = {{')
    lines.append('  ' + ', '.join(f'{v:.6f}f' for v in scaler.data_max_))
    lines.append('};')
    lines.append('')

    lines.append(f'const char* {var_name}_feature_names[{len(features)}] = {{')
    lines.append('  ' + ', '.join(f'"{f}"' for f in features))
    lines.append('};')
    lines.append('')

    lines.append(f'#endif // SCALER_{var_name.upper()}_H')
    lines.append('')
    return '\n'.join(lines)

def verify_tflite(tflite_model, scaler, features, model_name):
    """Run a test inference with the quantized model."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\n  Input  dtype: {input_details[0]['dtype']}, shape: {input_details[0]['shape']}")
    print(f"  Output dtype: {output_details[0]['dtype']}, shape: {output_details[0]['shape']}")

    input_scale, input_zero = input_details[0]['quantization']
    output_scale, output_zero = output_details[0]['quantization']
    print(f"  Input quant: scale={input_scale}, zero={input_zero}")
    print(f"  Output quant: scale={output_scale}, zero={output_zero}")

    # Load a few test samples
    df = pd.read_csv(os.path.join(DATA, 'aggregated_6h.csv'))
    y_true = df.apply(agronomic_target, axis=1).values

    correct = 0
    total = 0
    for idx in np.random.choice(len(df), 50, replace=False):
        row = df.iloc[idx]
        x = row[features].values.astype(np.float32)
        x_s = scaler.transform([x])[0]

        # Quantize input
        x_q = (x_s / input_scale + input_zero).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], x_q.reshape(1, -1))
        interpreter.invoke()

        # Dequantize output
        out_q = interpreter.get_tensor(output_details[0]['index'])
        out = (out_q.astype(np.float32) - output_zero) * output_scale
        pred = np.argmax(out[0])

        if pred == y_true[idx]:
            correct += 1
        total += 1

    print(f"  Spot-check accuracy ({total} samples): {correct}/{total} = {correct/total*100:.1f}%")

def export_config(features_b, features_a, scaler_b):
    """Export model config as C header."""
    lines = ['// Configuration for ESP32 irrigation model']
    lines.append('#ifndef CONFIG_IRRIGATION_H')
    lines.append('#define CONFIG_IRRIGATION_H')
    lines.append('')
    lines.append('#define DECISION_INTERVAL_HOURS 6')
    lines.append('')
    lines.append('#define N_FEATURES_MODE_B 9')
    lines.append('#define N_FEATURES_MODE_A 13')
    lines.append('#define N_CLASSES 3')
    lines.append('')
    lines.append('enum IrrigationClass { NO_WATERING = 0, SHORT_WATERING = 1, LONG_WATERING = 2 };')
    lines.append('')
    lines.append('#define CLASS_THRESHOLD_MOISTURE_LOW 20.0f   // < 20% → long watering')
    lines.append('#define CLASS_THRESHOLD_MOISTURE_MID 30.0f    // < 30% + stress → short')
    lines.append('#define CLASS_THRESHOLD_TEMP_STRESS 30.0f     // > 30°C')
    lines.append('#define CLASS_THRESHOLD_HUMIDITY_STRESS 40.0f // < 40%')
    lines.append('')
    lines.append('// Feature order for Mode B')
    lines.append('enum FeatureIndexB {')
    for i, f in enumerate(features_b):
        lines.append(f'  F_{f.upper()} = {i},')
    lines.append('};')
    lines.append('')
    lines.append('#endif // CONFIG_IRRIGATION_H')
    lines.append('')
    return '\n'.join(lines)

def main():
    print("=" * 60)
    print("  TFLite Quantization & Export")
    print("=" * 60)

    # Features
    features_b = ['air_temp', 'humidity', 'pressure', 'soil_moisture',
                  'soil_moisture_trend', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']
    features_a = features_b + ['rain_6h', 'wind_speed', 'et0', 'solar_radiation']

    # Load scalers
    with open(os.path.join(MODELS, 'final_scaler_b.pkl'), 'rb') as f:
        data_b = pickle.load(f)
    scaler_b = data_b['scaler']

    with open(os.path.join(MODELS, 'final_scaler_a.pkl'), 'rb') as f:
        data_a = pickle.load(f)
    scaler_a = data_a['scaler']

    # Load Keras models
    print("\nLoading Keras models...")
    model_b = tf.keras.models.load_model(os.path.join(MODELS, 'final_model_b.keras'))
    model_a = tf.keras.models.load_model(os.path.join(MODELS, 'final_model_a.keras'))

    # --- Quantize Mode B ---
    tflite_b = quantize_model(model_b, scaler_b, features_b, 'model_b')
    verify_tflite(tflite_b, scaler_b, features_b, 'Mode B')

    # --- Quantize Mode A ---
    tflite_a = quantize_model(model_a, scaler_a, features_a, 'model_a')
    verify_tflite(tflite_a, scaler_a, features_a, 'Mode A')

    # --- Export C headers ---
    print(f"\n{'='*60}")
    print("  Exporting C headers")
    print(f"{'='*60}")

    # Model as C array
    c_b = tflite_to_c_array(tflite_b, 'g_model_b')
    with open(os.path.join(FW, 'model_b.h'), 'w') as f:
        f.write(c_b + '\n')
    print(f"  model_b.h: {len(c_b)} bytes")

    c_a = tflite_to_c_array(tflite_a, 'g_model_a')
    with open(os.path.join(FW, 'model_a.h'), 'w') as f:
        f.write(c_a + '\n')
    print(f"  model_a.h: {len(c_a)} bytes")

    # Scaler params
    scaler_b_h = scaler_to_c(scaler_b, features_b, 'scaler_b')
    with open(os.path.join(FW, 'scaler_b.h'), 'w') as f:
        f.write(scaler_b_h + '\n')
    print(f"  scaler_b.h: {len(scaler_b_h)} bytes")

    scaler_a_h = scaler_to_c(scaler_a, features_a, 'scaler_a')
    with open(os.path.join(FW, 'scaler_a.h'), 'w') as f:
        f.write(scaler_a_h + '\n')
    print(f"  scaler_a.h: {len(scaler_a_h)} bytes")

    # Config
    config_h = export_config(features_b, features_a, scaler_b)
    with open(os.path.join(FW, 'config.h'), 'w') as f:
        f.write(config_h + '\n')
    print(f"  config.h: {len(config_h)} bytes")

    print(f"\n{'='*60}")
    print("  FIRMWARE FILES")
    print(f"{'='*60}")
    for f in sorted(os.listdir(FW)):
        path = os.path.join(FW, f)
        size = os.path.getsize(path)
        print(f"  {f:20s}  {size:>6d} bytes")

    # Summary
    print(f"\n{'='*60}")
    print("  EXPORT SUMMARY")
    print(f"{'='*60}")
    print(f"  Mode B TFLite: {len(tflite_b)} bytes ({len(tflite_b)/1024:.1f} KB)")
    print(f"  Mode A TFLite: {len(tflite_a)} bytes ({len(tflite_a)/1024:.1f} KB)")
    print(f"  Features B: {len(features_b)}, Features A: {len(features_a)}")
    print(f"  Output: 3 classes (no/short/long watering)")
    print(f"\n  ✅ Quantization complete. Ready for ESP32 deployment.")

if __name__ == '__main__':
    main()
