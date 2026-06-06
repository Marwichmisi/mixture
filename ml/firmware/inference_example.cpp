/**
 * inference_example.cpp
 * ESP32 irrigation inference using TFLite Micro (Mode B - offline, 9 features)
 * 
 * Hardware: ESP32 + BME280 + soil moisture sensor
 * Decision: every 6h (00:00, 06:00, 12:00, 18:00)
 * Output: 0 = no watering, 1 = short watering, 2 = long watering
 */

#include <TensorFlowLite_ESP32.h>
#include "model_b.h"
#include "scaler_b.h"
#include "config.h"

// Tensor arena for TFLite Micro
constexpr int kTensorArenaSize = 4 * 1024;  // 4 KB
static uint8_t tensor_arena[kTensorArenaSize];

// TFLite globals
static tflite::MicroMutableOpResolver<10> resolver;
static const tflite::Model* tflite_model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

// Sensor readings (populated by BME280 + soil sensor)
typedef struct {
    float air_temp;           // °C
    float humidity;           // % RH
    float pressure;           // hPa
    float soil_moisture;      // % (0-100)
    float soil_moisture_trend; // % change per hour
    int hour;                 // 0-23
    int weekday;              // 0=Sun, 6=Sat
} sensor_data_t;

static int8_t quantize_value(float value, float scale, int zero_point) {
    return (int8_t)(value / scale + zero_point);
}

static float dequantize_value(int8_t value, float scale, int zero_point) {
    return (float)(value - zero_point) * scale;
}

static void setup_temporal_features(int hour, int weekday,
                                     float* hour_sin, float* hour_cos,
                                     float* weekday_sin, float* weekday_cos) {
    const float two_pi = 2.0f * 3.14159265f;
    *hour_sin = sinf(two_pi * hour / 24.0f);
    *hour_cos = cosf(two_pi * hour / 24.0f);
    *weekday_sin = sinf(two_pi * weekday / 7.0f);
    *weekday_cos = cosf(two_pi * weekday / 7.0f);
}

static bool init_tflite() {
    tflite_model = tflite::GetModel(g_model_b);
    if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        return false;
    }

    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddRelu();
    resolver.AddReshape();

    static tflite::MicroInterpreter static_interpreter(
        tflite_model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        return false;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    return true;
}

static void normalize_input(const sensor_data_t* sensors, float* normalized) {
    int idx = 0;

    // Apply MinMax scaling using params from scaler_b.h
    normalized[idx] = (sensors->air_temp - scaler_b_min[idx])
                    / (scaler_b_max[idx] - scaler_b_min[idx]);
    idx++;

    normalized[idx] = (sensors->humidity - scaler_b_min[idx])
                    / (scaler_b_max[idx] - scaler_b_min[idx]);
    idx++;

    normalized[idx] = (sensors->pressure - scaler_b_min[idx])
                    / (scaler_b_max[idx] - scaler_b_min[idx]);
    idx++;

    normalized[idx] = (sensors->soil_moisture - scaler_b_min[idx])
                    / (scaler_b_max[idx] - scaler_b_min[idx]);
    idx++;

    normalized[idx] = (sensors->soil_moisture_trend - scaler_b_min[idx])
                    / (scaler_b_max[idx] - scaler_b_min[idx]);
    idx++;

    // Temporal features
    float hour_sin, hour_cos, weekday_sin, weekday_cos;
    setup_temporal_features(sensors->hour, sensors->weekday,
                            &hour_sin, &hour_cos, &weekday_sin, &weekday_cos);

    normalized[idx] = (hour_sin - scaler_b_min[idx])
                    / (scaler_b_max[idx] - scaler_b_min[idx]);
    idx++;
    normalized[idx] = (hour_cos - scaler_b_min[idx])
                    / (scaler_b_max[idx] - scaler_b_min[idx]);
    idx++;
    normalized[idx] = (weekday_sin - scaler_b_min[idx])
                    / (scaler_b_max[idx] - scaler_b_min[idx]);
    idx++;
    normalized[idx] = (weekday_cos - scaler_b_min[idx])
                    / (scaler_b_max[idx] - scaler_b_min[idx]);
}

static int run_inference(const sensor_data_t* sensors) {
    float normalized[N_FEATURES_MODE_B];
    normalize_input(sensors, normalized);

    // Quantize input to int8
    float input_scale = input_tensor->params.scale;
    int input_zero = input_tensor->params.zero_point;
    for (int i = 0; i < N_FEATURES_MODE_B; i++) {
        input_tensor->data.int8[i] = quantize_value(normalized[i], input_scale, input_zero);
    }

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        return -1;
    }

    // Dequantize output
    float output_scale = output_tensor->params.scale;
    int output_zero = output_tensor->params.zero_point;
    float probabilities[N_CLASSES];
    for (int i = 0; i < N_CLASSES; i++) {
        probabilities[i] = dequantize_value(output_tensor->data.int8[i],
                                            output_scale, output_zero);
    }

    // Find argmax
    int prediction = 0;
    float max_prob = probabilities[0];
    for (int i = 1; i < N_CLASSES; i++) {
        if (probabilities[i] > max_prob) {
            max_prob = probabilities[i];
            prediction = i;
        }
    }

    return prediction;
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("ESP32 Irrigation Controller");

    if (!init_tflite()) {
        Serial.println("ERROR: Failed to init TFLite");
        while (1);
    }
    Serial.printf("Model loaded: %d bytes\n", g_model_b_len);
    Serial.printf("Tensor arena: %d bytes\n", kTensorArenaSize);
}

void loop() {
    // Get current time from RTC
    // hour = rtc.getHour();
    // weekday = rtc.getDayOfWeek();

    // Read sensors
    // air_temp     = bme.readTemperature();
    // humidity     = bme.readHumidity();
    // pressure     = bme.readPressure() / 100.0f;
    // soil_moisture = analogRead(SOIL_SENSOR_PIN);

    // Compute soil_moisture_trend (slope over last 6h, stored in RTC memory)
    // Store last 6 readings (one per hour) and compute linear regression

    // Example with test values
    sensor_data_t sensors = {
        .air_temp = 32.5f,
        .humidity = 45.0f,
        .pressure = 1010.0f,
        .soil_moisture = 22.0f,
        .soil_moisture_trend = -1.5f,  // drying at 1.5%/hour
        .hour = 14,                    // 2 PM
        .weekday = 3                   // Wednesday
    };

    int decision = run_inference(&sensors);

    Serial.print("Decision: ");
    switch (decision) {
        case NO_WATERING:
            Serial.println("NO WATERING");
            // digitalWrite(VALVE_RELAY, LOW);
            break;
        case SHORT_WATERING:
            Serial.println("SHORT WATERING (15 min)");
            // digitalWrite(VALVE_RELAY, HIGH);
            // delay(15 * 60 * 1000);
            // digitalWrite(VALVE_RELAY, LOW);
            break;
        case LONG_WATERING:
            Serial.println("LONG WATERING (30 min)");
            // digitalWrite(VALVE_RELAY, HIGH);
            // delay(30 * 60 * 1000);
            // digitalWrite(VALVE_RELAY, LOW);
            break;
        default:
            Serial.println("ERROR");
    }

    // Deep sleep until next decision cycle (6h)
    // esp_sleep_enable_timer_wakeup(6 * 3600 * 1000000ULL);
    // esp_deep_sleep_start();

    delay(10000);  // Wait 10s before next inference (for testing)
}
