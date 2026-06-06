/**
 * inference_example_mode_a.cpp
 * ESP32 irrigation inference using TFLite Micro — Mode A (WiFi, 13 features)
 * 
 * Hardware: ESP32 + BME280 + soil moisture sensor + WiFi (Open-Meteo API)
 * Decision: every 6h (00:00, 06:00, 12:00, 18:00)
 * Output: 0 = no watering, 1 = short watering, 2 = long watering
 * 
 * Mode A adds 4 weather features vs Mode B:
 *   rain_6h, wind_speed, et0, solar_radiation
 * These are fetched from Open-Meteo API (free, no key)
 */

#include <TensorFlowLite_ESP32.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "model_a.h"
#include "scaler_a.h"
#include "config.h"

// WiFi credentials
const char* WIFI_SSID = "your_ssid";
const char* WIFI_PASS = "your_password";

// Open-Meteo API (Parma coordinates — update for Benin)
const float LATITUDE = 44.8;
const float LONGITUDE = 10.3;

// Tensor arena
constexpr int kTensorArenaSize = 4 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

// TFLite globals
static tflite::MicroMutableOpResolver<10> resolver;
static const tflite::Model* tflite_model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

// Sensor + weather readings
typedef struct {
    // From BME280
    float air_temp;             // °C
    float humidity;             // % RH
    float pressure;             // hPa
    // From soil sensor
    float soil_moisture;        // % (0-100)
    float soil_moisture_trend;  // % change per hour
    // From RTC
    int hour;                   // 0-23
    int weekday;                // 0=Sun, 6=Sat
    // From Open-Meteo API
    float rain_6h;              // mm (cumulative last 6h)
    float wind_speed;           // km/h
    float et0;                  // mm (evapotranspiration, cumulative 6h)
    float solar_radiation;      // W/m² (mean)
} sensor_data_a_t;

static int8_t quantize_value(float value, float scale, int zero_point) {
    return (int8_t)(value / scale + zero_point);
}

static float dequantize_value(int8_t value, float scale, int zero_point) {
    return (float)(value - zero_point) * scale;
}

static void setup_temporal_features(int hour, int weekday,
                                     float* hour_sin, float* hour_cos,
                                     float* weekday_sin, float* weekday_cos) {
    const float two_pi = 6.283185307f;
    *hour_sin = sinf(two_pi * hour / 24.0f);
    *hour_cos = cosf(two_pi * hour / 24.0f);
    *weekday_sin = sinf(two_pi * weekday / 7.0f);
    *weekday_cos = cosf(two_pi * weekday / 7.0f);
}

static bool init_tflite() {
    tflite_model = tflite::GetModel(g_model_a);
    if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("ERROR: Model schema %d != %d\n",
                      tflite_model->version(), TFLITE_SCHEMA_VERSION);
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
        Serial.println("ERROR: Failed to allocate tensors");
        return false;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.printf("Input shape: %d features\n", input_tensor->dims->data[1]);
    return true;
}

static bool fetch_weather(float* rain_6h, float* wind_speed,
                           float* et0, float* solar_radiation) {
    if (WiFi.status() != WL_CONNECTED) {
        return false;
    }

    HTTPClient http;
    char url[512];
    snprintf(url, sizeof(url),
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=%.1f&longitude=%.1f"
        "&hourly=precipitation,wind_speed_10m,"
        "et0_fao_evapotranspiration,shortwave_radiation"
        "&forecast_days=1"
        "&timezone=UTC",
        LATITUDE, LONGITUDE);

    http.begin(url);
    int code = http.GET();
    if (code != 200) {
        http.end();
        return false;
    }

    String payload = http.getString();
    http.end();

    JsonDocument doc;
    DeserializationError err = deserializeJson(doc, payload);
    if (err) {
        return false;
    }

    // Get current hour's data (index 0 = next full hour)
    JsonArray hourly = doc["hourly"];
    *rain_6h = hourly["precipitation"][0] | 0.0f;
    *wind_speed = hourly["wind_speed_10m"][0] | 0.0f;
    *et0 = hourly["et0_fao_evapotranspiration"][0] | 0.0f;
    *solar_radiation = hourly["shortwave_radiation"][0] | 0.0f;

    return true;
}

static void normalize_input(const sensor_data_a_t* sensors, float* normalized) {
    int idx = 0;

    // Sensor features (indices 0-4)
    normalized[idx] = (sensors->air_temp - scaler_a_min[idx])
                    / (scaler_a_max[idx] - scaler_a_min[idx]);
    idx++;
    normalized[idx] = (sensors->humidity - scaler_a_min[idx])
                    / (scaler_a_max[idx] - scaler_a_min[idx]);
    idx++;
    normalized[idx] = (sensors->pressure - scaler_a_min[idx])
                    / (scaler_a_max[idx] - scaler_a_min[idx]);
    idx++;
    normalized[idx] = (sensors->soil_moisture - scaler_a_min[idx])
                    / (scaler_a_max[idx] - scaler_a_min[idx]);
    idx++;
    normalized[idx] = (sensors->soil_moisture_trend - scaler_a_min[idx])
                    / (scaler_a_max[idx] - scaler_a_min[idx]);
    idx++;

    // Temporal features (indices 5-8)
    float hour_sin, hour_cos, weekday_sin, weekday_cos;
    setup_temporal_features(sensors->hour, sensors->weekday,
                            &hour_sin, &hour_cos, &weekday_sin, &weekday_cos);

    normalized[idx] = (hour_sin - scaler_a_min[idx])
                    / (scaler_a_max[idx] - scaler_a_min[idx]);
    idx++;
    normalized[idx] = (hour_cos - scaler_a_min[idx])
                    / (scaler_a_max[idx] - scaler_a_min[idx]);
    idx++;
    normalized[idx] = (weekday_sin - scaler_a_min[idx])
                    / (scaler_a_max[idx] - scaler_a_min[idx]);
    idx++;
    normalized[idx] = (weekday_cos - scaler_a_min[idx])
                    / (scaler_a_max[idx] - scaler_a_min[idx]);
    idx++;

    // Weather features (indices 9-12)
    normalized[idx] = (sensors->rain_6h - scaler_a_min[idx])
                    / (scaler_a_max[idx] - scaler_a_min[idx]);
    idx++;
    normalized[idx] = (sensors->wind_speed - scaler_a_min[idx])
                    / (scaler_a_max[idx] - scaler_a_min[idx]);
    idx++;
    normalized[idx] = (sensors->et0 - scaler_a_min[idx])
                    / (scaler_a_max[idx] - scaler_a_min[idx]);
    idx++;
    normalized[idx] = (sensors->solar_radiation - scaler_a_min[idx])
                    / (scaler_a_max[idx] - scaler_a_min[idx]);
}

static int run_inference(const sensor_data_a_t* sensors) {
    float normalized[N_FEATURES_MODE_A];
    normalize_input(sensors, normalized);

    float input_scale = input_tensor->params.scale;
    int input_zero = input_tensor->params.zero_point;
    for (int i = 0; i < N_FEATURES_MODE_A; i++) {
        input_tensor->data.int8[i] = quantize_value(normalized[i],
                                                     input_scale, input_zero);
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        return -1;
    }

    float output_scale = output_tensor->params.scale;
    int output_zero = output_tensor->params.zero_point;
    float probabilities[N_CLASSES];
    for (int i = 0; i < N_CLASSES; i++) {
        probabilities[i] = dequantize_value(output_tensor->data.int8[i],
                                            output_scale, output_zero);
    }

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

void setup_wifi() {
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    Serial.print("Connecting to WiFi");
    for (int i = 0; i < 30; i++) {
        if (WiFi.status() == WL_CONNECTED) {
            Serial.println("\nWiFi connected");
            return;
        }
        Serial.print(".");
        delay(500);
    }
    Serial.println("\nWiFi timeout — will use Mode B fallback");
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("ESP32 Irrigation Controller — Mode A");

    if (!init_tflite()) {
        Serial.println("FATAL: TFLite init failed");
        while (1);
    }
    Serial.printf("Model: %d bytes\n", g_model_a_len);
    Serial.printf("Tensor arena: %d bytes\n", kTensorArenaSize);

    setup_wifi();
}

void loop() {
    // Read sensors (replace with actual I2C reads)
    // air_temp     = bme.readTemperature();
    // humidity     = bme.readHumidity();
    // pressure     = bme.readPressure() / 100.0f;
    // soil_moisture = analogRead(SOIL_SENSOR_PIN);
    // hour, weekday from RTC

    sensor_data_a_t sensors = {
        .air_temp = 32.5f,
        .humidity = 45.0f,
        .pressure = 1010.0f,
        .soil_moisture = 22.0f,
        .soil_moisture_trend = -1.5f,
        .hour = 14,
        .weekday = 3,
        .rain_6h = 0.0f,
        .wind_speed = 12.0f,
        .et0 = 0.0f,
        .solar_radiation = 0.0f,
    };

    // Fetch live weather (fallback to 0 if WiFi fails)
    bool weather_ok = (WiFi.status() == WL_CONNECTED)
                    && fetch_weather(&sensors.rain_6h,
                                     &sensors.wind_speed,
                                     &sensors.et0,
                                     &sensors.solar_radiation);
    if (!weather_ok) {
        Serial.println("Weather unavailable — using Mode B values (0)");
    }

    int decision = run_inference(&sensors);

    Serial.printf("Rain: %.1fmm  Wind: %.1fkm/h  ET0: %.2fmm  Solar: %.0fW/m²\n",
                  sensors.rain_6h, sensors.wind_speed,
                  sensors.et0, sensors.solar_radiation);
    Serial.printf("Decision: %s\n",
                  decision == NO_WATERING ? "NO WATERING" :
                  decision == SHORT_WATERING ? "SHORT WATERING (15 min)" :
                  decision == LONG_WATERING ? "LONG WATERING (30 min)" :
                  "ERROR");

    // Deep sleep 6h
    // esp_sleep_enable_timer_wakeup(6 * 3600 * 1000000ULL);
    // esp_deep_sleep_start();

    delay(10000);
}
