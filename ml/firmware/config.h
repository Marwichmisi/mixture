// Configuration for ESP32 irrigation model
#ifndef CONFIG_IRRIGATION_H
#define CONFIG_IRRIGATION_H

#define DECISION_INTERVAL_HOURS 6

#define N_FEATURES_MODE_B 9
#define N_FEATURES_MODE_A 13
#define N_CLASSES 3

enum IrrigationClass { NO_WATERING = 0, SHORT_WATERING = 1, LONG_WATERING = 2 };

#define CLASS_THRESHOLD_MOISTURE_LOW 20.0f   // < 20% → long watering
#define CLASS_THRESHOLD_MOISTURE_MID 30.0f    // < 30% + stress → short
#define CLASS_THRESHOLD_TEMP_STRESS 30.0f     // > 30°C
#define CLASS_THRESHOLD_HUMIDITY_STRESS 40.0f // < 40%

// Feature order for Mode B
enum FeatureIndexB {
  F_AIR_TEMP = 0,
  F_HUMIDITY = 1,
  F_PRESSURE = 2,
  F_SOIL_MOISTURE = 3,
  F_SOIL_MOISTURE_TREND = 4,
  F_HOUR_SIN = 5,
  F_HOUR_COS = 6,
  F_WEEKDAY_SIN = 7,
  F_WEEKDAY_COS = 8,
};

#endif // CONFIG_IRRIGATION_H

