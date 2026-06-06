// Scaler parameters for scaler_a
#ifndef SCALER_SCALER_A_H
#define SCALER_SCALER_A_H

#define N_FEATURES_SCALER_A 13

const float scaler_a_min[13] = {
  11.602858f, 20.867647f, 994.743774f, 14.153429f, -3.978648f, -0.707107f, -0.707107f, -0.974928f, -0.900969f, 0.000000f, 2.212121f, 0.000000f, 0.000000f
};

const float scaler_a_max[13] = {
  41.909676f, 95.742859f, 1018.745178f, 54.737713f, 4.464367f, 0.707107f, 0.707107f, 0.974928f, 1.000000f, 264.600006f, 24.382353f, 20.260000f, 654.166687f
};

const char* scaler_a_feature_names[13] = {
  "air_temp", "humidity", "pressure", "soil_moisture", "soil_moisture_trend", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "rain_6h", "wind_speed", "et0", "solar_radiation"
};

#endif // SCALER_SCALER_A_H

