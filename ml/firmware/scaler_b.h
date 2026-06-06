// Scaler parameters for scaler_b
#ifndef SCALER_SCALER_B_H
#define SCALER_SCALER_B_H

#define N_FEATURES_SCALER_B 9

const float scaler_b_min[9] = {
  11.602858f, 20.867647f, 994.743774f, 14.153429f, -3.978648f, -0.707107f, -0.707107f, -0.974928f, -0.900969f
};

const float scaler_b_max[9] = {
  41.909676f, 95.742859f, 1018.745178f, 54.737713f, 4.464367f, 0.707107f, 0.707107f, 0.974928f, 1.000000f
};

const char* scaler_b_feature_names[9] = {
  "air_temp", "humidity", "pressure", "soil_moisture", "soil_moisture_trend", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"
};

#endif // SCALER_SCALER_B_H

