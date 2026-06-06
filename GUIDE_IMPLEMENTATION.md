# Guide d'Implémentation — Arrosage de Précision ESP32

> 📖 **Navigation :** [`README.md`](README.md) ← [`PRD`](PRD_Arrosage_Precision_ESP32.md) ← [`Plan ML`](PLAN_ML_DETAIL.md) ← [`Rapport`](RAPPORT_FINAL.md) ← **Guide implémentation**

**Public :** Techniciens, makers, ingénieurs déploiement
**Niveau :** Intermédiaire (Arduino/C++ de base)
**Temps estimé :** 2-4 heures

---

## Table des matières

1. [Prérequis matériel](#1-prérequis-matériel)
2. [Schéma de câblage](#2-schéma-de-câblage)
3. [Installation logicielle](#3-installation-logicielle)
4. [Structure du projet](#4-structure-du-projet)
5. [Comprendre les fichiers fournis](#5-comprendre-les-fichiers-fournis)
6. [Code complet étape par étape](#6-code-complet-étape-par-étape)
7. [Les deux modes de fonctionnement](#7-les-deux-modes-de-fonctionnement)
8. [Tests et calibration](#8-tests-et-calibration)
9. [Déploiement terrain](#9-déploiement-terrain)
10. [Dépannage](#10-dépannage)

---

## 1. Prérequis matériel

### Liste des composants

| Composant | Prix indicatif | Utilité | Important ? |
|-----------|---------------|---------|-------------|
| **ESP32** (NodeMCU-32S ou WROOM-32) | ~8€ | Cerveau du système | ✅ Obligatoire |
| **BME280** (module I2C) | ~8€ | Température, humidité, pression | ✅ Obligatoire |
| **Capteur d'humidité du sol capacitif** | ~3€ | Humidité du sol | ✅ Obligatoire |
| **Relais 1 canal** (5V, actif bas) | ~2€ | Commande électrovanne | ✅ Obligatoire |
| **Électrovanne 12V/24V** (½ pouce) | ~15€ | Ouvre/ferme l'eau | ✅ Obligatoire |
| **Alimentation 5V/2A** (secteur) | ~5€ | Alimentation ESP32 | ✅ Obligatoire (mains powered) |
| **DS18B20** (optionnel) | ~3€ | Température du sol | ⭐ Recommandé |
| **Capteur de pluie** (optionnel) | ~5€ | Détection pluie | ⭐ Recommandé |
| **RTC DS3231** (optionnel) | ~4€ | Horloge temps réel précise | ⭐ Recommandé |

**Coût total minimal :** ~26€ (ESP32 + BME280 + sol + relais + vanne + alim)
**Coût avec options :** ~38€

### Brochage ESP32

```
ESP32 DevKit V1 (38 pins)
┌──────────────────────────────────────┐
│ USB     EN                     D23  │
│         ┌─┐                    D22  │── SCL (BME280)
│ 3.3V ───┤ ├─── Alim capteurs   D21  │── SDA (BME280)
│ GND  ───┤ ├─── Masse commune   D19  │── DS18B20 (optionnel)
│ 5V   ───┤ ├─── Alim vanne      D18  │
│         └─┘                    D5   │── Capteur sol (analogique)
│                                 D4  │
│ D15                                D0 │
│ D13 ─── Relais (vanne)             D2 │
│ D12                                D15│
│ RX0                                TX0│
│                                     ──┘
└──────────────────────────────────────┘
```

### Schéma de câblage détaillé

```
                   ┌─────────────────────────────┐
                   │         ESP32               │
                   │                             │
  BME280           │  3.3V ─────┐                │
  ┌──────┐         │  GND  ─────┤                │
  │ VCC ─┼─────────┤            │                │
  │ GND ─┼─────────┤            │                │
  │ SCL ─┼─────────┤ D22        │                │
  │ SDA ─┼─────────┤ D21        │                │
  └──────┘         │            │                │
                    │            │                │
  Capteur sol       │            │                │
  ┌──────────┐      │  3.3V ────┘                │
  │ VCC ─────┼──────┤                            │
  │ GND ─────┼──────┤ GND                        │
  │ SIG ─────┼──────┤ D5 (ADC1)  (ou D34/D35)    │
  └──────────┘      │                            │
                     │                            │
  Relais             │                            │
  ┌──────┐           │                            │
  │ VCC ─┼───────────┤ 5V (ou VIN)               │
  │ GND ─┼───────────┤ GND                        │
  │ IN  ─┼───────────┤ D13                        │
  │ NO  ─┼───────────┤───┐                        │
  └──────┘           │   │                        │
                      │   │                        │
  Alimentation        │   │   Électrovanne         │
  ┌────────┐          │   │   ┌──────────┐        │
  │ 5V ────┼──────────┤   │   │ 12V ─────┼────────┤
  │ GND ───┼──────────┤   └───┤ GND ─────┼────────┤
  └────────┘          │       └──────────┘        │
                       └─────────────────────────────┘
```

**⚠️ Attention :** 
- L'ESP32 et l'électrovanne DOIVENT partager la même masse (GND)
- Le capteur capacitif se branche sur une entrée ADC (D5, D34, D35 ou D36)
- Le relais est en **actif bas** (LOW = vanne ouverte, HIGH = vanne fermée)

---

## 2. Installation logicielle

### Option A : PlatformIO (recommandé)

```bash
# 1. Installer PlatformIO dans VS Code
# Extension → PlatformIO IDE

# 2. Créer un nouveau projet
pio init --board esp32dev

# 3. Ajouter les dépendances dans platformio.ini
```

**`platformio.ini` :**
```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 115200

lib_deps =
    adafruit/Adafruit BME280 Library
    adafruit/Adafruit Unified Sensor
    tensorflow/tensorflow-lite-esp32
    adafruit/DHT sensor library       # si tu utilises DHT22
    paulstoffregen/OneWire            # si DS18B20
    milesburton/DallasTemperature     # si DS18B20

build_flags =
    -DCORE_DEBUG_LEVEL=0
    -DARDUINO_ARCH_ESP32
```

### Option B : Arduino IDE

1. Installer le support ESP32 dans l'Arduino IDE
2. Bibliothèques → Gérer les bibliothèques :
   - `Adafruit BME280 Library`
   - `TensorFlowLite_ESP32`
3. Télécharger ce projet et copier `ml/firmware/` dans ton dossier de sketch

---

## 3. Structure du projet

```
mixture/
│
├── README.md                     ← Page d'accueil du projet
├── PRD_Arrosage_Precision_ESP32.md  ← Cahier des charges
├── PLAN_ML_DETAIL.md             ← Plan détaillé du ML
├── RAPPORT_FINAL.md              ← Résultats d'entraînement
├── GUIDE_IMPLEMENTATION.md       ← ⬅️ CE GUIDE
│
└── ml/
    ├── data/                     ← Données (fusionnées, agrégées, météo)
    │   ├── merged_dataset.csv
    │   ├── aggregated_6h.csv
    │   └── weather_parma_2023.csv
    │
    ├── training/                 ← Scripts Python d'entraînement
    │   ├── 01_merge_datasets.py
    │   ├── 02_fetch_weather.py
    │   ├── 03_feature_engineering.py
    │   ├── final_train.py
    │   ├── 06_quantize_export.py
    │   └── experiments/          ← Approches testées
    │       ├── 01_binary_classification.py
    │       ├── 02_agronomic_target.py
    │       ├── 03_feature_engineering_v2.py
    │       ├── 04_architecture_search.py
    │       ├── 05_smote_ensemble.py
    │       └── 06_lstm_timeseries.py
    │
    ├── models/                   ← Modèles entraînés
    │   ├── final_model_b.keras
    │   ├── final_model_a.keras
    │   ├── model_b_int8.tflite        ← ⬅️ Modèle quantifié pour ESP32
    │   ├── model_a_int8.tflite
    │   ├── final_scaler_b.pkl
    │   ├── final_scaler_a.pkl
    │   ├── final_confusion_b.png
    │   ├── final_confusion_a.png
    │   ├── final_metrics.json
    │   ├── evaluation_report.md
    │   └── training_summary.json
    │
    └── firmware/                 ← ⬅️ Fichiers à copier dans l'ESP32
        ├── config.h              ← Configuration (features, classes, seuils)
        ├── model_b.h             ← Modèle B en tableau C (21 KB)
        ├── model_a.h             ← Modèle A en tableau C (22 KB)
        ├── scaler_b.h            ← Paramètres de normalisation Mode B
        ├── scaler_a.h            ← Paramètres de normalisation Mode A
        └── inference_example.cpp ← ⬅️ Exemple complet d'inférence
```

---

## 4. Comprendre les fichiers fournis

### `config.h` — Configuration générale

Ce fichier contient les constantes partagées :

```cpp
#define DECISION_INTERVAL_HOURS 6   // Décision toutes les 6h
#define N_FEATURES_MODE_B 9         // 9 entrées pour Mode B
#define N_FEATURES_MODE_A 13        // 13 entrées pour Mode A
#define N_CLASSES 3                 // 3 sorties possibles

// Les 3 classes de décision
enum IrrigationClass {
    NO_WATERING = 0,     // Ne pas arroser
    SHORT_WATERING = 1,  // Arroser 15 minutes
    LONG_WATERING = 2    // Arroser 30 minutes
};

// Seuils agronomiques (pour info / fallback)
#define THRESHOLD_MOISTURE_LOW 20.0f   // < 20% → long arrosage
#define THRESHOLD_MOISTURE_MID 30.0f   // < 30% + stress → court
#define THRESHOLD_TEMP_STRESS 30.0f    // > 30°C = stress
#define THRESHOLD_HUMIDITY_STRESS 40.0f // < 40% = stress
```

À quoi ça sert ? Quand tu changes le nombre de features ou de classes, tu modifies ce fichier.

### `model_b.h` — Le modèle lui-même

C'est le réseau de neurones **quantifié en int8** converti en tableau C.

```cpp
// Début du fichier (créé automatiquement par 06_quantize_export.py)
const unsigned char g_model_b[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, ...
};
const unsigned int g_model_b_len = 3560;
```

**⚠️ Ne JAMAIS modifier ce fichier à la main.** Si tu veux changer le modèle, relance `06_quantize_export.py`.

### `scaler_b.h` — Normalisation des entrées

```cpp
#define N_FEATURES_MODE_B 9

// Minimum de chaque feature (appris sur les données d'entraînement)
const float scaler_b_min[9] = {
    10.100000f,   // air_temp     (°C)
    18.000000f,   // humidity     (%)
    993.700012f,  // pressure     (hPa)
    0.000000f,    // soil_moisture (%)
    -7.821043f,   // soil_moisture_trend (%/h)
    -1.000000f,   // hour_sin
    -1.000000f,   // hour_cos
    -1.000000f,   // weekday_sin
    -1.000000f    // weekday_cos
};

// Maximum de chaque feature
const float scaler_b_max[9] = {
    43.700001f,   // air_temp
    97.000000f,   // humidity
    1019.400024f, // pressure
    56.799999f,   // soil_moisture
    5.903357f,    // soil_moisture_trend
    1.000000f,    // hour_sin
    1.000000f,    // hour_cos
    1.000000f,    // weekday_sin
    1.000000f     // weekday_cos
};
```

La normalisation se fait avec la formule :
```
normalisé = (valeur_brute - min) / (max - min)
```

### `inference_example.cpp` — Exemple complet

C'est un programme Arduino/ESP32 complet qui montre comment :
1. Initialiser TFLite Micro
2. Lire les capteurs
3. Normaliser les entrées
4. Quantifier en int8
5. Lancer l'inférence
6. Interpréter le résultat
7. Contrôler la vanne

---

## 5. Code complet étape par étape

### Étape 1 : Inclure les fichiers

```cpp
#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>

// Modèle et configuration
#include "model_b.h"        // ou model_a.h si tu utilises le WiFi
#include "scaler_b.h"       // ou scaler_a.h
#include "config.h"

// Capteurs
#include <Wire.h>
#include <Adafruit_BME280.h>
```

### Étape 2 : Définir les broches

```cpp
// Brochage
#define PIN_BME280_SCL  22
#define PIN_BME280_SDA  21
#define PIN_SOIL_MOISTURE  5    // ADC1 (GPIO5)
#define PIN_RELAY_VALVE  13     // Relais vanne
#define PIN_LED_STATUS   2      // LED intégrée ESP32

// Seuils de l'humidité du sol (à calibrer !)
#define SOIL_DRY_AIR     1200  // Valeur ADC quand le capteur est à l'air libre
#define SOIL_WET_WATER   800   // Valeur ADC quand le capteur est dans l'eau
```

### Étape 3 : Initialiser TFLite Micro

```cpp
// Le "tensor arena" = mémoire pour les calculs du réseau de neurones
constexpr int kTensorArenaSize = 4 * 1024;  // 4 KB
static uint8_t tensor_arena[kTensorArenaSize];

// Variables globales TFLite
static tflite::MicroMutableOpResolver<10> resolver;
static const tflite::Model* tflite_model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

bool initTFLite() {
    // Charger le modèle depuis le tableau C (model_b.h)
    tflite_model = tflite::GetModel(g_model_b);
    if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERREUR: Version du modèle incompatible !");
        Serial.print("  Modèle: v"); Serial.print(tflite_model->version());
        Serial.print(", Attendu: v"); Serial.println(TFLITE_SCHEMA_VERSION);
        return false;
    }

    // Enregistrer les opérations utilisées par le modèle
    // (notre MLP utilise: FullyConnected, Softmax, Relu)
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();

    // Créer l'interpréteur
    static tflite::MicroInterpreter static_interpreter(
        tflite_model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allouer la mémoire pour les tenseurs
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERREUR: Impossible d'allouer les tenseurs !");
        return false;
    }

    // Récupérer les pointeurs entrée/sortie
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.println("✅ TFLite initialisé !");
    Serial.print("  Entrée:  "); Serial.print(input_tensor->dims->data[1]);
    Serial.print(" features, type: int8\n");
    Serial.print("  Sortie:  "); Serial.print(output_tensor->dims->data[1]);
    Serial.print(" classes, type: int8\n");
    Serial.print("  Taille modèle: "); Serial.print(g_model_b_len);
    Serial.print(" bytes\n");

    return true;
}
```

### Étape 4 : Lire les capteurs

```cpp
Adafruit_BME280 bme;

bool initCapteurs() {
    Wire.begin(PIN_BME280_SDA, PIN_BME280_SCL);
    if (!bme.begin(0x76, &Wire)) {
        Serial.println("ERREUR: BME280 non trouvé !");
        Serial.println("  Vérifie le cablage (SCL→D22, SDA→D21)");
        return false;
    }
    Serial.println("✅ BME280 OK");
    return true;
}

float lireHumiditeSol() {
    // Lecture analogique (0-4095 sur ESP32)
    int raw = analogRead(PIN_SOIL_MOISTURE);

    // Conversion en pourcentage (0% = sec, 100% = dans l'eau)
    // Les valeurs SOIL_DRY_AIR et SOIL_WET_WATER sont à calibrer
    float pct = map(raw, SOIL_DRY_AIR, SOIL_WET_WATER, 0, 100);
    pct = constrain(pct, 0.0f, 100.0f);

    return pct;
}

float calculerTendanceHumidite(float nouvelle_valeur) {
    // Stocker les 6 dernières lectures (1 par heure)
    static float historique[6] = {0};
    static int index = 0;

    historique[index] = nouvelle_valeur;
    index = (index + 1) % 6;

    // Calculer la pente sur les valeurs non-nulles
    float somme_x = 0, somme_y = 0, somme_xy = 0, somme_xx = 0;
    int n = 0;
    for (int i = 0; i < 6; i++) {
        if (historique[i] != 0) {
            somme_x += i;
            somme_y += historique[i];
            somme_xy += i * historique[i];
            somme_xx += i * i;
            n++;
        }
    }

    if (n < 3) return 0.0f;

    // Pente = (n*Σxy - Σx*Σy) / (n*Σxx - (Σx)²)
    float pente = (n * somme_xy - somme_x * somme_y)
                / (n * somme_xx - somme_x * somme_x);

    // Convertir en % par heure
    return pente;
}
```

### Étape 5 : Normaliser et quantifier les entrées

```cpp
struct SensorData {
    float air_temp;            // de BME280
    float humidity;            // de BME280
    float pressure;            // de BME280
    float soil_moisture;       // du capteur sol
    float soil_moisture_trend; // calculé (pente)
    int hour;                  // de RTC ou millis()
    int weekday;               // 0=dimanche, 6=samedi
};

void preparerEntree(SensorData* data) {
    // Étape 1 : Normalisation (MinMax)
    float normalise[N_FEATURES_MODE_B];

    normalise[0] = (data->air_temp - scaler_b_min[0])
                 / (scaler_b_max[0] - scaler_b_min[0]);

    normalise[1] = (data->humidity - scaler_b_min[1])
                 / (scaler_b_max[1] - scaler_b_min[1]);

    normalise[2] = (data->pressure - scaler_b_min[2])
                 / (scaler_b_max[2] - scaler_b_min[2]);

    normalise[3] = (data->soil_moisture - scaler_b_min[3])
                 / (scaler_b_max[3] - scaler_b_min[3]);

    normalise[4] = (data->soil_moisture_trend - scaler_b_min[4])
                 / (scaler_b_max[4] - scaler_b_min[4]);

    // Features temporelles (encodage cyclique)
    const float PI = 3.14159265f;
    float hour_sin = sin(2 * PI * data->hour / 24.0f);
    float hour_cos = cos(2 * PI * data->hour / 24.0f);
    float wday_sin = sin(2 * PI * data->weekday / 7.0f);
    float wday_cos = cos(2 * PI * data->weekday / 7.0f);

    normalise[5] = (hour_sin - scaler_b_min[5])
                 / (scaler_b_max[5] - scaler_b_min[5]);
    normalise[6] = (hour_cos - scaler_b_min[6])
                 / (scaler_b_max[6] - scaler_b_min[6]);
    normalise[7] = (wday_sin - scaler_b_min[7])
                 / (scaler_b_max[7] - scaler_b_min[7]);
    normalise[8] = (wday_cos - scaler_b_min[8])
                 / (scaler_b_max[8] - scaler_b_min[8]);

    // Étape 2 : Quantification int8
    // L'ESP32 attend des int8, pas des float !
    float input_scale = input_tensor->params.scale;
    int input_zero = input_tensor->params.zero_point;

    for (int i = 0; i < N_FEATURES_MODE_B; i++) {
        // Formule : value_int8 = value_float / scale + zero_point
        float q = normalise[i] / input_scale + input_zero;
        input_tensor->data.int8[i] = (int8_t)constrain(q, -128, 127);
    }
}
```

### Étape 6 : Exécuter l'inférence

```cpp
int lancerInference() {
    // Vérifier que TFLite est prêt
    if (interpreter == nullptr) {
        Serial.println("ERREUR: TFLite non initialisé !");
        return -1;
    }

    // Exécuter le réseau de neurones
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("ERREUR: Échec de l'inférence !");
        return -1;
    }

    // Lire la sortie
    float output_scale = output_tensor->params.scale;
    int output_zero = output_tensor->params.zero_point;

    // Déquantifier les 3 sorties
    float probabilites[3];
    for (int i = 0; i < N_CLASSES; i++) {
        // Formule : value_float = (value_int8 - zero_point) * scale
        probabilites[i] = (output_tensor->data.int8[i] - output_zero)
                        * output_scale;
    }

    // Afficher les probabilités
    Serial.print("  Pas d'arrosage:  ");
    Serial.println(probabilites[0], 4);
    Serial.print("  Arrosage court:  ");
    Serial.println(probabilites[1], 4);
    Serial.print("  Arrosage long:   ");
    Serial.println(probabilites[2], 4);

    // Choisir la classe avec la plus haute probabilité (argmax)
    int prediction = 0;
    float max_prob = probabilites[0];
    for (int i = 1; i < N_CLASSES; i++) {
        if (probabilites[i] > max_prob) {
            max_prob = probabilites[i];
            prediction = i;
        }
    }

    return prediction;
}
```

### Étape 7 : Contrôler la vanne

```cpp
void controlerVanne(int decision) {
    switch (decision) {
        case NO_WATERING:
            Serial.println("🔴 DÉCISION: Pas d'arrosage");
            digitalWrite(PIN_RELAY_VALVE, HIGH);  // Relais actif bas → HIGH = fermé
            digitalWrite(PIN_LED_STATUS, LOW);
            break;

        case SHORT_WATERING:
            Serial.println("🟡 DÉCISION: Arrosage court (15 min)");
            digitalWrite(PIN_RELAY_VALVE, LOW);   // LOW = ouvert
            digitalWrite(PIN_LED_STATUS, HIGH);
            delay(15 * 60 * 1000);                // 15 minutes
            digitalWrite(PIN_RELAY_VALVE, HIGH);  // Refermer
            digitalWrite(PIN_LED_STATUS, LOW);
            Serial.println("  ✅ Arrosage court terminé");
            break;

        case LONG_WATERING:
            Serial.println("🟢 DÉCISION: Arrosage long (30 min)");
            digitalWrite(PIN_RELAY_VALVE, LOW);   // LOW = ouvert
            digitalWrite(PIN_LED_STATUS, HIGH);
            delay(30 * 60 * 1000);                // 30 minutes
            digitalWrite(PIN_RELAY_VALVE, HIGH);  // Refermer
            digitalWrite(PIN_LED_STATUS, LOW);
            Serial.println("  ✅ Arrosage long terminé");
            break;

        default:
            Serial.println("ERREUR: Décision invalide !");
            digitalWrite(PIN_RELAY_VALVE, HIGH);  // Sécurité : fermer la vanne
    }
}
```

### Étape 8 : Programme complet `setup()` et `loop()`

```cpp
// ============================================================
//  VARIABLES GLOBALES
// ============================================================

// Pour la tendance de l'humidité du sol
unsigned long dernier_reveil = 0;
float humidite_precedente = 0;

// ============================================================
//  SETUP — Exécuté une fois au démarrage
// ============================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n\n=== SYSTÈME D'ARROSAGE INTELLIGENT ===");
    Serial.println("Démarrage...");

    // Configurer les broches
    pinMode(PIN_RELAY_VALVE, OUTPUT);
    pinMode(PIN_LED_STATUS, OUTPUT);
    digitalWrite(PIN_RELAY_VALVE, HIGH);  // Vanne fermée par défaut
    digitalWrite(PIN_LED_STATUS, LOW);

    // Initialiser les capteurs
    if (!initCapteurs()) {
        Serial.println("❌ Arrêt: capteurs non trouvés !");
        while (1) { delay(1000); }
    }

    // Initialiser TFLite
    if (!initTFLite()) {
        Serial.println("❌ Arrêt: TFLite non initialisé !");
        while (1) { delay(1000); }
    }

    Serial.println("\n✅ Système prêt !");
    Serial.print("  Modèle: Mode B (");
    Serial.print(g_model_b_len);
    Serial.print(" bytes, ");
    Serial.print(N_FEATURES_MODE_B);
    Serial.print(" features)\n");
    Serial.println("  Prochaine décision dans 6h");
}

// ============================================================
//  LOOP — Exécuté en boucle
// ============================================================
void loop() {
    Serial.println("\n--- Nouveau cycle de décision ---");

    // 1. Lire les capteurs
    float temp = bme.readTemperature();
    float hum = bme.readHumidity();
    float pres = bme.readPressure() / 100.0f;
    float sol = lireHumiditeSol();
    float tendance = calculerTendanceHumidite(sol);

    Serial.println("Données capteurs :");
    Serial.print("  Température:       "); Serial.print(temp); Serial.println(" °C");
    Serial.print("  Humidité air:      "); Serial.print(hum); Serial.println(" %");
    Serial.print("  Pression:          "); Serial.print(pres); Serial.println(" hPa");
    Serial.print("  Humidité sol:      "); Serial.print(sol); Serial.println(" %");
    Serial.print("  Tendance sol:      "); Serial.print(tendance, 2); Serial.println(" %/h");

    // 2. Obtenir l'heure (à remplacer par RTC)
    // Exemple avec millis() pour le test (à NE PAS utiliser en production)
    int hour = (millis() / 3600000) % 24;  // À remplacer par rtc.getHour()
    int weekday = 3;                       // À remplacer par rtc.getDayOfWeek()

    // 3. Préparer les données pour le modèle
    SensorData data = { temp, hum, pres, sol, tendance, hour, weekday };
    preparerEntree(&data);

    // 4. Lancer l'inférence
    Serial.println("Exécution du modèle...");
    int decision = lancerInference();

    // 5. Afficher le résultat
    if (decision >= 0 && decision < 3) {
        const char* classes[] = {"Pas d'arrosage", "Arrosage court", "Arrosage long"};
        Serial.print("✅ Décision finale: ");
        Serial.println(classes[decision]);

        // 6. Contrôler la vanne
        controlerVanne(decision);
    }

    // 7. Attendre la prochaine décision (6h)
    Serial.println("\n--- Cycle terminé, attente 6h ---");
    Serial.flush();

    // Deep sleep
    // esp_sleep_enable_timer_wakeup(6 * 3600 * 1000000ULL);  // 6h en microsecondes
    // esp_deep_sleep_start();

    // Pour le test (sans deep sleep) : attendre 30 secondes
    delay(30000);
}
```

---

## 6. Les deux modes de fonctionnement

### Mode B — Offline (9 features)

Utilise uniquement les capteurs locaux. Pas besoin de WiFi.

```cpp
// Mode B — fichier à inclure
#include "model_b.h"    // Modèle 9 features
#include "scaler_b.h"   // Normalisation 9 features

// 9 features à fournir :
// 1. temperature     → BME280
// 2. humidite_air    → BME280
// 3. pression        → BME280
// 4. humidite_sol    → capteur sol
// 5. tendance_sol    → calculée (pente 6h)
// 6. heure_sin       → RTC
// 7. heure_cos       → RTC
// 8. jour_sin        → RTC
// 9. jour_cos        → RTC
```

**Taille modèle :** 3.5 KB — **recommandé pour la plupart des cas**

### Mode A — WiFi (13 features)

Nécessite une connexion WiFi pour récupérer les données météo via Open-Meteo.

```cpp
// Mode A — fichier à inclure
#include "model_a.h"    // Modèle 13 features
#include "scaler_a.h"   // Normalisation 13 features

// 13 features : 9 du Mode B + 4 météo
// 10. pluie_6h        → Open-Meteo API
// 11. vent            → Open-Meteo API
// 12. evapotransp.    → Open-Meteo API
// 13. rayonnement     → Open-Meteo API
```

#### Exemple d'appel API Open-Meteo (sur ESP32)

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// Coordonnées du Bénin (à ajuster selon ta localisation)
#define LATITUDE  6.5f    // Exemple: Cotonou
#define LONGITUDE 2.5f

struct WeatherData {
    float rain_6h;
    float wind_speed;
    float et0;
    float solar_radiation;
};

bool fetchWeather(WeatherData* weather) {
    HTTPClient http;
    WiFiClient client;

    char url[256];
    snprintf(url, sizeof(url),
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=%.1f&longitude=%.1f"
        "&hourly=precipitation,wind_speed_10m,"
        "et0_fao_evapotranspiration,shortwave_radiation"
        "&forecast_hours=6&timezone=UTC",
        LATITUDE, LONGITUDE);

    http.begin(client, url);
    int code = http.GET();

    if (code != 200) {
        Serial.printf("Météo: erreur HTTP %d\n", code);
        http.end();
        return false;
    }

    String body = http.getString();
    http.end();

    // Analyser le JSON
    JsonDocument doc;
    DeserializationError err = deserializeJson(doc, body);
    if (err) {
        Serial.println("Météo: JSON invalide");
        return false;
    }

    JsonArray hourly = doc["hourly"];
    weather->rain_6h = 0;
    weather->wind_speed = 0;
    weather->et0 = 0;
    weather->solar_radiation = 0;

    // Somme/cumul sur les 6 prochaines heures
    for (int h = 0; h < 6; h++) {
        weather->rain_6h += hourly["precipitation"][h].as<float>();
        weather->wind_speed += hourly["wind_speed_10m"][h].as<float>();
        weather->et0 += hourly["et0_fao_evapotranspiration"][h].as<float>();
        weather->solar_radiation += hourly["shortwave_radiation"][h].as<float>();
    }
    weather->wind_speed /= 6.0f;  // moyenne
    weather->solar_radiation /= 6.0f;

    return true;
}
```

### Basculement automatique entre Mode A et Mode B

```cpp
void loop() {
    bool wifi_ok = (WiFi.status() == WL_CONNECTED);

    if (wifi_ok) {
        Serial.println("📶 WiFi disponible → Mode A (météo)");
        // Utiliser model_a.h et scaler_a.h
        // 13 features incluant les données météo
        WeatherData meteo;
        if (fetchWeather(&meteo)) {
            // Ajouter les 4 features météo aux 9 features capteurs
            preparerEntreeModeA(&data, &meteo);
        } else {
            // Fallback Mode B si la météo échoue
            preparerEntreeModeB(&data);
        }
    } else {
        Serial.println("📡 Pas de WiFi → Mode B (offline)");
        // Utiliser model_b.h et scaler_b.h
        // Seulement 9 features capteurs
        preparerEntreeModeB(&data);
    }

    int decision = lancerInference();
    controlerVanne(decision);
    esp_deep_sleep_start();
}
```

---

## 7. Calibration du capteur d'humidité du sol

Avant la première utilisation, tu dois calibrer ton capteur :

```cpp
void calibrerCapteurSol() {
    Serial.println("\n=== CALIBRATION CAPTEUR SOL ===");
    Serial.println("1. Laisse le capteur À L'AIR LIBRE");
    Serial.println("   (lecture dans 5 secondes...)");
    delay(5000);

    int air_value = 0;
    for (int i = 0; i < 10; i++) {
        air_value += analogRead(PIN_SOIL_MOISTURE);
        delay(100);
    }
    air_value /= 10;
    Serial.print("   Valeur air (sec): "); Serial.println(air_value);

    Serial.println("2. Plonge le capteur DANS UN VERRE D'EAU");
    Serial.println("   (ne pas immerger l'électronique!)");
    Serial.println("   (lecture dans 5 secondes...)");
    delay(5000);

    int water_value = 0;
    for (int i = 0; i < 10; i++) {
        water_value += analogRead(PIN_SOIL_MOISTURE);
        delay(100);
    }
    water_value /= 10;
    Serial.print("   Valeur eau (humide): "); Serial.println(water_value);

    // Mettre à jour les constantes
    Serial.println("\n🔧 Mets à jour ton code :");
    Serial.print("   #define SOIL_DRY_AIR    "); Serial.println(air_value);
    Serial.print("   #define SOIL_WET_WATER  "); Serial.println(water_value);
}
```

---

## 8. Tests et vérification

### Test 1 : Vérifier que TFLite s'initialise

```
Étape : Téléverser le programme et ouvrir le moniteur série
Résultat attendu :

=== SYSTÈME D'ARROSAGE INTELLIGENT ===
Démarrage...
✅ BME280 OK
✅ TFLite initialisé !
  Entrée:  9 features, type: int8
  Sortie:  3 classes, type: int8
  Taille modèle: 3560 bytes
✅ Système prêt !
```

**Si TFLite échoue :**
- Vérifie que `model_b.h` est bien dans le même dossier que le .ino
- Vérifie les dépendances dans `platformio.ini`
- Essaie d'augmenter `kTensorArenaSize` à 8192 (8 KB)

### Test 2 : Vérifier les lectures capteurs

```
Résultat attendu :

Données capteurs :
  Température:       29.5 °C
  Humidité air:      65.2 %
  Pression:          1012.3 hPa
  Humidité sol:      45.7 %
  Tendance sol:      -0.83 %/h
```

**Si BME280 échoue :**
- Vérifie le cablage I2C (SCL→D22, SDA→D21)
- Vérifie l'adresse I2C (`0x76` ou `0x77`)
- Essaie `bme.begin(0x77, &Wire)`

### Test 3 : Vérifier l'inférence

```
Résultat attendu :

Exécution du modèle...
  Pas d'arrosage:  0.8234
  Arrosage court:  0.1521
  Arrosage long:   0.0245
✅ Décision finale: Pas d'arrosage
🔴 DÉCISION: Pas d'arrosage
```

### Test 4 : Forcer chaque classe

Pour tester que la vanne s'ouvre correctement :

```cpp
void testVanne() {
    // Forcer arrosage court (15 min)
    Serial.println("TEST: Arrosage court (15 min)");
    digitalWrite(PIN_RELAY_VALVE, LOW);  // LOW = ouvert
    delay(1000);
    digitalWrite(PIN_RELAY_VALVE, HIGH); // HIGH = fermé
    Serial.println("  ✅ OK");

    // Forcer arrosage long (30 min simulé)
    Serial.println("TEST: Arrosage long (1 sec simulé)");
    digitalWrite(PIN_RELAY_VALVE, LOW);
    delay(1000);
    digitalWrite(PIN_RELAY_VALVE, HIGH);
    Serial.println("  ✅ OK");
}
```

---

## 9. Déploiement terrain

### Installation physique

```
1. Choisir un emplacement à l'abri de la pluie directe
   (boîtier IP65 recommandé)

2. Installer le capteur d'humidité du sol :
   - Enterrer à 10-15 cm de profondeur
   - À 20-30 cm du pied de la plante
   - Laisser dépasser la partie électronique

3. Installer le BME280 :
   - À l'ombre (pas de soleil direct)
   - À ~50 cm du sol (hauteur de la culture)
   - Ventilé (ne pas enfermer dans un boîtier étanche)

4. Installer l'électrovanne :
   - Sur la conduite d'irrigation principale
   - Avec un filtre en amont (important !)
   - Connectée au relais via une alim 12V/24V

5. Alimentation secteur :
   - L'ESP32 est "mains powered" dans le PRD
   - Bloc 5V/2A avec câble micro-USB
```

### Mise en route

```
1. Vérifier le cablage
2. Brancher l'alimentation
3. Ouvrir le robinet d'eau principal
4. Vérifier le moniteur série (115200 bauds)
5. Attendre la première décision (6h max)
```

---

## 10. Dépannage

### Problème : L'ESP32 ne démarre pas
```
Causes possibles :
  □ Alimentation insuffisante (besoin de 5V/1A minimum)
  □ Boucle infinie dans setup() (vérifier Serial)
  □ Conflit de broches (I2C vs ADC)
```

### Problème : La vanne ne s'ouvre pas
```
Causes possibles :
  □ Relais HS (tester avec testVanne())
  □ Alimentation électrovanne absente
  □ Filtre obstrué (nettoyer le filtre)
  □ Pression d'eau insuffisante
```

### Problème : L'humidité du sol est toujours à 0%
```
Causes possibles :
  □ Capteur non branché (vérifier SIG→D5)
  □ Calibration incorrecte (relancer calibrerCapteurSol())
  □ Capteur enfoncé trop profondément
```

### Problème : TFLite retourne toujours la même classe
```
Causes possibles :
  □ Normalisation incorrecte (vérifier min/max dans scaler_b.h)
  □ Quantification mal faite (vérifier input_scale)
  □ Features temporelles non fournies (hour_sin, etc.)
```

### Problème : L'ESP32 redémarre en boucle
```
Causes possibles :
  □ Deep sleep sans réveil programmé
  □ Tension alimentation instable
  □ Conflit WiFi (si Mode A)
  
Solution : Ajouter un délai avant le deep sleep
  delay(100);
  esp_sleep_enable_timer_wakeup(6 * 3600 * 1000000ULL);
  esp_deep_sleep_start();
```

---

## Annexe : Arbre de décision rapide

```
L'ESP32 démarre
│
├─ WiFi disponible ?
│   ├─ OUI → Mode A (13 features)
│   │   ├─ Lire BME280
│   │   ├─ Lire capteur sol
│   │   ├─ Appeler Open-Meteo API
│   │   └─ model_a_int8.tflite
│   │
│   └─ NON → Mode B (9 features)
│       ├─ Lire BME280
│       ├─ Lire capteur sol
│       └─ model_b_int8.tflite
│
├─ Normaliser les entrées (MinMax)
├─ Quantifier en int8
├─ Lancer l'inférence TFLite
│
├─ Classe 0 ? → Ne PAS arroser
├─ Classe 1 ? → Arroser 15 min
└─ Classe 2 ? → Arroser 30 min
    │
    └─ Deep sleep 6h
```

---

*Document généré le 6 juin 2026 — Projet Arrosage de Précision ESP32*
