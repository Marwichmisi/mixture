# Plan ML détaillé — Arrosage de précision ESP32

> 📖 **Navigation :** [`README.md`](README.md) ← [`PRD`](PRD_Arrosage_Precision_ESP32.md) ← **Plan ML** → [`Rapport final`](RAPPORT_FINAL.md) → [`Guide implémentation`](GUIDE_IMPLEMENTATION.md)

## Dataset choisi : Stuard (IoT Tomato Cultivation)

**Source :** Mendeley DOI 10.17632/35wh56287y/2
**Localisation :** Azienda Sperimentale Stuard, Parma, Italie — été 2023
**Culture :** Tomates (Solanum lycopersicum L. cv. HEINZ 1301)
**Irrigation :** Goutte-à-goutte enterré

---

## 1. Matching capteurs disponibles vs dataset

### Capteurs disponibles
| Capteur | Mesures | Coût |
|---------|---------|------|
| **DHT22** | Température air (-40~80°C), Humidité air (0-100%) | ~5€ |
| **BME280** | Température air (-40~85°C), Humidité air (0-100%), **Pression** (300-1100 hPa) | ~8€ |
| **Capteur humidité sol** | Humidité du sol (valeur analogique) | ~3€ |
| **ESP32** | Horloge RTC → heure, jour | inclus |

### Match avec les colonnes Stuard
| Colonne Stuard | Capteur dispo ? | Utilisé ? |
|---|---|---|
| `air_temp` | DHT22 ✅ ou BME280 ✅ | **OUI** |
| `humidity` | DHT22 ✅ ou BME280 ✅ | **OUI** |
| `pressure` | BME280 ✅ | **OUI** |
| `soil_moisture` | Capteur sol ✅ | **OUI** |
| `soil_temp` | ❌ Aucun | **NON** (pas de capteur) |
| `ec` (conductivité) | ❌ Aucun | **NON** (pas de capteur) |
| `co2` | ❌ Aucun | **NON** (pas de capteur) |

### Donc, on garde uniquement :
```
air_temp, humidity, pressure, soil_moisture → 4 features terrain
```
+ features temporelles calculées par l'ESP32 :
```
hour_sin, hour_cos, weekday_sin, weekday_cos → 4 features temps
```
= **8 features d'entrée pour le modèle**

### Capteurs recommandés pour améliorer le modèle (+8€)
| Capteur | Prix | Utilité |
|---------|------|---------|
| **DS18B20** (température sol) | ~3€ | Remplace `soil_temp` perdu → donne l'inertie thermique du sol |
| **Capteur de pluie résistif** | ~5€ | Détection pluie binaire → évite d'arroser sous la pluie |

---

## 2. Structure des données

### Fichiers disponibles
| Fichier | Contenu | Lignes | Fréquence |
|---------|---------|--------|-----------|
| `stuard_environmental_data.csv` | Température, humidité, pression, CO₂ | 10 964 | ~10 min |
| `stuard_soil_data.csv` | Humidité sol, température sol, conductivité | 32 668 | ~10 min |
| `stuard_water_meter_data.csv` | Volume d'eau cumulé | 32 649 | ~10 min |

### Période
29 juin 2023 → 13 septembre 2023 (~77 jours)

### Les 3 lignes d'irrigation (notre TARGET)
| Ligne | % recommandation | Intensité | Classe PRD |
|-------|------------------|-----------|------------|
| 1 | 100% | Forte | **2** (arrosage long) |
| 2 | 60% | Modérée | **1** (arrosage court) |
| 3 | 30% | Faible | **0** (pas d'arrosage) |

---

## 3. Stratégie de fusion

### Étape 1 : Nettoyage individuel
- Supprimer les lignes où `device_identifier` = en-tête (doublon dans le CSV)
- Supprimer les NULLs
- Convertir `ts_generation` (epoch ms → datetime)

### Étape 2 : Merge soil + water_meter
- Jointure sur `ts_generation` par fenêtre glissante (±5 min)
- Chaque capteur sol est lié à une ligne (1/2/3) et un compteur d'eau
- Résultat : soil_moisture, current_volume, line

### Étape 3 : Merge avec environnement
- L'environnement est commun aux 3 lignes (1 station météo)
- Pour chaque timestamp sol, prendre la mesure env la plus proche (±5 min)
- Résultat : air_temp, humidity, pressure + features sol + line

### Schéma final après fusion
```
ts                 | datetime
air_temp           | float  (✅ DHT22/BME280)
air_humidity       | float  (✅ DHT22/BME280)
pressure           | float  (✅ BME280)
soil_moisture      | float  (✅ capteur sol)
current_volume     | float  (compteur d'eau — pas de capteur, sert à analyser)
line               | int    (TARGET: 1, 2, ou 3)
```

### Colonnes Stuard qu'on IGNORE volontairement
| Colonne | Raison |
|---------|--------|
| `co2` | Pas de capteur CO₂ dans notre setup |
| `soil_temp` | Pas de capteur (DS18B20 optionnel à 3€) |
| `ec` | Pas de capteur conductivité |

---

## 4. Gestion des données météo (pluie, vent, ET0)

### Constat
Le dataset Stuard n'a PAS de données météo externes.
Nos capteurs non plus (pas d'anémomètre, ni pluviomètre).

### Solution : API Open-Meteo (gratuite, sans clé)
On récupère les données historiques pour Parma, Italie (44.8°N, 10.3°E)
pour juin-sept 2023 :

```python
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 44.8, "longitude": 10.3,
    "start_date": "2023-06-29", "end_date": "2023-09-13",
    "hourly": [
        "precipitation",
        "wind_speed_10m",
        "et0_fao_evapotranspiration",
        "shortwave_radiation"
    ]
}
```

Les données Open-Meteo seront mergées par heure avec notre dataset.

### Deux modes de fonctionnement

#### Mode A — Avec WiFi (recommandé)
- L'ESP32 appelle Open-Meteo (ou une autre API gratuite)
- Features complètes : capteurs terrain + pluie + vent + ET0
- Modèle plus précis

#### Mode B — Sans WiFi (dégradé)
- Uniquement les 4 capteurs terrain
- Features : air_temp, humidity, pressure, soil_moisture + hour/cos + weekday/cos
- Modèle moins précis mais fonctionne offline

---

## 5. Feature engineering (après fusion)

### Variables dérivées par fenêtre de 6h

#### Features terrain (4) — capteurs disponibles
| Feature | Source capteur | Transformations |
|---------|---------------|-----------------|
| `soil_moisture` | Capteur humidité sol | brute + trend 6h + moyenne 6h |
| `air_temp` | DHT22 ou BME280 | brute + moyenne 6h |
| `humidity` | DHT22 ou BME280 | brute + moyenne 6h |
| `pressure` | BME280 | brute |

#### Features météo (si API disponible → Mode A)
| Feature | Source | Transformation |
|---------|--------|----------------|
| `rain_6h` | Open-Meteo | cumul sur 6h |
| `rain_24h` | Open-Meteo | cumul sur 24h |
| `wind_speed` | Open-Meteo | moyenne 6h |
| `et0` | Open-Meteo | cumul sur 6h |
| `solar_radiation` | Open-Meteo | moyenne 6h |

#### Features temporelles (calculées par l'ESP32)
| Feature | Calcul | Utilité |
|---------|--------|---------|
| `hour_sin` | sin(2π × h/24) | Cycle jour/nuit |
| `hour_cos` | cos(2π × h/24) | Cycle jour/nuit |
| `weekday_sin` | sin(2π × d/7) | Cycle semaine |
| `weekday_cos` | cos(2π × d/7) | Cycle semaine |

#### Target
- `line` (1, 2, 3) → mappée en (0, 1, 2)

### Résumé des architectures

| Mode | Features terrain | Features temps | Features météo | Total |
|------|-----------------|---------------|----------------|-------|
| **B** (offline) | 4 | 4 | 0 | **8 entrées** |
| **A** (en ligne) | 4 | 4 | 5 | **13 entrées** |

---

## 6. Pipeline de preprocessing

### Nettoyage
1. Supprimer outliers via IQR (3× écart interquartile)
2. Supprimer les lignes avec valeurs manquantes
3. Garder uniquement les colonnes qui matchent nos capteurs

### Agrégation 6h (alignée sur 00:00, 06:00, 12:00, 18:00)
- Fenêtres glissantes de 6h
- Moyenne des features numériques
- Pente (trend) pour soil_moisture (indique si le sol sèche ou s'humidifie)
- Cumul pour rain_6h

### Normalisation MinMax
```python
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# Paramètres sauvegardés → firmware
```

### Split temporel (pas aléatoire !)
```
Train : 01/07 → 15/08  (60%)
Val   : 16/08 → 31/08  (20%)
Test  : 01/09 → 13/09  (20%)
```

---

## 7. Modèle MLP (identique PRD)

### Architecture
```
Input (8 features)
  ↓
Dense(16, ReLU)
  ↓
Dense(8, ReLU)
  ↓
Dense(3, Softmax)
```

### Entraînement
- Loss : SparseCategoricalCrossentropy
- Optimizer : Adam (lr=0.001)
- Early stopping (patience=20)
- Class weights si déséquilibré

### Quantification int8 pour ESP32
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
```

**Taille estimée :** ~2-4 KB (largement dans les 4MB de l'ESP32)

---

## 8. Évaluation

### Métriques
- Accuracy, Precision, Recall, F1
- Matrice de confusion
- Stabilité des décisions (pas de fluctuation 0→2→0)

### Objectifs
- Accuracy > 80%
- Recall classe 2 (arrosage long) > 85%

---

## 9. Export pour ESP32

### Livrables
| Fichier | Usage |
|---------|-------|
| `model_int8.tflite` | Inférence ESP32 |
| `model.h` | Tableau C inclusion firmware |
| `scaler_params.json` | Min/max normalisation |
| `config.h` | Mapping features, temps cycle |

---

## 10. Plan d'exécution (scripts)

```
ml/
├── data/
│   ├── raw/
│   │   ├── stuard_environmental_data.csv
│   │   ├── stuard_soil_data.csv
│   │   ├── stuard_water_meter_data.csv
│   │   └── weather_parma_2023.csv   (Open-Meteo)
│   ├── merged_dataset.csv
│   └── aggregated_6h.csv
├── training/
│   ├── 01_merge_datasets.py        Fusion + nettoyage
│   ├── 02_fetch_weather.py         Télécharge Open-Meteo
│   ├── 03_feature_engineering.py   Features 6h + normalisation
│   ├── 04_train_mlp.py             Entraînement MLP
│   ├── 05_evaluate.py              Métriques + matrice confusion
│   └── 06_quantize_export.py       Quantification int8 + export
├── models/
│   ├── mlp_model.keras
│   ├── model_int8.tflite
│   └── scaler_params.json
└── firmware/
    ├── model.h                     Modèle en tableau C
    ├── scaler_params.h             Normalisation en C
    └── inference_example.cpp       Exemple inférence
```

---

## 11. Tableau récap' capteurs

| Capteur | Mesure | Prix | Inclus dans MVP ? | Priorité |
|---------|--------|------|-------------------|----------|
| DHT22 | Température air, Humidité air | ~5€ | ✅ Oui | Haute |
| BME280 | Température air, Humidité air, **Pression** | ~8€ | ✅ Oui (remplace DHT22) | Haute |
| Capteur sol capacitif | Humidité sol | ~3€ | ✅ Oui | Haute |
| **DS18B20** | **Température sol** | **~3€** | ❌ Optionnel | **Moyenne** |
| **Capteur pluie** | **Pluie oui/non** | **~5€** | ❌ Optionnel | **Basse** |

**Coût total MVP :** ~16€ (ESP32 + DHT22 + BME280 + capteur sol)
**Avec options :** ~24€ (+ DS18B20 + pluie)
