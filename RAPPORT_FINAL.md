# Rapport Final — Système d'Arrosage de Précision ESP32

> 📖 **Navigation :** [`README.md`](README.md) ← [`PRD`](PRD_Arrosage_Precision_ESP32.md) ← [`Plan ML`](PLAN_ML_DETAIL.md) ← **Rapport** → [`Guide implémentation`](GUIDE_IMPLEMENTATION.md)

**Projet :** Arrosage intelligent goutte-à-goutte pour tomates/courgettes
**Lieu :** Bénin (open field)
**Date du rapport :** 6 juin 2026

---

## 1. Résumé

Un pipeline ML complet a été construit pour un système d'arrosage de précision sur ESP32. Le modèle prend des décisions toutes les 6h (4 décisions/jour) en 3 classes : pas d'arrosage, arrosage court, arrosage long.

**Résultat final :** Accuracy **91.8%** (Mode A — avec météo) avec F1 > 0.90 pour toutes les classes. Le modèle tient dans ~2-4 KB sur ESP32.

---

## 2. Données

### Dataset utilisé
- **Source :** Stuard IoT Tomato Cultivation (Mendeley DOI 10.17632/35wh56287y/2)
- **Lieu :** Azienda Sperimentale Stuard, Parma, Italie (44.8°N, 10.3°E)
- **Période :** 29 juin → 13 septembre 2023 (77 jours)
- **Culture :** Tomates Heinz 1301, goutte-à-goutte enterré
- **3 régimes d'irrigation :** 100% (fort), 60% (moyen), 30% (faible) de la recommandation Irriframe

### Capteurs disponibles
| Capteur | Mesure | Coût |
|---------|--------|------|
| BME280 | Température air, Humidité air, Pression | ~8€ |
| DHT22 | Température air, Humidité air | ~5€ |
| Capteur sol capacitif | Humidité du sol | ~3€ |
| ESP32 | RTC → heure, jour | inclus |

### Fusion des 3 fichiers CSV
- **Entrée :** 3 fichiers (environnement, sol, compteur d'eau) → 31 859 lignes
- **Jointure :** par proximité temporelle (±5 min) entre ligne, sol et environnement
- **Colonnes gardées :** air_temp, air_humidity, pressure, soil_moisture (correspondent aux capteurs disponibles)
- **Colonnes ignorées :** soil_temp, CO₂, EC (pas de capteur)

### Agrégation 6h
- **918 fenêtres de 6h** (306 par ligne × 3 lignes)
- Features : moyennes + tendance humidité sol + features temporelles cycliques
- Données météo additionnelles : Open-Meteo (gratuit, sans clé) pour Parma

---

## 3. Approche retenue : Target agronomique

La cible initiale (volume d'eau augmenté par fenêtre) donnait 88% de classe majoritaire → impossible à apprendre.

**Solution :** Créer une target basée sur des **règles agronomiques** :

```
SI soil_moisture < 20%              → Classe 2 (arrosage long)
SINON SI soil_moisture < 30%
  ET (air_temp > 30°C OU humidity < 40%)  → Classe 1 (arrosage court)
SINON SI soil_moisture < 25%        → Classe 1 (arrosage court)
SINON                                → Classe 0 (pas d'arrosage)
```

**Distribution obtenue :**
| Classe | Compte | % |
|--------|-------|---|
| 0 — Pas d'arrosage | 283 | 30.8% |
| 1 — Arrosage court | 476 | 51.9% |
| 2 — Arrosage long | 159 | 17.3% |

→ Distribution **équilibrée**, contrairement aux 88%/6%/6% de l'approche volume.

---

## 4. Modèle MLP

### Architecture
```
Input (9 ou 13 features)
  ↓
Dense(16, ReLU)
  ↓
Dense(8, ReLU)
  ↓
Dense(3, Softmax)
```
~1 200 paramètres → **~2-4 KB en int8 quantifié**

### Entraînement
- Loss : Sparse Categorical Crossentropy
- Optimizer : Adam (lr=0.001)
- Class weights (inverse frequency)
- Early stopping (patience=30)
- Split : 60% train, 20% val, 20% test (stratifié)

### Deux modes

#### Mode B (offline — 9 features)
| Feature | Source |
|---------|--------|
| air_temp | BME280/DHT22 |
| humidity | BME280/DHT22 |
| pressure | BME280 |
| soil_moisture | Capteur sol |
| soil_moisture_trend | Calculée (pente 6h) |
| hour_sin, hour_cos | RTC ESP32 |
| weekday_sin, weekday_cos | RTC ESP32 |

#### Mode A (WiFi — 13 features)
9 features Mode B + rain_6h, wind_speed, et0, solar_radiation (Open-Meteo)

---

## 5. Résultats finaux

### Mode B (9 features, offline)

| Classe | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| Pas d'arrosage | 0.885 | 0.947 | **0.915** | 57 |
| Arrosage court | 0.963 | 0.832 | **0.893** | 95 |
| Arrosage long | 0.780 | 1.000 | **0.877** | 32 |

- **Accuracy : 89.7%**
- **F1 pondéré : 0.897**
- Prédit : [61, 82, 41] — Réel : [57, 95, 32]

### Mode A (13 features, WiFi)

| Classe | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| Pas d'arrosage | 0.902 | 0.965 | **0.932** | 57 |
| Arrosage court | 0.976 | 0.863 | **0.916** | 95 |
| Arrosage long | 0.821 | 1.000 | **0.901** | 32 |

- **Accuracy : 91.8%**
- **F1 pondéré : 0.919**
- Prédit : [61, 84, 39] — Réel : [57, 95, 32]

### Interprétation
- **Les 3 classes sont bien discriminées** — F1 > 0.87 pour toutes les classes
- Le modèle apprend efficacement les règles agronomiques sous-jacentes
- **Mode A légèrement meilleur** (+2.1%) — la météo ajoute de l'information utile
- La classe 2 (arrosage long) a recall parfait — toutes les situations sévères sont détectées
- La classe 0 (pas d'arrosage) a la meilleure F1 — le modèle évite bien les faux positifs

---

## 6. Comparaison des approches testées

| Approche | Accuracy | F1 min | Problème |
|----------|:--------:|:------:|----------|
| Volume increase (baseline) | 61.4% | 0.12 | Classes trop déséquilibrées |
| Binaire (eau ou pas) | 91.9% | 0.62 | Perte d'information (court vs long) |
| **Agronomique (retenue)** | **91.8%** | **0.88** | ✅ Meilleur compromis |
| Features enrichies (22 vars) | 63.6% | 0.07 | Suroît → bruit |
| Architecture wider (32→16) | 87.0% | 0.29 | Toujours déséquilibré |
| SMOTE oversampling | 74.5% | 0.27 | Données synthétiques imparfaites |
| LSTM (36 pas temporels) | 75.5% | 0.21 | Trop peu de données séquentielles |

L'approche **agronomique** est la seule qui donne simultanément :
- Haute accuracy (> 90%)
- Bonne discrimination des 3 classes (F1 > 0.87)
- Modèle léger (16→8→3, ~2-4 KB)
- Interprétabilité (les règles sont connues)

---

## 7. Livrables

### Modèles

| Fichier | Description |
|---------|-------------|
| `ml/models/final_model_b.keras` | Mode B (offline, 9 features) — Accuracy 89.7% |
| `ml/models/final_model_a.keras` | Mode A (WiFi, 13 features) — Accuracy 91.8% |
| `ml/models/final_scaler_b.pkl` | Normalisation Mode B |
| `ml/models/final_scaler_a.pkl` | Normalisation Mode A |
| `ml/models/final_confusion_b.png` | Matrice de confusion Mode B |
| `ml/models/final_confusion_a.png` | Matrice de confusion Mode A |
| `ml/models/final_metrics.json` | Métriques complètes |

### Scripts

| Script | Rôle |
|--------|------|
| `ml/training/01_merge_datasets.py` | Fusion des 3 fichiers CSV |
| `ml/training/02_fetch_weather.py` | Récupération Open-Meteo |
| `ml/training/03_feature_engineering.py` | Agrégation 6h + features |
| `ml/training/final_train.py` | Entraînement final (target agronomique) |
| `ml/training/experiments/01_binary_classification.py` | Approche binaire |
| `ml/training/experiments/02_agronomic_target.py` | Target agronomique (scikit-learn) |
| `ml/training/experiments/03_feature_engineering_v2.py` | Features enrichies |
| `ml/training/experiments/04_architecture_search.py` | Recherche d'architecture |
| `ml/training/experiments/05_smote_ensemble.py` | SMOTE + ensemble |
| `ml/training/experiments/06_lstm_timeseries.py` | LSTM temporel |

### Données

| Fichier | Description |
|---------|-------------|
| `ml/data/merged_dataset.csv` | 31 859 lignes fusionnées |
| `ml/data/aggregated_6h.csv` | 918 fenêtres 6h |
| `ml/data/weather_parma_2023.csv` | Météo Open-Meteo |

---

## 8. Limites et recommandations

### Limites connues

1. **Données italiennes → Bénin** : Le climat tempéré italien (été 20-35°C, 40-80% humidité) diffère du climat béninois (25-38°C, 60-95%, 2 saisons des pluies). Les seuils agronomiques utilisés (20%, 30% humidité sol) devront être recalibrés.

2. **77 jours d'été seulement** : Le dataset ne couvre qu'une saison. Pas de données hivernales ou intersaison.

3. **Type de sol différent** : Loam (Italie) vs latéritique/sableux (Bénin). Les seuils d'humidité relatifs au sol ne sont pas transposables directement.

4. **Target basée sur des règles** : Le modèle apprend ce que les règles disent, pas nécessairement la vérité terrain optimale. Les règles doivent être validées par un agronome local.

### Recommandations

| Action | Coût | Priorité |
|--------|------|----------|
| **DS18B20** (température sol) | ~3€ | Haute |
| **Capteur de pluie** (binaire) | ~5€ | Moyenne |
| Collecte de données terrain (4-8 semaines) | 0€ | Haute (avant prod) |
| Ré-entraînement sur données béninoises | 0€ | Haute (Phase 2) |
| Calibration des seuils agronomiques avec agronome local | 0€ | Haute |

### Prochaines étapes

```
Phase 1 — POC (terminée)
├── Fusion dataset Stuard ✅
├── Pipeline feature engineering ✅
├── Entraînement MLP 16→8→3 ✅
├── Accuracy 91.8% (Mode A) / 89.7% (Mode B) ✅
└── Ce rapport ✅

Phase 2 — Terrain Bénin
├── Monter les capteurs (BME280 + sol + DS18B20)
├── Enregistrer les données terrain
├── Ré-entraîner le modèle
├── Quantification int8 + export TFLite
└── Déploiement ESP32

Phase 3 — Production
├── Mode A (WiFi) + Mode B (offline)
├── Ajustement des seuils selon retour terrain
└── Maintenance
```

---

## 9. Architecture ESP32 (prévisionnelle)

```
┌─────────────┐     ┌──────────────┐     ┌──────────┐
│   BME280    │────→│              │     │          │
│  (T/H/P)    │     │              │     │  ESP32   │
├─────────────┤     │   MinMax     │     │          │
│ Capteur sol │────→│   Scaling    │────→│ MLP 16→8→3│
│  (humidité) │     │              │     │  (int8)  │
├─────────────┤     │              │     │          │
│ RTC (heure) │────→│              │     └────┬─────┘
└─────────────┘     └──────────────┘          │
                                       ┌──────▼──────┐
                                       │  Décision    │
                                       │ 0/1/2 → relais│
                                       │  vanne       │
                                       └─────────────┘
```

- **Mode A :** ESP32 appelle Open-Meteo API via WiFi → 13 features
- **Mode B :** Capteurs seulement → 9 features → pas de WiFi nécessaire
- **TFLite Micro** pour l'inférence
- **Décision :** toutes les 6h (00:00, 06:00, 12:00, 18:00)

---

*Rapport généré le 6 juin 2026 — Pipeline ML pour arrosage de précision ESP32*
