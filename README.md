# 🌱 Arrosage de Précision ESP32

**Système d'irrigation intelligent** basé sur un MLP (réseau de neurones) quantifié, tournant sur ESP32, pour des décisions d'arrosage toutes les 6h.

**Target :** Tomates / courgettes, goutte-à-goutte, open field, climat tropical (Bénin)

---

## 📚 Documentation

### Pour comprendre le projet
| Document | Description |
|----------|-------------|
| [`PRD_Arrosage_Precision_ESP32.md`](PRD_Arrosage_Precision_ESP32.md) | Cahier des charges complet |
| [`PLAN_ML_DETAIL.md`](PLAN_ML_DETAIL.md) | Plan détaillé de la pipeline ML |
| [`RAPPORT_FINAL.md`](RAPPORT_FINAL.md) | Résultats d'entraînement (91.8% accuracy) |

### Pour implémenter (techniciens / makers)
| Document | Description |
|----------|-------------|
| [`GUIDE_IMPLEMENTATION.md`](GUIDE_IMPLEMENTATION.md) | ⬅️ Guide pas-à-pas pour déployer sur ESP32 |

---

## 🎯 Résumé rapide

| Métrique | Mode B (offline) | Mode A (WiFi) |
|----------|:----------------:|:--------------:|
| **Accuracy** | **89.7%** | **91.8%** |
| Features | 9 capteurs | 13 capteurs + météo |
| Taille TFLite | **3.5 KB** | **3.6 KB** |
| RAM ESP32 | 4 KB tensor arena | 4 KB tensor arena |
| WiFi requis | ❌ Non | ✅ Oui |

### Classes de décision
- **Classe 0** — Pas d'arrosage
- **Classe 1** — Arrosage court (15 min)
- **Classe 2** — Arrosage long (30 min)

---

## 🗂️ Structure du projet

```
mixture/
│
├── PRD, plans, rapports, guide    ← Documentation (tu es ici)
│
└── ml/
    ├── data/                      ← Données (CSV fusionnés)
    ├── training/                  ← Scripts Python d'entraînement
    │   └── experiments/           ← 6 approches alternatives testées
    ├── models/                    ← Modèles .keras, .tflite, scalers
    └── firmware/                  ← ⬅️ Fichiers pour ESP32
        ├── config.h               ← Configuration
        ├── model_b.h / model_a.h  ← Modèles en tableau C
        ├── scaler_b.h / scaler_a.h ← Normalisation
        └── inference_example.cpp  ← Exemple complet
```

---

## 🚀 Pour commencer

**Si tu es technicien** → lis le [`GUIDE_IMPLEMENTATION.md`](GUIDE_IMPLEMENTATION.md)

**Si tu veux comprendre le ML** → lis le [`RAPPORT_FINAL.md`](RAPPORT_FINAL.md)

**Si tu veux tout savoir** → commence par le [`PRD_Arrosage_Precision_ESP32.md`](PRD_Arrosage_Precision_ESP32.md)

---

## 📊 Pipeline ML (schéma)

```
┌─────────┐   ┌──────────┐   ┌─────────┐   ┌────────┐   ┌─────────┐
│ Fichiers │→ │ Fusion   │→ │ Features│→ │ MLP    │→ │ TFLite  │
│ CSV      │  │ +/-5min  │  │ 6h +    │  │ 16→8→3 │  │ int8    │
│ Stuard   │  │ 31k→918  │  │ temps   │  │ ~1k    │  │ 3.5 KB  │
└─────────┘   └──────────┘  │cyclique │  │ params  │  │ (C array)│
                            └─────────┘  └────────┘   └─────────┘
```

---

## 📝 Licence

Projet open-source — MIT
