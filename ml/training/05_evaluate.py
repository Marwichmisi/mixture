import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')
REPORT = os.path.join(os.path.dirname(__file__), '..', 'models', 'evaluation_report.md')

CLASS_LABELS = ['No watering', 'Short watering', 'Long watering']

def evaluate_model(model, X_test, y_test, scaler, features, mode_name):
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_LABELS, zero_division=0, output_dict=True)

    per_class = {}
    for i, label in enumerate(CLASS_LABELS):
        key = label
        per_class[key] = {
            'precision': report[key]['precision'] if key in report else 0.0,
            'recall': report[key]['recall'] if key in report else 0.0,
            'f1': report[key]['f1-score'] if key in report else 0.0,
            'support': int(report[key]['support']) if key in report else 0,
        }

    return {
        'accuracy': float(acc),
        'precision_weighted': float(prec),
        'recall_weighted': float(rec),
        'f1_weighted': float(f1),
        'confusion_matrix': cm.tolist(),
        'per_class': per_class,
        'pred_dist': np.bincount(y_pred, minlength=3).tolist(),
        'true_dist': np.bincount(y_test, minlength=3).tolist(),
    }

def plot_confusion_matrix(cm, labels, title, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def generate_report(agg_df, results_b, results_a, cm_b, cm_a):
    lines = []
    lines.append("# Rapport d'évaluation — Modèle d'arrosage ESP32\n")
    lines.append(f"Généré le : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")

    lines.append("## Résumé du dataset\n")
    lines.append(f"- **Total fenêtres 6h :** {len(agg_df)}")
    lines.append(f"- **Période :** {agg_df['ts'].min()} → {agg_df['ts'].max()}")
    lines.append(f"- **Fréquence :** 4 décisions/jour (6h)")
    lines.append(f"- **Source :** Dataset Stuard (Parma, Italie)")
    lines.append(f"- **3 lignes d'irrigation fusionnées** → 918 échantillons\n")

    lines.append("### Target")
    lines.append("La cible est créée à partir du **volume d'eau réellement mesuré**")
    lines.append("par fenêtre de 6h :")
    lines.append("- 0 : augmentation nulle ou négligeable (pas d'irrigation)")
    lines.append("- 1 : augmentation modérée (irrigation courte)")
    lines.append("- 2 : augmentation forte (irrigation longue)")
    class_counts = agg_df['class'].value_counts().sort_index()
    total = len(agg_df)
    for cls in [0, 1, 2]:
        count = class_counts.get(cls, 0)
        pct = count / total * 100
        lines.append(f"- **Classe {cls}** ({CLASS_LABELS[cls]}) : {count} ({pct:.1f}%)")

    pct0 = class_counts.get(0, 0) / total * 100
    lines.append(f"\n⚠️ **Déséquilibre :** {pct0:.0f}% des fenêtres sont classe 0 (pas d'irrigation).")
    lines.append("Les événements d'irrigation sont rares mais concentrés.")

    lines.append("\n### Split")
    lines.append("Split stratifié aléatoire (60% train, 20% val, 20% test) ")
    lines.append("pour préserver la distribution des classes rares.\n")

    rb = results_b
    ra = results_a
    lines.append("### Features\n")
    lines.append("**Mode B (offline)** — 9 features :")
    lines.append("`air_temp`, `humidity`, `pressure`, `soil_moisture`, `soil_moisture_trend`,")
    lines.append("`hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos`\n")
    if results_a:
        lines.append("**Mode A (WiFi)** — 13 features :")
        lines.append("9 features Mode B + `rain_6h`, `wind_speed`, `et0`, `solar_radiation`\n")

    # Mode B
    lines.append("---\n## Mode B — Capteurs uniquement (offline)\n")
    lines.append(f"| Métrique | Valeur |")
    lines.append(f"|----------|--------|")
    lines.append(f"| Accuracy | {rb['accuracy']:.4f} ({rb['accuracy']*100:.1f}%) |")
    lines.append(f"| Precision (weighted) | {rb['precision_weighted']:.4f} |")
    lines.append(f"| Recall (weighted) | {rb['recall_weighted']:.4f} |")
    lines.append(f"| F1-score (weighted) | {rb['f1_weighted']:.4f} |")

    lines.append("\n**Distribution prédite vs réelle :**")
    lines.append(f"- Prédit : {rb['pred_dist']}")
    lines.append(f"- Réel :   {rb['true_dist']}")

    lines.append("\n### Performances par classe")
    lines.append("\n| Classe | Precision | Recall | F1 | Support |")
    lines.append("|--------|-----------|--------|----|---------|")
    for label in CLASS_LABELS:
        p = rb['per_class'][label]
        lines.append(f"| {label} | {p['precision']:.3f} | {p['recall']:.3f} | {p['f1']:.3f} | {p['support']} |")

    if cm_b:
        lines.append(f"\n![ConfMat B]({os.path.relpath(cm_b, MODELS)})")

    # Mode A
    if results_a:
        lines.append("\n---\n## Mode A — Capteurs + Météo (WiFi)\n")
        lines.append(f"| Métrique | Valeur |")
        lines.append(f"|----------|--------|")
        lines.append(f"| Accuracy | {ra['accuracy']:.4f} ({ra['accuracy']*100:.1f}%) |")
        lines.append(f"| Precision (weighted) | {ra['precision_weighted']:.4f} |")
        lines.append(f"| Recall (weighted) | {ra['recall_weighted']:.4f} |")
        lines.append(f"| F1-score (weighted) | {ra['f1_weighted']:.4f} |")

        lines.append("\n**Distribution prédite vs réelle :**")
        lines.append(f"- Prédit : {ra['pred_dist']}")
        lines.append(f"- Réel :   {ra['true_dist']}")

        lines.append("\n### Performances par classe")
        lines.append("\n| Classe | Precision | Recall | F1 | Support |")
        lines.append("|--------|-----------|--------|----|---------|")
        for label in CLASS_LABELS:
            p = ra['per_class'][label]
            lines.append(f"| {label} | {p['precision']:.3f} | {p['recall']:.3f} | {p['f1']:.3f} | {p['support']} |")

        if cm_a:
            lines.append(f"\n![ConfMat A]({os.path.relpath(cm_a, MODELS)})")

    # Interpretation
    lines.append("\n---\n## Analyse\n")

    for m_name, res in [('Mode B', results_b), ('Mode A', results_a)]:
        if res is None:
            continue
        acc = res['accuracy']
        pred_set = sum(1 for c in res['pred_dist'] if c > 0)
        true_set = sum(1 for c in res['true_dist'] if c > 0)

        if pred_set < 2:
            lines.append(f"⚠️ **{m_name} :** Le modèle prédit toujours la même classe. ")
            lines.append(f"Le déséquilibre des classes empêche la discrimination.\n")
        elif acc >= 0.65:
            lines.append(f"✅ **{m_name} :** Accuracy {acc*100:.1f}% — Le modèle discrimine entre les classes.\n")
        elif acc >= 0.40:
            lines.append(f"⚠️ **{m_name} :** Accuracy {acc*100:.1f}% — Partiellement au-dessus du hasard.\n")
        else:
            lines.append(f"❌ **{m_name} :** Accuracy {acc*100:.1f}% — Proche du hasard.\n")

    lines.append("""### Limites du POC
1. **Target basé sur le comportement** : on apprend *ce qui a été fait*, pas *ce qu'il aurait fallu faire*.
   Les 3 lignes reçoivent toutes de l'eau en même temps, donc la distinction entre classes 1 et 2 est floue.
2. **Déséquilibre des classes** : 88% des fenêtres n'ont pas d'irrigation. Le modèle est pénalisé s'il prédit autre chose que la classe 0.
3. **Split aléatoire** (pas temporel) : nécessaire pour garder des classes rares dans le test set. En production, le modèle verra des séquences temporelles.
4. **Transposabilité Bénin** : climat, sol et saisonnalité différents → un ré-entraînement local sera nécessaire.

### Recommandations
| Action | Coût | Impact |
|--------|------|--------|
| **DS18B20** (température sol) | ~3€ | Récupère une feature discriminante manquante |
| **Capteur pluie** (binaire) | ~5€ | Détection pluie temps réel |
| **Ré-entraînement terrain** | 0€ | Collecter 4-8 semaines → modèle adapté au Bénin |
""")

    lines.append(f"\n---\n*Rapport généré automatiquement — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n")
    return '\n'.join(lines)

def main():
    agg = pd.read_csv(os.path.join(DATA, 'aggregated_6h.csv'), parse_dates=['ts'])
    y = agg['class'].values

    # --- Mode B ---
    print("Loading Mode B model...")
    model_b = keras.models.load_model(os.path.join(MODELS, 'mlp_model_b.keras'))
    with open(os.path.join(MODELS, 'scaler_b.pkl'), 'rb') as f:
        data_b = pickle.load(f)
    scaler_b, features_b = data_b['scaler'], data_b['features']

    X = agg[features_b].values.astype(np.float32)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_test_s = scaler_b.transform(X_test)

    results_b = evaluate_model(model_b, X_test_s, y_test, scaler_b, features_b, 'B')
    print(f"Mode B - Acc: {results_b['accuracy']:.4f}, Pred: {results_b['pred_dist']}, True: {results_b['true_dist']}")

    cm_b_path = os.path.join(MODELS, 'confusion_matrix_b.png')
    plot_confusion_matrix(np.array(results_b['confusion_matrix']), CLASS_LABELS,
                          'Mode B - Matrice de confusion', cm_b_path)

    # --- Mode A ---
    results_a = None
    cm_a_path = None
    model_a_path = os.path.join(MODELS, 'mlp_model_a.keras')

    if os.path.exists(model_a_path):
        print("\nLoading Mode A model...")
        model_a = keras.models.load_model(model_a_path)
        with open(os.path.join(MODELS, 'scaler_a.pkl'), 'rb') as f:
            data_a = pickle.load(f)
        scaler_a, features_a = data_a['scaler'], data_a['features']

        X_a = agg[features_a].values.astype(np.float32)
        _, X_test_a, _, y_test_a = train_test_split(X_a, y, test_size=0.2, stratify=y, random_state=42)
        X_test_a_s = scaler_a.transform(X_test_a)

        results_a = evaluate_model(model_a, X_test_a_s, y_test_a, scaler_a, features_a, 'A')
        print(f"Mode A - Acc: {results_a['accuracy']:.4f}, Pred: {results_a['pred_dist']}, True: {results_a['true_dist']}")

        cm_a_path = os.path.join(MODELS, 'confusion_matrix_a.png')
        plot_confusion_matrix(np.array(results_a['confusion_matrix']), CLASS_LABELS,
                              'Mode A - Matrice de confusion', cm_a_path)

    # Generate report
    report_md = generate_report(agg, results_b, results_a, cm_b_path, cm_a_path)
    with open(REPORT, 'w') as f:
        f.write(report_md)
    print(f"\n✅ Report → {REPORT}")

    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    print(f"Mode B : Acc={results_b['accuracy']:.2%}")
    for label in CLASS_LABELS:
        p = results_b['per_class'][label]
        print(f"  {label:20s}  P={p['precision']:.2f}  R={p['recall']:.2f}  F1={p['f1']:.2f}  n={p['support']}")
    if results_a:
        print(f"Mode A : Acc={results_a['accuracy']:.2%}")
        for label in CLASS_LABELS:
            p = results_a['per_class'][label]
            print(f"  {label:20s}  P={p['precision']:.2f}  R={p['recall']:.2f}  F1={p['f1']:.2f}  n={p['support']}")

if __name__ == '__main__':
    main()
