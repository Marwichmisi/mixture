
# PRD — Système d’arrosage de précision piloté par ML sur ESP32

> 📖 **Navigation :** [`README.md`](README.md) ← **PRD** → [`Plan ML`](PLAN_ML_DETAIL.md) → [`Rapport final`](RAPPORT_FINAL.md) → [`Guide implémentation`](GUIDE_IMPLEMENTATION.md)

## 1. Résumé exécutif

Ce projet vise à concevoir un système d’arrosage intelligent, exécutable sur ESP32, capable de prendre une décision d’irrigation toutes les 6 heures à partir de capteurs terrain et de données météo.  
Le rôle du modèle ML est de fournir une recommandation exploitable par l’équipe électronique/Arduino pour piloter une pompe ou une électrovanne.

Le système est conçu pour un contexte de **plein air, climat tempéré, sol de type loam sablo-argileux**, avec une alimentation secteur (non batterie). Le type de plante cible est une **culture maraîchère de pleine terre (tomates, courgettes)** avec un système d'irrigation **goutte-à-goutte**. Ces paramètres conditionnent l'ensemble des choix techniques du document.

Objectif fonctionnel :

- décider s’il faut arroser ou non ;
- ou, mieux, décider d’un **niveau d’arrosage** parmi plusieurs classes ;
- exécuter la décision de manière robuste, simple à maintenir et compatible avec les limites mémoire d’un ESP32.

---

## 2. Contexte du projet

L’équipe ML prépare le modèle et la logique de décision.  
L’équipe électronique/Arduino se charge de :

- la lecture des capteurs ;
- l’intégration du modèle dans le firmware ;
- le pilotage des relais / pompe ;
- les tests matériels.

Le système doit être compréhensible par une équipe ayant de bonnes bases en programmation Arduino et électronique, mais peu d’expérience en déploiement ML embarqué.

---

## 3. Périmètre

### Inclus
- lecture périodique des capteurs ;
- préparation des données ;
- entraînement du modèle ;
- export du modèle pour ESP32 ;
- inférence locale sur ESP32 ;
- commande d’un actionneur d’arrosage ;
- journalisation minimale pour vérification.

### Exclu
- application mobile complète ;
- backend cloud complexe ;
- déploiement multi-serres ;
- optimisation avancée type MLOps industriel.

---

## 4. Objectif métier

Le système doit permettre un **arrosage de précision toutes les 6 heures** en fonction :

- de l’état réel du sol ;
- de l’état de l’air ;
- des conditions météo immédiates ou prévues ;
- d’un historique court des mesures.

La décision finale doit éviter :

- l’arrosage inutile ;
- l’arrosage trop fréquent ;
- l’arrosage avant une pluie prévue ;
- le sous-arrosage en période chaude et sèche.

---

## 5. Données d’entrée disponibles

### Capteurs disponibles dans le projet
- DHT22 : température de l’air, humidité relative de l’air
- Capteur d’humidité du sol
- BME280 : température, humidité, pression atmosphérique

### Données demandées mais non mesurées directement par vos capteurs
- pluie récente
- pluie prévue
- vent
- ET0 si disponible
- heure du jour / jour de la semaine
- température du sol

### Décision d’architecture sur les variables
Comme vous n’avez pas de capteur de pluie, de vent, ni de mesure directe ET0, il faut séparer les variables en deux catégories :

1. **Variables locales mesurées en direct**
   - humidité du sol
   - température de l’air
   - humidité de l’air
   - pression atmosphérique

2. **Variables météo externes ou dérivées**
   - pluie récente
   - pluie prévue
   - vent
   - ET0
   - heure du jour
   - jour de la semaine

### Recommandation pratique
Pour le MVP, le modèle doit fonctionner en deux modes :

- **Mode A : avec internet**
  - récupérer pluie prévue, pluie récente, vent, ET0 via API météo ;
  - meilleure précision.

- **Mode B : sans internet**
  - utiliser uniquement les capteurs locaux ;
  - supprimer pluie prévue / vent / ET0 ;
  - modèle dégradé mais toujours exploitable.

---

## 6. Choix du modèle ML recommandé

## Modèle retenu : petit réseau de neurones dense quantifié int8 (TinyML MLP)

### Pourquoi ce choix
C’est le meilleur compromis pour votre cas parce que :

- il est compatible avec ESP32 ;
- il est plus simple à entraîner qu’un modèle séquentiel complexe ;
- il se convertit bien en modèle léger pour microcontrôleur ;
- il gère correctement des données tabulaires ;
- il reste plus flexible qu’une simple règle dure ;
- il est plus facile à maintenir par une équipe Arduino qu’un modèle lourd.

### Pourquoi pas un LSTM / GRU
- trop coûteux pour un petit ESP32 ;
- utile surtout pour des séries temporelles longues ;
- plus difficile à déployer et à déboguer ;
- pas nécessaire pour un cycle de décision toutes les 6 h avec peu de variables.

### Pourquoi pas XGBoost / Random Forest comme modèle final
- très bons sur données tabulaires ;
- mais moins naturels à embarquer proprement dans un firmware ESP32 ;
- plus difficiles à maintenir avec un code Arduino simple ;
- ils peuvent servir de **modèle de référence** côté PC, mais pas comme modèle final embarqué.

### Pourquoi pas une règle seule
- trop rigide ;
- ne s’adapte pas aux variations de climat, saisons et sol ;
- ne permet pas d’apprendre automatiquement à partir des données.

### Format de sortie recommandé
Pour l’arrosage de précision toutes les 6 h, la sortie la plus utile n’est pas seulement “oui/non”, mais une **classe d’irrigation** :

- **Classe 0** : ne pas arroser
- **Classe 1** : arroser faiblement
- **Classe 2** : arroser fortement

Cette sortie est plus précise qu’une simple décision binaire tout en restant simple à exécuter sur ESP32.

### Architecture du modèle recommandée
- Entrée : 8 à 12 variables selon disponibilité réelle
- Couche cachée 1 : 16 neurones, ReLU
- Couche cachée 2 : 8 neurones, ReLU
- Sortie : 3 neurones, softmax
- Quantification : int8
- Format d’export : TensorFlow Lite / LiteRT Micro
- Taille cible du modèle : très compacte, orientée microcontrôleur

---

## 7. Variables d’entrée du modèle

## 7.1 Variables minimales du MVP
- humidité du sol
- température de l’air
- humidité de l’air
- pression atmosphérique
- heure de la journée
- jour de la semaine

## 7.2 Variables enrichies si API météo disponible
- pluie récente (ex. cumul 6 h ou 24 h)
- pluie prévue sur les 6 prochaines heures
- vitesse du vent
- ET0
- tendance de l’humidité du sol
- moyenne mobile des capteurs sur 6 h

## 7.3 Variables dérivées à créer
Ces variables améliorent fortement le modèle :

- `soil_moisture_avg_6h`
- `soil_moisture_trend_6h`
- `air_temp_avg_6h`
- `air_humidity_avg_6h`
- `rain_sum_6h`
- `rain_sum_24h`
- `weather_risk_index`
- `irrigation_cooldown`
- `hour_sin`, `hour_cos`
- `weekday_sin`, `weekday_cos`

Les variables cycliques `sin/cos` sont recommandées pour représenter l’heure et le jour sans rupture artificielle.

---

## 8. Définition de la cible (labels)

## Option recommandée
Créer une cible à 3 classes :

- **0 = Ne pas arroser**
- **1 = Arroser peu**
- **2 = Arroser plus longtemps**

### Traduction côté actionneur
Le firmware convertit la classe en durée de pompe :

- classe 0 → pompe OFF
- classe 1 → pompe ON pendant une durée courte
- classe 2 → pompe ON pendant une durée plus longue

### Exemple de mapping initial
- Classe 0 : 0 seconde
- Classe 1 : 30 à 60 secondes
- Classe 2 : 90 à 180 secondes

Ces durées doivent être ajustées après test terrain.

---

## 9. Stratégie de création des labels

Au départ, vous n’aurez probablement pas assez de vérité terrain.  
Il faut donc construire les labels avec une logique progressive.

### Phase 1 — Labels initiaux par règle experte
Créer une règle de démarrage :

- si l’humidité du sol est basse
- et si la pluie récente est faible
- et si la pluie prévue est faible
- et si la météo indique une forte demande évaporative
- alors label = arroser

### Phase 2 — Validation manuelle
L’équipe peut corriger certains labels après test réel.

### Phase 3 — Réentraînement
Une fois assez de données collectées, le modèle est réentraîné sur des décisions réellement observées.

### Remarque importante
Les coefficients peuvent être utilisés pour :
- normaliser ;
- pondérer ;
- calibrer ;
- créer des indices dérivés.

Ils ne doivent pas remplacer les données réelles.

---

## 10. Sources de données

## 10.1 Données locales
- DHT22
- capteur d’humidité du sol
- BME280

## 10.2 Données météo externes
Si l’ESP32 est connecté au Wi‑Fi, utiliser une API météo pour obtenir :

- pluie récente
- pluie prévue
- vent
- température
- humidité
- pression
- éventuellement ET0

## 10.3 Recommandation d’architecture
Le système doit pouvoir fonctionner même sans API externe.  
Dans ce cas, le modèle utilisé est une version simplifiée entraînée avec les variables locales disponibles.

---

## 11. Prétraitement des données

### Nettoyage
- suppression des valeurs impossibles ;
- suppression des doublons ;
- gestion des valeurs manquantes ;
- lissage des pics anormaux.

### Synchronisation
Toutes les données doivent être alignées sur le même horodatage.

### Échantillonnage
Le système de décision doit fonctionner toutes les 6 heures.  
Le dataset d’entraînement doit donc être agrégé à cette échelle.

### Normalisation
Les variables numériques doivent être normalisées pour l’entraînement :
- min-max normalization
- ou standardisation z-score

La méthode choisie doit être conservée exactement à l’identique dans le firmware.

---

## 12. Pipeline de données recommandé

### Étape 1
Collecter les mesures brutes.

### Étape 2
Construire une table temporelle.

### Étape 3
Créer les variables dérivées.

### Étape 4
Construire les labels.

### Étape 5
Séparer les données selon le temps :
- train
- validation
- test

### Étape 6
Entraîner le modèle.

### Étape 7
Comparer plusieurs modèles.

### Étape 8
Choisir le meilleur modèle embarquable.

### Étape 9
Quantifier et exporter.

### Étape 10
Intégrer dans le firmware ESP32.

---

## 13. Métriques d’évaluation

Les métriques doivent refléter la qualité réelle de l’arrosage, pas seulement l’exactitude globale.

### Métriques obligatoires
- accuracy
- precision
- recall
- F1-score
- matrice de confusion

### Métriques métiers importantes
- nombre d’arrosages inutiles évités
- nombre de sous-arrosages réduits
- stabilité des décisions
- fréquence des basculements entre classes

### Objectif minimal
Le modèle doit surtout bien détecter les situations où il faut arroser, sans déclencher trop de faux positifs.

---

## 14. Contraintes embarquées ESP32

Le modèle final doit :

- tenir dans la mémoire disponible ;
- s’exécuter rapidement ;
- ne pas dépendre d’une machine distante ;
- rester compatible avec le code Arduino/ESP-IDF ;
- être simple à maintenir.

### Recommandation technique
Utiliser une bibliothèque de type **LiteRT / TensorFlow Lite for Microcontrollers**.  
Ce runtime est conçu pour les microcontrôleurs et a déjà été porté sur ESP32.

### Implication pour l’équipe Arduino
Le modèle sera fourni sous forme :
- de fichier quantifié ;
- ou de tableau C ;
- avec une logique d’inférence simple à appeler dans le code.

---

## 15. Architecture fonctionnelle du firmware

## Chaîne de traitement
1. lecture des capteurs
2. récupération des données météo si réseau disponible
3. calcul des variables dérivées
4. normalisation avec les mêmes paramètres que l’entraînement
5. inférence du modèle
6. filtrage par règles de sécurité
7. décision d’arrosage
8. activation du relais / pompe
9. journalisation du résultat

## Règles de sécurité
Même si le modèle propose d’arroser, bloquer l’action si :
- le sol est déjà trop humide ;
- une pluie forte est prévue ;
- un arrosage vient d’avoir lieu trop récemment ;
- une erreur capteur est détectée.

---

## 16. Spécification du modèle final à implémenter

### Nom du modèle
`TinyML_Irrigation_MLP_v1`

### Type
Réseau de neurones dense quantifié int8

### Entrée
Vecteur de features normalisées

### Sortie
3 classes :
- 0 : pas d’arrosage
- 1 : arrosage court
- 2 : arrosage long

### Avantages
- très léger ;
- portable ;
- facile à convertir ;
- adapté aux données tabulaires ;
- suffisant pour le besoin.

### Version de secours si difficulté mémoire
`DecisionTree_Irrigation_v1`

Ce modèle peut être utilisé comme fallback si le déploiement du réseau de neurones devient trop compliqué.

---

## 17. Choix technique recommandé pour votre équipe

### Choix principal
- **ML** : petit MLP quantifié
- **Runtime** : LiteRT / TensorFlow Lite Micro
- **Cible** : ESP32
- **Langage embarqué** : Arduino C++ ou ESP-IDF C++

### Pourquoi c’est le meilleur compromis
- l’équipe électronique sait déjà coder sur Arduino ;
- le modèle reste simple à appeler ;
- le passage Python → TFLite → C++ est standard ;
- le maintien dans le temps est plus simple qu’avec un modèle séquentiel.

---

## 18. Données de sortie attendues

Le firmware doit produire au minimum :

- classe prédite
- niveau d’arrosage
- durée d’activation de la pompe
- horodatage
- état des capteurs
- état Wi‑Fi / API météo
- message de diagnostic

Exemple de sortie série :

- `class=1`
- `pump_duration=45s`
- `soil=41%`
- `air_temp=32.4C`
- `air_humidity=58%`
- `decision=IRRIGATE`

---

## 19. Architecture du projet

### Dossier ML
- `data/` : données brutes et nettoyées
- `notebooks/` : exploration
- `training/` : scripts d’entraînement
- `export/` : modèle quantifié
- `metrics/` : résultats d’évaluation
- `docs/` : documentation technique

### Dossier firmware
- `sensors/`
- `network/`
- `model/`
- `actuator/`
- `main/`

---

## 20. Livrables attendus

### Livrables ML
- dataset structuré
- script de nettoyage
- script d’entraînement
- script d’évaluation
- modèle final exporté
- paramètres de normalisation
- documentation d’usage du modèle

### Livrables firmware
- lecture capteurs
- récupération météo
- intégration du modèle
- logique d’arrosage
- pilotage pompe/relais
- logs série
- code commenté

---

## 21. Critères d’acceptation

Le projet est considéré comme fonctionnel si :

- l’ESP32 lit correctement les capteurs ;
- les données météo sont récupérées si Wi‑Fi disponible ;
- le modèle donne une classe cohérente ;
- la pompe s’active avec la bonne durée ;
- le système fonctionne sur plusieurs cycles de 6 h ;
- les décisions restent stables et compréhensibles.

---

## 22. Plan de travail recommandé

### Phase 1 — définition
- figer les capteurs et les variables d’entrée ;
- définir les classes de sortie ;
- définir la cadence de décision (6 h).

### Phase 2 — collecte
- collecter les données capteurs ;
- récupérer la météo externe ;
- archiver tout avec horodatage.

### Phase 3 — préparation
- nettoyer ;
- aligner ;
- dériver les features ;
- construire les labels.

### Phase 4 — entraînement
- entraîner un MLP léger ;
- comparer avec un modèle de référence simple ;
- sélectionner le meilleur compromis.

### Phase 5 — export
- quantifier ;
- convertir ;
- préparer le fichier modèle pour ESP32.

### Phase 6 — intégration
- intégrer dans Arduino/ESP-IDF ;
- tester la prédiction ;
- brancher la pompe.

### Phase 7 — validation terrain
- mesurer les résultats ;
- corriger ;
- réentraîner ;
- stabiliser.

---

## 23. Risques principaux et mitigation

### Risque 1 : données insuffisantes
**Mitigation** : démarrer avec une règle experte et réentraîner dès que possible.

### Risque 2 : capteurs bruyants
**Mitigation** : filtrage, moyenne glissante, détection d’anomalies.

### Risque 3 : absence de météo externe
**Mitigation** : prévoir un mode dégradé autonome.

### Risque 4 : modèle trop lourd pour l’ESP32
**Mitigation** : réseau plus petit, quantification int8, ou fallback arbre de décision.

### Risque 5 : mauvaise logique d’arrosage
**Mitigation** : couche de règles de sécurité au-dessus du modèle.

---

## 24. Décision finale recommandée

### Modèle exact recommandé
**Petit réseau de neurones dense quantifié int8, de type TinyML MLP, avec sortie en 3 classes d’irrigation.**

### Pourquoi c’est le meilleur choix
- adapté à l’ESP32 ;
- simple à déployer par une équipe Arduino ;
- assez puissant pour apprendre à partir des capteurs ;
- compatible avec une logique d’arrosage toutes les 6 heures ;
- bon compromis entre précision, simplicité et maintenance.

---

## 25. Annexes techniques

### Exemple de features finales
- `soil_moisture`
- `air_temp`
- `air_humidity`
- `pressure`
- `rain_6h`
- `rain_24h`
- `wind_speed`
- `et0`
- `hour_sin`
- `hour_cos`
- `weekday_sin`
- `weekday_cos`

### Exemple de labels
- `0` : pas d’arrosage
- `1` : arrosage court
- `2` : arrosage long

### Exemple de seuils initiaux pour génération de labels
- sol sec + pluie faible + ET0 forte → classe 2
- sol moyennement sec + météo neutre → classe 1
- sol humide ou pluie proche → classe 0

---

## 26. Références techniques conseillées

- LiteRT / TensorFlow Lite for Microcontrollers
- ESP32 / ESP-IDF / Arduino ESP32
- Documentation capteur BME280
- Documentation capteur DHT22
- Documentation du capteur d’humidité du sol utilisé par votre équipe
- API météo externe choisie pour pluie, vent et ET0

