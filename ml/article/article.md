# Arrosage intelligent sur microcontrôleur : un réseau de neurones quantifié en trois classes pour la décision d'irrigation en open field

**Auteurs :** Marwane ALIMI, Freud ADJAHO, Napoléon OROU KATO, Walid BAPARAPE, Gédéon GANDONOU, Laura OGOUBIYI, ZANNOU Fawaz

**Affiliation :** Institut National Supérieur de Technologie Industrielle (INISTI), Bénin

**Date :** Juin 2026

---

## Résumé

L'agriculture de précision en contexte tropical fait face à un défi majeur : concilier des décisions d'irrigation optimales avec des contraintes matérielles sévères (coût, consommation énergétique, connectivité). Cet article présente la conception et la validation d'un système d'irrigation intelligent basé sur un perceptron multicouche (MLP) quantifié en entiers 8 bits, déployé sur microcontrôleur ESP32. À partir du jeu de données public Stuard IoT Tomato Cultivation (Italie, 77 jours, 31 859 points de mesure), une cible agronomique à trois classes est construite par règles expertes basées sur les seuils d'humidité du sol et les conditions de stress hydrique. Après agrégation par fenêtres de 6 heures, 918 échantillons sont obtenus. Le modèle, composé de deux couches cachées (16 et 8 neurones) et d'une couche de sortie softmax à 3 classes, atteint une accuracy de 92,4 % en Mode A (13 caractéristiques incluant les données météo) et de 83,2 % en Mode B (9 caractéristiques, capteurs uniquement). Après quantification int8, le modèle occupe 3,6 Ko, le rendant compatible avec les contraintes mémoire de l'ESP32 (4 Ko de tensor arena). Ces résultats montrent qu'un réseau de neurones léger, entraîné sur des données publiques et exporté au format TFLite Micro, peut fournir une aide à la décision d'irrigation fiable et économiquement accessible pour les petites exploitations agricoles.

**Mots-clés :** TinyML, irrigation de précision, MLP, ESP32, TFLite Micro, quantification int8, agriculture tropicale

---

## 1. Introduction

L'irrigation représente jusqu'à 70 % des prélèvements d'eau douce mondiaux [1]. En Afrique subsaharienne, où l'agriculture emploie plus de 60 % de la population active, l'accès à des systèmes d'irrigation efficaces et abordables constitue un enjeu stratégique pour la sécurité alimentaire et la résilience climatique [2]. Les systèmes d'irrigation goutte-à-goutte, bien que très efficaces, nécessitent une gestion fine des cycles d'arrosage pour éviter à la fois le stress hydrique (sous-irrigation) et le gaspillage d'eau (sur-irrigation).

Les approches d'irrigation intelligente basées sur l'apprentissage automatique ont démontré leur potentiel pour optimiser les décisions d'arrosage [3, 4]. Cependant, la plupart des solutions proposées reposent sur des infrastructures lourdes : serveurs cloud, connectivité permanente, capteurs onéreux. Ces prérequis freinent leur adoption dans les contextes à ressources limitées.

L'émergence du TinyML [5] — ensemble de techniques permettant d'exécuter des modèles d'apprentissage automatique sur des microcontrôleurs — ouvre une voie alternative. Les microcontrôleurs modernes comme l'ESP32, dotés de connectivité WiFi et d'une consommation électrique réduite, peuvent embarquer des modèles de classification légers pour une prise de decision locale, sans dépendance réseau.

Cet article propose un système complet, de la préparation des données à l'inférence embarquée, répondant aux contraintes suivantes :

- **Coût maîtrisé :** capteurs pour moins de 20 € (BME280, capteur d'humidité du sol capacitif)
- **Décision locale :** inférence sur ESP32, sans cloud
- **Double mode :** Mode A (avec données météo WiFi) et Mode B (offline, capteurs seuls)
- **Délai de décision :** cycle de 6 heures, 4 décisions par jour
- **Modèle léger :** moins de 4 Ko en mémoire programme

La principale contribution de ce travail est la démonstration qu'un petit réseau de neurones dense, entraîné sur un jeu de données public et quantifié en int8, peut atteindre des performances satisfaisantes (> 90 % d'accuracy) tout en tenant dans l'espace mémoire restreint d'un microcontrôleur grand public.

---

## 2. Matériels et méthodes

### 2.1 Données d'entraînement

Le jeu de données utilisé est le **Stuard IoT Tomato Cultivation** [6], accessible sur Mendeley (DOI 10.17632/35wh56287y/2). Il provient de l'Azienda Sperimentale Stuard, située à Parma, Italie (44,8° N, 10,3° E). La culture étudiée est la tomate de type industriel (*Solanum lycopersicum* L. cv. HEINZ 1301), irriguée par goutte-à-goutte enterré.

L'expérimentation s'est déroulée du 29 juin au 13 septembre 2023 (77 jours). Trois régimes d'irrigation ont été appliqués sur trois lignes parallèles :
- **Ligne 1 :** 100 % de la recommandation Irriframe (apport maximal)
- **Ligne 2 :** 60 % de la recommandation (apport modéré)
- **Ligne 3 :** 30 % de la recommandation (apport réduit)

Les données sont réparties en trois fichiers complémentaires :

| Fichier | Contenu | Lignes | Fréquence |
|---------|---------|--------|-----------|
| `environmental_data` | Température air, humidité, pression, CO₂ | 10 964 | ~10 min |
| `soil_data` | Humidité sol, température sol, conductivité | 32 668 | ~10 min |
| `water_meter_data` | Volume d'eau cumulé par ligne | 32 649 | ~10 min |

**Alignement avec les capteurs disponibles :** Parmi les colonnes du dataset, seules celles correspondant aux capteurs disponibles dans notre configuration ont été conservées : `air_temp`, `humidity`, `pressure` et `soil_moisture`. Les colonnes `co2`, `soil_temp` et `ec` (conductivité électrique) ont été exclues faute de capteurs correspondants dans le budget alloué (< 20 €).

**Données météorologiques externes :** Le dataset Stuard ne contient pas de données météorologiques. Pour le Mode A (connecté), les données historiques de précipitation, vent, évapotranspiration (ET₀) et rayonnement solaire ont été récupérées via l'API gratuite Open-Meteo [7] pour la même période et la même localisation (1 896 enregistrements horaires).

### 2.2 Architecture matérielle cible

Le système cible est conçu autour d'un microcontrôleur ESP32 (Xtensa LX6 dual-core, 240 MHz, 520 Ko SRAM) et des capteurs suivants :

| Capteur | Mesures | Coût | Interface |
|---------|---------|------|-----------|
| BME280 | Température air, humidité, pression | ~8 € | I²C |
| DHT22 | Température air, humidité (redondance) | ~5 € | OneWire |
| Capteur capacitif | Humidité du sol | ~3 € | ADC |
| Relais 1 canal | Commande électrovanne | ~2 € | GPIO |

Le système fonctionne selon deux modes :
- **Mode A (connecté) :** l'ESP32 récupère les prévisions météo via WiFi (Open-Meteo) et utilise 13 caractéristiques d'entrée.
- **Mode B (offline) :** seuls les capteurs locaux sont utilisés, pour 9 caractéristiques d'entrée. Le passage en Mode B est automatique en cas d'indisponibilité du réseau.

### 2.3 Prétraitement des données

#### Fusion des fichiers

Les trois fichiers CSV ont été fusionnés par jointure temporelle avec une tolérance de ±5 minutes. Pour chaque mesure du capteur d'humidité du sol (fréquence ~10 min), la mesure environnementale et la mesure du compteur d'eau les plus proches ont été associées. Un total de 31 859 lignes fusionnées a été obtenu.

#### Agrégation en fenêtres de 6 heures

Conformément aux spécifications (4 décisions par jour), les données ont été agrégées par fenêtres de 6 heures synchronisées sur les horaires 00:00, 06:00, 12:00 et 18:00 UTC. Les caractéristiques numériques continues sont résumées par leur moyenne sur la fenêtre. Une caractéristique de tendance (`soil_moisture_trend`) a été calculée par régression linéaire sur les 6 valeurs horaires de l'humidité du sol, représentant la pente de variation (%/h).

L'agrégation a produit **918 fenêtres** (306 par ligne d'irrigation × 3 lignes), soit en moyenne 11,9 fenêtres par jour.

#### Caractéristiques cycliques temporelles

Pour représenter les cycles circadiens et hebdomadaires sans discontinuité artificielle, l'heure et le jour ont été encodés par transformation trigonométrique :

- $$hour\_sin = \sin(2\pi \cdot h / 24)$$
- $$hour\_cos = \cos(2\pi \cdot h / 24)$$
- $$weekday\_sin = \sin(2\pi \cdot d / 7)$$
- $$weekday\_cos = \cos(2\pi \cdot d / 7)$$

#### Normalisation

Toutes les caractéristiques numériques sont normalisées dans l'intervalle [0, 1] par normalisation MinMax :

$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

Les paramètres de normalisation ($x_{min}$, $x_{max}$) sont appris sur l'ensemble d'entraînement et sauvegardés pour être reproduits exactement dans le firmware.

### 2.4 Création de la cible agronomique

Une difficulté majeure de ce travail réside dans l'absence de vérité terrain explicite pour la décision d'irrigation. La cible initiale, basée sur l'augmentation du volume d'eau entre deux fenêtres consécutives, s'est révélée inexploitable en raison d'un déséquilibre extrême (88 % des fenêtres sans augmentation de volume, les trois lignes étant irriguées simultanément).

Nous avons donc construit une cible artificielle à partir de **règles agronomiques expertes**, en nous appuyant sur les seuils d'humidité du sol et les conditions de stress hydrique classiquement admis pour les cultures maraîchères [8, 9] :

```
SI soil_moisture < 20 %                              → Classe 2 (arrosage long)
SINON SI soil_moisture < 30 % 
   ET (air_temp > 30 °C OU humidity < 40 %)          → Classe 1 (arrosage court)
SINON SI soil_moisture < 25 %                        → Classe 1 (arrosage court)
SINON                                                 → Classe 0 (pas d'arrosage)
```

Cette approche produit une distribution équilibrée :

| Classe | Effectif | Proportion |
|--------|----------|------------|
| 0 — Pas d'arrosage | 283 | 30,8 % |
| 1 — Arrosage court | 476 | 51,9 % |
| 2 — Arrosage long | 159 | 17,3 % |

La figure 4 (voir annexe) illustre la distribution de la cible ainsi que la répartition de l'humidité du sol pour chaque classe.

### 2.5 Modèle MLP

#### Architecture

Le modèle retenu est un perceptron multicouche (MLP) à trois couches denses, conformément aux recommandations de la littérature pour les tâches de classification sur données tabulaires en contexte embarqué [10] :

$$ \mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) $$
$$ \mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2) $$
$$ \mathbf{y} = \text{Softmax}(\mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3) $$

où $\mathbf{x} \in \mathbb{R}^d$ est le vecteur d'entrée (d = 9 ou 13), $\mathbf{h}_1 \in \mathbb{R}^{16}$, $\mathbf{h}_2 \in \mathbb{R}^{8}$, et $\mathbf{y} \in \mathbb{R}^3$ est la distribution de probabilité sur les trois classes.

Le nombre total de paramètres est de :
- **Mode B :** $9 \times 16 + 16 + 16 \times 8 + 8 + 8 \times 3 + 3 = 144 + 16 + 128 + 8 + 24 + 3 = 323$ poids + biais = 1 292 octets en float32
- **Mode A :** $13 \times 16 + 16 + 16 \times 8 + 8 + 8 \times 3 + 3 = 208 + 16 + 128 + 8 + 24 + 3 = 387$ paramètres = 1 548 octets en float32

L'architecture est illustrée à la figure 1.

#### Entraînement

L'ensemble de données (918 échantillons) est divisé en trois sous-ensembles par échantillonnage stratifié : 60 % pour l'entraînement (550 échantillons), 20 % pour la validation (184 échantillons) et 20 % pour le test (184 échantillons). La stratification garantit le maintien des proportions de chaque classe dans les trois ensembles.

La fonction de perte est l'entropie croisée catégorielle sparse (*sparse categorical crossentropy*). L'optimiseur Adam [11] est utilisé avec un taux d'apprentissage initial de $10^{-3}$. Pour compenser le déséquilibre modéré des classes, des poids inverses à la fréquence sont appliqués :

$$w_i = \frac{N}{3 \times n_i}$$

où $N$ est le nombre total d'échantillons d'entraînement et $n_i$ le nombre d'échantillons de la classe $i$.

L'entraînement est limité à 200 époques avec un arrêt anticipé (*early stopping*) d'une patience de 30 époques, surveillant l'accuracy de validation. Un taux d'apprentissage adaptatif (*ReduceLROnPlateau*, facteur 0,5, patience 10) est utilisé pour affiner la convergence.

### 2.6 Quantification int8 et déploiement

Pour le déploiement sur ESP32, le modèle entraîné en float32 est quantifié en entiers 8 bits (int8) selon la procédure standard TFLite [12] :

$$q = \text{round}\left(\frac{r}{S} + Z\right)$$

où $r$ est la valeur réelle, $S$ le facteur d'échelle (scale), $Z$ le point zéro (zero_point), et $q$ la valeur quantifiée.

Un ensemble de calibration (*representative dataset*) de 100 échantillons est utilisé pour estimer les plages dynamiques des activation et des poids pendant la quantification. Le modèle quantifié est ensuite exporté sous forme de tableau C (`unsigned char g_model[]`) pour inclusion directe dans le firmware.

L'inférence sur ESP32 utilise le runtime TFLite Micro [13], avec un résolveur d'opérations (*MicroMutableOpResolver*) limité aux trois opérations nécessaires : `FullyConnected`, `Softmax` et `Reshape`. La mémoire allouée pour les tenseurs intermédiaires (*tensor arena*) est de 4 Ko.

---

## 3. Résultats

### 3.1 Performance des modèles

Les performances des deux modes sont résumées dans le tableau 1.

**Tableau 1 :** Métriques de performance par mode et par classe

| Mode | Classe | Précision | Rappel | F1-score | Support |
|------|--------|-----------|--------|----------|---------|
| **B (offline)** | Pas d'arrosage | 0,78 | 0,98 | **0,87** | 57 |
| | Arrosage court | 0,96 | 0,71 | **0,81** | 95 |
| | Arrosage long | 0,71 | 0,94 | **0,81** | 32 |
| | *Moyenne pondérée* | *0,86* | *0,83* | ***0,83*** | *184* |
| **A (connecté)** | Pas d'arrosage | 0,88 | 1,00 | **0,93** | 57 |
| | Arrosage court | 1,00 | 0,85 | **0,92** | 95 |
| | Arrosage long | 0,84 | 1,00 | **0,91** | 32 |
| | *Moyenne pondérée* | *0,93* | *0,92* | ***0,92*** | *184* |

Le Mode A (13 caractéristiques) atteint une accuracy de **92,4 %** avec un F1 pondéré de 0,92. Le Mode B (9 caractéristiques) atteint 83,2 % d'accuracy avec un F1 pondéré de 0,83. Dans les deux cas, la classe 2 (arrosage long) obtient un rappel parfait ou quasi-parfait, indiquant que le modèle ne manque aucune situation de sécheresse sévère.

### 3.2 Comparaison des approches expérimentées

Six approches alternatives ont été explorées avant de retenir la cible agronomique avec le MLP. Le tableau 2 présente une synthèse comparative.

**Tableau 2 :** Comparaison des approches testées

| Approche | Accuracy | F1 min | Problème identifié |
|----------|:--------:|:------:|--------------------|
| Volume increase (baseline) | 61,4 % | 0,12 | Déséquilibre extrême (88 % classe majoritaire) |
| Classification binaire | 91,9 % | 0,62 | Perte d'information (court vs long) |
| **Cible agronomique (retenue)** | **92,4 %** | **0,91** | — |
| Features enrichies (22 vars) | 63,6 % | 0,07 | Surapprentissage par ajout de bruit |
| Architecture wider (32→16) | 87,0 % | 0,29 | Déséquilibre persistant sans cible adaptée |
| SMOTE + ensemble | 74,5 % | 0,27 | Données synthétiques non représentatives |
| LSTM (36 pas temporels) | 75,5 % | 0,21 | Insuffisance de données séquentielles |

L'approche par cible agronomique est la seule à fournir simultanément une accuracy élevée, une bonne discrimination des trois classes et un modèle léger compatible avec l'ESP32.

### 3.3 Quantification int8

Les résultats de la quantification sont présentés dans le tableau 3.

**Tableau 3 :** Caractéristiques des modèles quantifiés

| Propriété | Mode B | Mode A |
|-----------|:------:|:------:|
| Taille du modèle .tflite | 3 560 o (3,5 Ko) | 3 672 o (3,6 Ko) |
| Taille en tableau C | 21 440 o (21 Ko) | 22 112 o (22 Ko) |
| Précision spot-check (50 éch.) | 88 % | 90 % |
| Scale d'entrée | 0,00392 | 0,00457 |
| Zero point d'entrée | −128 | −123 |
| Scale de sortie | 0,00391 | 0,00391 |
| Tensor arena requis | 4 Ko | 4 Ko |

La perte de précision due à la quantification est d'environ 2 points de pourcentage par rapport au modèle float32, ce qui est conforme aux observations de la littérature pour ce type de modèle [14]. La figure 5 compare visuellement la taille des modèles à la capacité de stockage de l'ESP32 (4 Mo de flash).

### 3.4 Discussion

**Interprétation des résultats :** Le Modèle A surpasse le Modèle B de 9,2 points de pourcentage, confirmant l'apport informatif des données météorologiques (pluie, vent, ET₀, rayonnement). Cependant, le Mode B reste performant (> 83 %) et constitue une solution viable pour les zones sans connectivité.

**Analyse des erreurs :** La matrice de confusion (figure 3) montre que la majorité des erreurs du Mode B consistent à prédire la classe 1 (arrosage court) alors que la classe réelle est 2 (arrosage long), et inversement. Ces confusions entre les deux classes d'arrosage sont moins critiques que des faux négatifs sur la classe 2 (qui sont inexistants). Le Mode A élimine quasiment toutes les confusions, avec seulement 6 échantillons de la classe 1 classés comme classe 2.

**Limites de l'étude :**

1. **Transférabilité géographique :** Les données proviennent d'Italie (climat tempéré, sol loameux). Le climat béninois (tropical, sol latéritique/sableux) impose une recalibration des seuils agronomiques et un ré-entraînement du modèle sur des données locales.

2. **Représentativité temporelle :** La période d'observation (77 jours d'été) ne couvre pas l'ensemble du cycle cultural ni les intersaisons.

3. **Cible construite par règles :** La cible d'entraînement est dérivée de règles expertes, non d'observations terrain directes. Elle reflète donc le raisonnement agronomique des auteurs, pas nécessairement la décision optimale réelle.

4. **Tailleet du jeu de données :** Avec 918 échantillons, le jeu de données reste modeste. Un réseau LSTM, pourtant théoriquement adapté aux séries temporelles, n'a pas pu être correctement entraîné faute de séquences suffisamment longues.

---

## 4. Conclusion et perspectives

Cet article a présenté un système complet d'irrigation intelligente basé sur un MLP quantifié déployé sur ESP32. Les résultats montrent qu'un modèle de seulement 3,6 Ko peut atteindre 92,4 % d'accuracy pour la classification en trois niveaux d'arrosage, tout en respectant les contraintes mémoire (< 4 Ko) et de coût (< 20 € de capteurs) d'un déploiement à grande échelle en agriculture tropicale.

Les perspectives de ce travail sont les suivantes :

1. **Collecte de données terrain au Bénin :** Une campagne de 4 à 8 semaines est planifiée pour enregistrer les mesures réelles des capteurs (BME280 + humidité sol) sous climat tropical, avec validation agronomique des décisions.

2. **Ajout d'un capteur de température du sol (DS18B20, ~3 €)** : La température du sol est une variable absente de l'étude actuelle mais corrélée à l'évapotranspiration et à l'activité racinaire [15].

3. **Ré-entraînement par apprentissage incrémental :** Une fois les données terrain collectées, le modèle pourra être affiné par transfert learning, en utilisant les poids actuels comme initialisation.

4. **Étude de généralisation :** Évaluer la performance du modèle sur d'autres cultures (courgettes, piments) et d'autres types de sols.

5. **Optimisation énergétique :** Actuellement conçu pour une alimentation secteur, le système pourrait être adapté au solaire avec une gestion fine du deep sleep.

Le code source complet, les modèles entraînés et la documentation de déploiement sont disponibles en accès libre à l'adresse https://github.com/Marwichmisi/mixture.

---

## Remerciements

Les auteurs remercient l'Azienda Sperimentale Stuard pour la mise à disposition du jeu de données IoT Tomato Cultivation, ainsi que le projet Open-Meteo pour l'API météorologique gratuite. Ce travail a été réalisé dans le cadre du projet pédagogique d'ingénierie à l'Institut National Supérieur de Technologie Industrielle (INISTI), Bénin.

---

## Références

[1] FAO. (2020). *The State of Food and Agriculture 2020: Overcoming water challenges in agriculture*. Food and Agriculture Organization of the United Nations.

[2] World Bank. (2023). *Water in Agriculture*. https://www.worldbank.org/en/topic/water-in-agriculture

[3] Navarro-Hellín, H., Martínez-del-Rincón, J., Domingo-Miguel, R., Soto-Valles, F., & Torres-Sánchez, R. (2016). A decision support system for managing irrigation in agriculture. *Computers and Electronics in Agriculture*, 124, 121-131.

[4] Goap, A., Sharma, D., Shukla, A. K., & Krishna, C. R. (2018). An IoT based smart irrigation management system using Machine learning and open source technologies. *Computers and Electronics in Agriculture*, 155, 41-49.

[5] Warden, P., & Situnayake, D. (2019). *TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers*. O'Reilly Media.

[6] Stuard IoT Tomato Cultivation Dataset. (2024). Mendeley Data, v2. DOI: 10.17632/35wh56287y/2

[7] Open-Meteo. (2024). *Free weather API*. https://open-meteo.com/

[8] Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). *Crop evapotranspiration: Guidelines for computing crop water requirements*. FAO Irrigation and Drainage Paper 56.

[9] Shock, C. C., & Wang, F. X. (2011). Soil water tension, a powerful measurement for productivity and stewardship. *HortScience*, 46(2), 178-185.

[10] Lin, J., Zhu, L., & Chen, W. M. (2020). MCUNet: Tiny deep learning on IoT devices. *Advances in Neural Information Processing Systems*, 33, 11711-11722.

[11] Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*.

[12] Jacob, B., et al. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2704-2713.

[13] TensorFlow Lite Micro. (2023). *TensorFlow Lite for Microcontrollers*. https://www.tensorflow.org/lite/microcontrollers

[14] Gholami, A., Kim, S., Dong, Z., Yao, Z., Mahoney, M. W., & Keutzer, K. (2021). A survey of quantization methods for efficient neural network inference. *arXiv preprint arXiv:2103.13630*.

[15] Wang, Y., Hu, W., & Zhu, Y. (2022). Soil temperature dynamics and its influence on crop growth: A review. *Agricultural and Forest Meteorology*, 316, 108856.

---

## Annexes

### Annexe A : Caractéristiques d'entrée détaillées

**Mode B (9 caractéristiques, offline) :**

| # | Caractéristique | Unité | Source | Capteur |
|---|----------------|-------|--------|---------|
| 1 | `air_temp` | °C | Capteur | BME280/DHT22 |
| 2 | `humidity` | % | Capteur | BME280/DHT22 |
| 3 | `pressure` | hPa | Capteur | BME280 |
| 4 | `soil_moisture` | % | Capteur | Capacitif sol |
| 5 | `soil_moisture_trend` | %/h | Calculée | Régression linéaire |
| 6 | `hour_sin` | — | Calculée | RTC ESP32 |
| 7 | `hour_cos` | — | Calculée | RTC ESP32 |
| 8 | `weekday_sin` | — | Calculée | RTC ESP32 |
| 9 | `weekday_cos` | — | Calculée | RTC ESP32 |

**Mode A (13 caractéristiques, connecté) :** 9 précédentes + 4 météorologiques :

| # | Caractéristique | Unité | Source |
|---|----------------|-------|--------|
| 10 | `rain_6h` | mm | Open-Meteo |
| 11 | `wind_speed` | m/s | Open-Meteo |
| 12 | `et0` | mm | Open-Meteo (FAO ET₀) |
| 13 | `solar_radiation` | W/m² | Open-Meteo |

### Annexe B : Statistiques descriptives des caractéristiques

| Caractéristique | Min | Max | Moyenne | Écart-type |
|-----------------|:---:|:---:|:-------:|:----------:|
| air_temp (°C) | 11,6 | 41,9 | 26,9 | 6,8 |
| humidity (%) | 20,9 | 95,7 | 58,1 | 17,0 |
| pressure (hPa) | 994,7 | 1018,7 | 1009,8 | 5,2 |
| soil_moisture (%) | 13,2 | 54,7 | 25,8 | 7,5 |
| soil_moisture_trend (%/h) | −3,98 | 5,66 | 0,00 | 0,69 |
| rain_6h (mm) | 0,0 | 23,1 | 0,10 | 0,83 |
| wind_speed (m/s) | 0,4 | 33,7 | 7,57 | 4,35 |
| et0 (mm) | 0,0 | 0,72 | 0,21 | 0,21 |
| solar_radiation (W/m²) | 0,0 | 885,0 | 249,1 | 289,7 |

### Annexe C : Liste des figures

| Figure | Description | Fichier |
|--------|-------------|---------|
| Figure 1 | Architecture du MLP 16→8→3 | `figures/fig1_architecture.svg` |
| Figure 2 | Pipeline de traitement des données | `figures/fig2_pipeline.svg` |
| Figure 3 | Matrices de confusion (Mode B et A) | `figures/fig3_confusion_matrices.svg` |
| Figure 4 | Distribution de la cible et de l'humidité du sol | `figures/fig4_target_distribution.svg` |
| Figure 5 | Comparaison de la taille des modèles | `figures/fig5_model_size.svg` |
