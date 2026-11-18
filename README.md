# coEnergy — Prédire la performance thermique d'un bâtiment à partir de 8 h de mesure

Assessment de stage coEnergy. Ce dépôt contient l'analyse, le code, les modèles entraînés et les résultats
d'un projet qui estime **six indicateurs de performance énergétique** d'un bâtiment (HLC, sa
décomposition Hb, trois constantes de temps τ). Les entrées sont deux séries de 8 heures : la température
intérieure **T_int(t)** et la puissance de chauffe **U(t)**.

Le modèle livré est un **ensemble pondéré de réseaux CNN-GRU (et un TCN) implémentés en PyTorch**, entraîné sur
1 000 000 de simulations thermiques.

| R² sur le test | HLC | HLC_mit | HLC_ext | τ_b | τ_n | τ_∞ |
|---|---|---|---|---|---|---|
| Ridge (référence) | 0.313 | 0.245 | 0.071 | 0.011 | 0.113 | 0.007 |
| LightGBM (référence) | 0.349 | 0.254 | 0.136 | 0.024 | 0.142 | 0.015 |
| **Modèle final (CNN-GRU, PyTorch)** | **0.451** | **0.289** | **0.470** | **0.126** | **0.257** | **0.093** |

- **HLC** : 8 % d'erreur relative moyenne. Le score est confirmé sur un second test indépendant (R² = 0.452).
- **Constantes de temps** : accessibles seulement en ordre de grandeur (12 à 18 % d'erreur). La fenêtre
  de 8 h est trop courte devant les dynamiques lentes.
- **Bruit de capteur** : un modèle entraîné sur données parfaites s'effondre dès 0,05 °C de bruit. Le
  **modèle robuste** livré garde un R² HLC de 0.39 à 0,10 °C : c'est lui qu'il faut déployer sur des mesures
  réelles.

> Le R² vaut 1 pour une prédiction parfaite et 0 si l'on ne fait pas mieux que prédire la moyenne. Un R² de
> 0.45 signifie que le modèle explique 45 % des différences entre bâtiments.

---

## Sommaire

1. [Démarrage rapide](#1-démarrage-rapide)
2. [Le problème](#2-le-problème)
3. [La démarche, étape par étape](#3-la-démarche-étape-par-étape)
4. [Les choix importants, et pourquoi](#4-les-choix-importants-et-pourquoi)
5. [Installation détaillée](#5-installation-détaillée)
6. [Données](#6-données)
7. [Utilisation](#7-utilisation)
8. [Organisation du dépôt, fichier par fichier](#8-organisation-du-dépôt-fichier-par-fichier)
9. [Reproductibilité](#9-reproductibilité)
10. [Limites et pistes](#10-limites-et-pistes)

**Ordre de lecture conseillé** : ce README, puis `notebooks/rapport.ipynb` (physique et analyse
exploratoire), puis `notebooks/modelisation.ipynb` (méthode, modèle final, résultats, conclusion). Les
annexes et le code servent à vérifier un chiffre ou un détail.

---

## 1. Démarrage rapide

```bash
# Depuis le dossier livrable/
pip install uv          # une seule fois (ou l'installeur officiel, voir section 5)
uv sync                 # crée .venv/ avec Python 3.12 et les versions exactes de uv.lock
uv run pytest -q        # 14 tests, sans données ni GPU
```

Prédire avec le modèle livré. Les données ne sont pas nécessaires : il suffit de séries à la minute.

```python
import numpy as np
from coenergy.final_model import FinalModel

model = FinalModel.load("results/results_full/modele_robuste.joblib")   # ou modele_final.joblib
pred = model.predict(T_int, U)   # T_int en °C, U en W : tableaux (N, 480), un pas par minute
print(pred)                      # DataFrame (N, 6) : HLC, HLC_mit, HLC_ext, tau_b, tau_n, tau_inf
```

---

## 2. Le problème

On mesure pendant **8 heures** la température intérieure d'un bâtiment, T_int(t), pendant qu'on le chauffe
selon un programme connu, U(t). On veut en déduire sa performance thermique :

| Indicateur | Signification | Unité |
|---|---|---|
| **HLC** | coefficient de déperdition total : puissance perdue par degré d'écart de température | W/K |
| **HLC_ext** (Hb_ext) | part des déperditions vers l'extérieur | W/K |
| **HLC_mit** (Hb_mit) | part des déperditions vers les locaux mitoyens | W/K |
| **τ_b, τ_n, τ_∞** | constantes de temps : vitesse à laquelle le bâtiment se réchauffe ou se refroidit | h |

Contrainte physique, respectée exactement par le modèle : **HLC = HLC_ext + HLC_mit**.

**Les données.**
- **1 000 000 de simulations** (20 fichiers de 50 000) d'un même bâtiment type, « quentin_house ».
- Chaque simulation fait varier les propriétés physiques du bâtiment (isolation, inertie…) jusqu'à
  **±60 %** autour de leurs valeurs nominales (tirage Monte Carlo). Elle calcule T_int(t) avec un modèle
  thermique à état, C·dT/dt = A·T + E·U.
- Le chauffage U(t) est un signal en créneaux pseudo-aléatoire (PRBS) : allumé à 7 000 W ou éteint, avec
  des durées de palier variables.
- Entrées du modèle : T_int et U, 480 pas d'une minute. Sorties : les 6 indicateurs.

---

## 3. La démarche, étape par étape

```
 ① Comprendre la physique      ② Explorer les données          ③ Références simples
   (rapport.ipynb, §1)            (rapport.ipynb, §2-3)           (ridge, LightGBM)
            │                              │                              │
            └──────────────┬───────────────┴──────────────────────────────┘
                           ▼
 ④ Deep learning (annexe A2) → ⑤ Pistes d'amélioration (annexe A3) → ⑥ Modèle final (modelisation.ipynb)
```

| Étape | Où | Ce qu'on y apprend | Résultats dans |
|---|---|---|---|
| ① Physique | `rapport.ipynb` §1 | Ce que chaque indicateur change dans T_int(t) | — |
| ② Exploration | `rapport.ipynb` §2-3 | Doublons d'excitation, plafond d'information, forme des cibles | sorties du notebook |
| ③ Références | annexe A2 §5 | Score à battre sans deep learning | `results/results_dl/` |
| ④ Deep learning | annexe A2 | Le CNN-GRU est le meilleur ; Optuna ; analyse des erreurs | `results/results_dl/` |
| ⑤ Améliorations | annexe A3 | Le pas de 2 min aide ; les autres idées non | `results/results_idees/` |
| ⑥ Modèle final | `modelisation.ipynb` | Réentraînement sur tout, ensemble, test unique, robustesse | `results/results_full/` |

**① Comprendre la physique** (`rapport.ipynb`, section 1). On relie chaque indicateur au modèle
thermique :
- HLC fixe l'élévation de température à l'équilibre pour une puissance donnée ;
- les τ fixent la vitesse à laquelle on s'en approche (τ ≈ capacité thermique / HLC).

On en déduit ce que les données devraient montrer.

**② Explorer les données** (`rapport.ipynb`, sections 2 et 3). Trois constats guident toute la suite :
1. **Beaucoup de simulations partagent exactement le même programme de chauffe U(t).** Un découpage
   aléatoire mettrait le même programme en train et en test et gonflerait les scores. On découpe donc
   **par programme de chauffe** : chaque programme va entièrement en train, en validation ou en test.
2. **Il existe un plafond d'information.** 94 à 98 % des variations de T_int s'expliquent par le
   programme de chauffe, et seulement 2 à 6 % par le bâtiment. Deux causes limitent encore ce signal :
   - on ignore la chaleur déjà stockée dans les murs au début de la mesure ;
   - 8 heures, c'est court devant les constantes de temps lentes : on ne voit qu'environ 52 %, 23 % et
     15 % de leur réponse.
3. **Les cibles sont asymétriques et strictement positives** : on modélise les τ en logarithme.

**③ Références simples.** Avant tout réseau de neurones, on mesure ce que font des méthodes classiques
sur des **variables résumées** des séries : énergie injectée, montée de température, coefficients d'un
modèle autorégressif ARX.
- moyenne : R² = 0 par construction ;
- régression ridge : R² HLC = 0.313 ;
- LightGBM : R² HLC = 0.349.

**④ Deep learning** (`notebooks/annexes/A2_experiences_deep_learning.ipynb`). Des réseaux PyTorch
lisent **directement les séries** : CNN, TCN, CNN-GRU, et hybrides séquence + variables. Le **CNN-GRU**
est le meilleur :
- les convolutions détectent les réponses locales aux créneaux de chauffe ;
- le GRU intègre la dynamique lente.

Ses hyperparamètres sont optimisés avec **Optuna**, ce qui donne R² HLC = 0.43, contre 0.35 pour
LightGBM. L'annexe contient aussi une analyse des erreurs par segment de programme de chauffe, qui fonde
les recommandations de protocole de mesure.

**⑤ Pistes d'amélioration** (`notebooks/annexes/A3_pistes_amelioration.ipynb`) :
- un pas de temps plus fin (2 min au lieu de 5) : **gain retenu** ;
- un GRU bidirectionnel et des modèles « résiduels » : **pas de gain**, ce qui confirme le plafond
  d'information vu en ②.

**⑥ Modèle final** (`notebooks/modelisation.ipynb`).
- On réentraîne le meilleur CNN-GRU sur **toutes les données** (20 fichiers au lieu de 10).
- On teste une autre formulation des τ et on entraîne une version **robuste au bruit de mesure**.
- Les 8 meilleurs réseaux (7 CNN-GRU et 1 TCN) sont combinés en un **ensemble pondéré**, choisi sur la
  validation.
- L'ensemble est évalué **une seule fois** sur le test, puis sur un second test indépendant.

---

## 4. Les choix importants, et pourquoi

| Choix | Raison |
|---|---|
| Découpage par programme de chauffe | Un même U(t) en train et en test surestimerait les scores |
| Le réseau prédit HLC_ext et HLC_mit, puis HLC = leur somme | La contrainte physique est respectée exactement, par construction |
| τ en logarithme | Les τ sont positifs et très asymétriques |
| Réseau de séquence plutôt que variables résumées | L'information est dans la *forme* de la réponse : +0.10 de R² HLC |
| Validation séparée du test, test évalué une seule fois | Le score annoncé n'a influencé aucune décision |
| Second test indépendant | Vérifier que le score ne dépend pas de l'échantillon tiré |
| Modèle robuste (bruit ajouté à l'entraînement) | Un capteur réel est moins précis que la simulation (arrondie à 0,01 °C) |
| Deep learning exclusivement en PyTorch | Exigence de l'assessment ; scikit-learn et LightGBM ne servent que de références |

---

## 5. Installation détaillée

### Prérequis
- **uv** gère Python et les dépendances. Il installe lui-même Python 3.12 si besoin.
  - Windows : `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - Linux / macOS : `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Ou, si Python est déjà installé : `pip install uv`, puis `python -m uv …` si `uv` n'est pas dans le PATH.
- Pas besoin de GPU pour lire les notebooks, lancer les tests, l'analyse exploratoire ou le modèle livré.

### Les trois fichiers d'environnement

| Fichier | Rôle |
|---|---|
| `pyproject.toml` | Déclare le paquet (`coenergy-assessment`, importé sous le nom `coenergy`), ses dépendances minimales, l'extra `references`, le groupe `dev`, et la configuration de pytest et ruff |
| `uv.lock` | Fige la version et l'empreinte de **toutes** les dépendances, directes et transitives (152 paquets), pour Windows, Linux et macOS : chacun obtient le même environnement |
| `.python-version` | Version de Python utilisée : 3.12 (le paquet accepte ≥ 3.11) |

Contenu de `pyproject.toml` :

| Groupe | Paquets | Quand l'installer |
|---|---|---|
| Dépendances de base | torch, numpy, pandas, scipy, matplotlib, joblib, tqdm | Toujours (`uv sync`) |
| Extra `references` | scikit-learn, lightgbm, optuna | Pour réexécuter les annexes A2/A3 (ridge, LightGBM, Optuna) |
| Groupe `dev` | jupyterlab, ipykernel, pytest, ruff | Installé par défaut par `uv sync` |

### Commandes

```bash
uv sync                          # environnement de base + outils dev
uv sync --extra references       # + scikit-learn, LightGBM, Optuna (annexes)
uv run pytest -q                 # tests unitaires
uv run ruff check src tests      # style et erreurs courantes
```

Pour ajouter une dépendance : `uv add <paquet>` (ou `uv add --dev <paquet>`). La commande met à jour
`pyproject.toml` et `uv.lock`, qui se versionnent toujours ensemble. `uv lock --check` vérifie qu'ils sont
cohérents.

> **Note sur PyTorch.** Sous Windows et macOS, `uv.lock` installe la version CPU de PyTorch. Sous Linux, il
> installe la version CUDA standard de PyPI, avec ses bibliothèques NVIDIA (plusieurs Go). Les
> entraînements lourds ont été faits sur Kaggle (GPU T4), pas en local.

### Utiliser l'environnement dans VS Code ou Jupyter
- **JupyterLab** : `uv run jupyter lab notebooks/`
- **VS Code** : choisir l'interpréteur `livrable/.venv` comme noyau du notebook (*Select Kernel →
  Python Environments*). Si VS Code ne le propose pas, enregistrer le noyau une fois :
  ```bash
  .venv/Scripts/python -m ipykernel install --user --name coenergy --display-name "Python 3.12 (coenergy)"
  # Linux / macOS : .venv/bin/python -m ipykernel …
  ```
  puis choisir *Jupyter Kernel → Python 3.12 (coenergy)*. Une erreur
  `ModuleNotFoundError: No module named 'coenergy'` signifie que le notebook tourne sur un autre Python.

---

## 6. Données

Les données ne sont pas versionnées (`data/` est dans `.gitignore`). Il faut les placer ainsi :

```
livrable/
├── util_assessement.py          ← fourni avec le jeu de données (lecture des fichiers .pt)
└── data/
    └── quentin_house/
        ├── batch_1.pt           ← 20 fichiers (batch_1 à batch_20) de 50 000 simulations
        └── ...
```

`util_assessement.py` est le script fourni avec le jeu de données. Il enregistre le faux paquet
`thermal_dynamics_sim` dont pickle a besoin pour relire les fichiers `.pt`. `coenergy.data` l'importe
automatiquement.

**Mémoire.** Charger les 20 fichiers demande près de 4 Go en float32. Pour un premier essai sur un
ordinateur portable, réduire la configuration en tête de `rapport.ipynb` :

```python
DATA_CFG = DataConfig(n_batches=1, n_per_batch=10_000, ...)
```

L'analyse sur les 20 fichiers passe par `coenergy.scan`, qui lit les fichiers un par un.

---

## 7. Utilisation

### Notebooks

| Notebook | Contenu | Exécution |
|---|---|---|
| `notebooks/rapport.ipynb` | Sections 1 à 3 : physique du problème, chargement, analyse exploratoire | Local, avec les données dans `data/quentin_house/` |
| `notebooks/modelisation.ipynb` | Sections 4 à 7 : choix méthodologiques, modèle final, résultats, analyse critique, conclusion | Kaggle, GPU T4, 4,8 h ; sorties conservées. La section 6.3 s'exécute en local |
| `notebooks/annexes/A2_experiences_deep_learning.ipynb` | Références, architectures PyTorch, Optuna, stabilité entre graines, analyse des erreurs | Kaggle, GPU T4 ; sorties conservées |
| `notebooks/annexes/A3_pistes_amelioration.ipynb` | Pas de 2 min, GRU bidirectionnel, résidu à la réponse moyenne | Kaggle, GPU T4 ; sorties conservées |

**Lire sans réexécuter.** Les sorties des exécutions Kaggle sont conservées dans les notebooks, et les
fichiers produits sont copiés dans `results/`. Tout se lit donc sans GPU ni données.

**Réexécuter sur Kaggle.** Les notebooks de modélisation cherchent dans `/kaggle/input/` :
- le code du projet (dossier contenant `pyproject.toml`) ;
- les fichiers `batch_*.pt` ;
- pour `modelisation.ipynb`, les membres et découpages produits par A2 et A3.

Il faut donc ajouter ces jeux de données Kaggle au notebook, puis l'exécuter avec un GPU. Les résultats
sont écrits dans `/kaggle/working/` puis copiés dans `results/`.

### Modèle livré

| Fichier | Contenu | Usage |
|---|---|---|
| `results/results_full/modele_final.joblib` | Ensemble pondéré de 8 réseaux : 7 CNN-GRU et 1 TCN, au pas de 2 min ou de 5 min (meilleur score sur données propres) | Données simulées ou capteur très précis |
| `results/results_full/modele_robuste.joblib` | Un CNN-GRU au pas de 2 min, entraîné avec bruit capteur | **Mesures réelles** (recommandé) |

```python
from coenergy.final_model import FinalModel

model = FinalModel.load("results/results_full/modele_robuste.joblib")
df = model.predict(T_int, U, device="cpu")   # ou device="cuda"
```

- Entrées : `T_int` (°C) et `U` (W), tableaux NumPy de forme `(N, 480)`, un pas par minute.
- Sortie : DataFrame `(N, 6)`, colonnes `HLC, HLC_mit, HLC_ext, tau_b, tau_n, tau_inf` (W/K et h).
- Ne nécessite que les dépendances de base : ni scikit-learn, ni données, ni GPU.
- Chaque fichier contient les réseaux, les normalisations et la pondération : il est autonome.

### Tests

```bash
uv run pytest -q
```

Les 14 tests tournent sur des données synthétiques, sans le vrai jeu de données ni GPU :

| Fichier | Vérifie |
|---|---|
| `tests/test_pipeline.py` | Découpages disjoints et complets, normalisation réversible, formes du CNN-GRU, contrainte HLC = somme des Hb, métriques |
| `tests/test_modeling.py` | ARX (retrouve le gain statique), transformations de cibles, prétraitement tabulaire, découpage par groupe, ensembles et sauvegarde |
| `tests/test_final_model.py` | Le modèle final respecte HLC = HLC_ext + HLC_mit ; le bruit capteur n'est appliqué qu'à l'entraînement |
| `tests/test_variants.py` | La réponse moyenne de la population retrouve un système connu ; standardisation sur le train ; formes du GRU bidirectionnel |

---

## 8. Organisation du dépôt, fichier par fichier

```
livrable/
├── README.md                    ← ce fichier
├── pyproject.toml               ← paquet, dépendances, configuration pytest et ruff
├── uv.lock                      ← versions exactes de toutes les dépendances
├── .python-version              ← Python 3.12
├── .gitignore                   ← exclut .venv, data/, caches et poids intermédiaires
├── util_assessement.py          ← lecteur des fichiers .pt fourni avec les données
├── notebooks/                   ← rapport (voir section 7)
├── src/coenergy/                ← code du paquet
├── tests/                       ← tests unitaires
└── results/                     ← résultats chiffrés et modèles entraînés
```

### `src/coenergy/` : le code

Le code est organisé par rôle. Chaque module commence par une docstring qui décrit son contenu.

| Étape | Module | Rôle |
|---|---|---|
| Configuration | `config.py` | Hyperparamètres sous forme de dataclasses : `DataConfig`, `NoiseConfig`, `ModelConfig`, `TrainConfig` |
| | `utils.py` | Graines aléatoires, choix du device, nombre de paramètres |
| Données | `data.py` | Chargement batch par batch, sous-échantillonnage, découpage, normalisation, itérateur de batchs |
| | `scan.py` | Parcours en flux des 20 fichiers pour l'analyse exploratoire, avec empreintes pour détecter les doublons |
| Analyse exploratoire | `eda.py` | Plan d'expérience, doublons, corrélations de Spearman, spectre, modèle RC, ACP des courbes, profils d'information |
| | `features.py` | Descripteurs physiques de l'excitation et de la réponse en température |
| | `plots.py` | Figures matplotlib de l'exploration et de l'évaluation |
| Variables et cibles | `fe.py` | Descripteurs tabulaires : physiques, coefficients ARX, séries brutes ; prétraitement |
| | `targets.py` | Paramétrisations des cibles (A : directe, B : physique) ; garantit HLC = Hb0 + Hb1 et la positivité |
| Modèles | `nets.py` | Réseaux PyTorch (CNN-GRU, TCN, CNN, MLP, hybride), boucle d'entraînement `fit_torch`, espaces de recherche Optuna |
| | `baselines.py`, `gbm.py` | Références : moyenne, ridge, LightGBM (et XGBoost, CatBoost, forêts, SVR, non retenus). Imports à la demande |
| | `ensemble.py` | Ensemble pondéré (poids positifs de somme 1) et stacking ridge |
| | `variants.py` | Variantes de l'annexe A3 : réponse moyenne de la population, standardisation des canaux, encodeur GRU bidirectionnel (`BiGRUEncoder`) |
| Évaluation | `evaluation.py` | Découpage par motif d'excitation, sauvegarde des splits, métriques par indicateur et par segment, journal |
| | `metrics.py` | Métriques par indicateur, par famille et par sous-groupe ; contrôle de la contrainte HLC |
| Orchestration | `runner.py` | `ExperimentContext` : données, découpage, journal et prédictions pour les annexes. Toute statistique est ajustée sur le train, tout choix sur la validation |
| Livraison | `final_model.py` | Augmentation par bruit capteur, entraînement des membres, chargement et prédiction (`FinalModel`) |
| | `predict.py` | Prédicteur autonome générique (`Predictor`, `Member`) utilisé par l'ensemble |
| Première version | `model.py`, `losses.py`, `train.py` | Pipeline CNN-GRU initial (multi-têtes, perte multi-tâches, entraînement). Utilisé par `rapport.ipynb` et `tests/test_pipeline.py` |

### `results/` : ce qu'a produit chaque phase

| Dossier | Phase | Fichiers principaux |
|---|---|---|
| `results_dl/` | Annexe A2 | `experiences_dl.csv` (journal de toutes les expériences), `tableau_comparatif_dl.csv` (validation), `resultats_test_dl.csv`, `optuna_dl.csv`, `graines.csv` (stabilité), `robustesse_bruit_dl.csv`, `segment_*.csv` (erreurs par segment de programme de chauffe), `splits.npz`, `predictions.npz`, `selection_dl.json`, `modele_final_dl.joblib`, `erreurs.log` (étape en échec pendant l'exécution Kaggle : le segment « quintile de HLC », voir A2 section 12) |
| `results_idees/` | Annexe A3 | `comparaison_idees.csv`, `experiences_idees.csv`, `predictions.npz`, `reponse_population.npz` |
| `results_full/` | Modèle final | `modele_final.joblib`, `modele_robuste.joblib`, `validation_modeles.csv`, `selection_final.json` (membres de l'ensemble), `resultats_test_final.csv`, `second_test_final.csv`, `robustesse_bruit_final.csv`, `splits_final.npz` |

Les poids intermédiaires (`results/**/membres/`, `results/**/poids/`) servent à réexécuter
`modelisation.ipynb`. Ils sont exclus de Git par `.gitignore` et ne sont pas nécessaires pour utiliser le
modèle livré.

### Où trouver quoi

| Je cherche… | Où |
|---|---|
| Les figures et constats de l'analyse exploratoire | `notebooks/rapport.ipynb`, section 3 (synthèse en 3.13) |
| Le score final et sa justification | `notebooks/modelisation.ipynb`, sections 5 et 6 |
| L'architecture du réseau | `src/coenergy/nets.py` (`SeqEncoder`, `HybridNet`) |
| Le chargement et le découpage des données | `src/coenergy/data.py`, `src/coenergy/evaluation.py` |
| La garantie HLC = HLC_ext + HLC_mit | `src/coenergy/targets.py` |
| Comment utiliser le modèle livré | section 7 ci-dessus et `src/coenergy/final_model.py` |
| Les chiffres bruts | `results/*/*.csv` |

---

## 9. Reproductibilité

- **Environnement** : `uv.lock` fige toutes les versions ; `uv sync` recrée le même environnement.
- **Aléa** : graines fixées (`coenergy.utils.set_seed`, `seed=42` par défaut). Le meilleur modèle est
  réentraîné avec 3 graines pour mesurer la variabilité (`results_dl/graines.csv`).
- **Découpages** : sauvegardés (`results/*/splits*.npz`) et réutilisés à l'identique d'une phase à l'autre.
  Aucun programme de chauffe n'apparaît dans deux ensembles.
- **Pas de fuite** : normalisations et statistiques ajustées sur le seul train, puis sauvegardées avec
  chaque modèle. Choix de modèles et de poids sur la validation. Test évalué une seule fois.
- **Qualité du code** : tests unitaires (`pytest`) et linter (`ruff`, règles E, F, I, UP, B, lignes de 100
  caractères).

---

## 10. Limites et pistes

- **Un seul bâtiment type simulé.** La validité sur d'autres bâtiments et sur des mesures réelles reste à
  démontrer.
- **Constantes de temps lentes.** τ_b et τ_∞ sont mal déterminées avec 8 h de mesure : le signal ne
  contient pas l'information, ce n'est pas une limite du modèle (voir les essais négatifs de l'annexe A3).
- **Bruit de mesure.** Le modèle robuste réduit fortement la sensibilité au bruit, mais au prix de quelques
  points de R² sur données parfaites.
- **Pour aller plus loin**, côté protocole de mesure : une mesure plus longue, une phase sans chauffage
  au début (pour connaître l'état thermique initial) et la mesure de la température extérieure.
