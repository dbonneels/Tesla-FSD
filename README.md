# Tesla-FSD

Analyse statistique et dashboard de tarification assurance auto autour du Tesla Full Self-Driving (FSD), avec données NHTSA (USA).

## Projet

**AssuranceAuto.be** — comparateur d'assurances auto belge (FR / EN / NL / DE) avec un design identitaire basé sur la plaque d'immatriculation belge et les marquages routiers jaunes.

Ce dépôt contient :
- **`01_eda.ipynb`** — Exploration descriptive des données NHTSA (SGO/ADAS + FARS 2024).
- **`02_modeling.ipynb`** — Régression logistique (risque d'accident mortel, Tesla vs autres véhicules) et export `dashboard_data.json`.
- **`src/tesla_fsd/pipeline.py`** — Pipeline Python réutilisable (point d'entrée `tesla-fsd`).
- **`dashboard.html`** — Dashboard de tarification pour assurances (KPI, simulateur de prime, alerte transfert USA→Europe).
- **`assuranceauto.html` / `tesla-fsd.html`** — Pages web multilingues (FR/EN/NL/DE).

## Données

Les données proviennent de la **NHTSA (USA)** :
- `data/raw/nhtsa_sgo/level2_ADAS.csv` — rapports d'incidents ADAS
- `data/raw/nhtsa_sgo/other_incidents.csv` — autres incidents
- `data/raw/fars/FARS2024NationalCSV/` — accidents mortels 2024 (accident, vehicle, person)

⚠️ Le dossier `data/` (≈1,1 Go) est **exclu du dépôt** via `.gitignore`.

## Hypothèse de transférabilité USA → Europe

Le modèle est calibré sur des données américaines. Le contexte européen (infrastructure, densité, climat, comportement conducteur) étant différent, **le facteur de risque mesuré aux USA n'est pas directement transférable en Europe sans recalibration sur des données locales.**

Aucune conclusion tarifaire Europe ne doit être tirée directement de ces résultats sans données européennes (déclarations d'assureurs, télématique, etc.).

## Installation

```bash
cd ~/Tesla-FSD
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Dépendances principales : `pandas`, `numpy`, `statsmodels`, `jupyter`, `nbformat`, `matplotlib`, `seaborn`.

## Utilisation

### Notebooks

Lancer avec Jupyter :

```bash
jupyter notebook 01_eda.ipynb
jupyter notebook 02_modeling.ipynb
```

### Pipeline Python

```bash
cd ~/Tesla-FSD
python -m src.tesla_fsd.pipeline
```

Ce script recharge les données, entraîne le modèle et exporte `dashboard_data.json` (lu par `dashboard.html`).

## Structure du dépôt

```
Tesla-FSD/
├── 01_eda.ipynb              # Exploration descriptive (EDA)
├── 02_modeling.ipynb         # Modélisation + export JSON
├── README.md
├── pyproject.toml
├── .python-version
├── .gitignore
├── dashboard_data.json       # Résultat du modèle (pour le dashboard)
├── src/
│   └── tesla_fsd/
│       ├── __init__.py
│       └── pipeline.py       # Pipeline Python réutilisable
└── data/                     # Données brutes NHTSA (exclu de Git via .gitignore)
    ├── nhtsa_sgo/
    └── fars/
```

## Dashboard Web

Les fichiers HTML se trouvent dans le dossier Windows :
`C:\Users\dbonn\OneDrive - Sun Power Energy srl\Trabajo\Scripts\assuranceauto.be v2\`

- `assuranceauto.html` — Page d'accueil (comparateur)
- `tesla-fsd.html` — Page d'analyse FSD
- `dashboard.html` — Dashboard de tarification (lit `dashboard_data.json`)

## Avertissement

Ce projet est une démonstration d'analyse et de conception. Les facteurs de tarification affichés sont des estimations statistiques basées sur des données USA, pas des offres contractuelles. Ne pas utiliser pour une tarification Europe sans recalibration sur des données locales.

## Licence

Voir `LICENSE.txt` à la racine du projet (si présente).
