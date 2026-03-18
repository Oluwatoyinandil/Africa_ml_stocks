# 📈 Africa ML Stocks — JSE

Projet de Machine Learning appliqué à la **Bourse de Johannesburg (JSE)** 
pour prédire la surperformance des actions et construire un portefeuille optimal.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Objectif

Construire un pipeline complet de A à Z qui :

1. Collecte les données de 22 actions JSE via Yahoo Finance
2. Prédit quelles actions vont surperformer l'indice JSE All Share
3. Valide la stratégie sur des données historiques (backtesting)
4. Optimise l'allocation du capital avec la théorie de Markowitz

---

## 🏗️ Architecture du projet
```
Africa_ml_stocks/
│
├── data/                         # Données brutes et traitées
│   ├── jse_tickers.csv           # Liste des 22 actions JSE
│   ├── prices.csv                # Prix historiques 2019-2024
│   ├── fundamentals.csv          # Ratios financiers
│   └── dataset.csv               # Dataset ML final
│
├── notebooks/                    # Exploration et visualisation
│   ├── 01_exploration.ipynb      # Analyse des données JSE
│   ├── 02_preprocessing.ipynb   # Création du label
│   ├── 03_ml_model.ipynb         # Évaluation des modèles
│   ├── 04_backtesting.ipynb      # Simulation de la stratégie
│   └── 05_optimisation.ipynb    # Portefeuille optimal
│
├── src/                          # Scripts Python
│   ├── jse_tickers.py            # Collecte des tickers
│   ├── data_collection.py        # Téléchargement des données
│   ├── preprocessing.py          # Nettoyage + features
│   ├── ml_model.py               # Entraînement des modèles
│   ├── backtesting.py            # Simulation historique
│   ├── optimisation.py           # Optimisation Markowitz
│   └── dashboard.py              # Dashboard Streamlit
│
└── results/
    ├── figures/                  # Graphiques exportés
    ├── backtesting_results.csv   # Résultats du backtesting
    └── portfolio_allocation.csv  # Allocation finale
```

---

## 🔄 Pipeline
```
Données JSE (yfinance)
        ↓
Preprocessing + features dynamiques
(momentum, volatilité, force relative)
        ↓
Modèle ML — Gradient Boosting
        ↓
Backtesting 2022
        ↓
Optimisation Markowitz
        ↓
Portefeuille optimal
```

---

## 📊 Résultats

| Métrique | Valeur |
|---|---|
| Actions analysées | 22 |
| Période | 2019 — 2024 |
| Features | 13 (8 fondamentaux + 5 dynamiques) |
| Meilleur modèle | Gradient Boosting |
| AUC Score | 0.462 |
| Rendement ML 2022 | +6.5% |
| Rendement Buy & Hold 2022 | +1.3% |
| Avantage ML | +5.2% |
| Sharpe ratio portefeuille | 0.443 |

---

## 💼 Portefeuille final

| Ticker | Entreprise | Poids |
|---|---|---|
| GFI.JO | Gold Fields | 40.0% |
| IMP.JO | Impala Platinum | 27.4% |
| REM.JO | Remgro | 13.8% |
| SBK.JO | Standard Bank | 5.0% |
| FSR.JO | FirstRand | 5.0% |

---

## ⚙️ Installation

### 1. Cloner le projet
```bash
git clone https://github.com/ton-username/Africa_ml_stocks.git
cd Africa_ml_stocks
```

### 2. Créer l'environnement virtuel
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## 🚀 Utilisation

### Reproduire le pipeline complet
```bash
python src/jse_tickers.py
python src/data_collection.py
python src/preprocessing.py
python src/ml_model.py
python src/backtesting.py
python src/optimisation.py
```

### Lancer le dashboard interactif
```bash
streamlit run src/dashboard.py
```

### Explorer les notebooks
```bash
jupyter notebook
```

---

## ⚠️ Limites et pistes d'amélioration

**Limites actuelles :**
- Fondamentaux statiques — un seul snapshot pour toute la période
- Univers restreint — 22 actions seulement
- Biais de survie — uniquement les entreprises encore cotées en 2024

**Améliorations possibles :**
- Données fondamentales trimestrielles historiques
- Élargir l'univers à toutes les actions JSE (~350 titres)
- Ajouter des données macro (taux ZAR, prix matières premières)
- Modèles LSTM pour capturer les séquences temporelles
- Adapter le pipeline à la BRVM (Afrique de l'Ouest)

---

## 🛠️ Technologies

- **Python 3.11**
- **yfinance** — collecte des données boursières
- **pandas / numpy** — manipulation des données
- **scikit-learn** — modèles de Machine Learning
- **PyPortfolioOpt** — optimisation de portefeuille
- **Plotly / Streamlit** — visualisation interactive

---

## 👤 Auteur

**Ton Nom**  
[LinkedIn](www.linkedin.com/in/abdul-andil-toyin-afodji-aa2956226) · 
[GitHub](https://github.com/Oluwatoyinandil)

---

## 📄 Licence

MIT License — libre d'utilisation avec attribution.