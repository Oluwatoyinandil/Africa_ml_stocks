# src/optimisation.py
# Objectif : calculer l'allocation optimale du portefeuille
# avec PyPortfolioOpt (théorie de Markowitz)

import pandas as pd
import numpy as np
import pickle
import os
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

DATASET_PATH = "data/dataset.csv"
PRICES_PATH  = "data/prices.csv"
MODEL_PATH   = "results/gradient_boosting.pkl"
RESULTS_PATH = "results/"

FEATURES = [
    "trailingPE", "priceToBook", "debtToEquity",
    "returnOnEquity", "returnOnAssets", "profitMargins",
    "currentRatio", "floatShares", "momentum_1m",
    "momentum_3m", "momentum_6m", "volatility", "rel_strength",
]

CAPITAL = 100_000  # ZAR
THRESHOLD = 0.55   # Seuil de confiance ML


# ─────────────────────────────────────────────
# 1. Sélection des actions via le modèle ML
# ─────────────────────────────────────────────

def select_stocks(model, dataset: pd.DataFrame) -> list:
    """
    Utilise le modèle pour sélectionner les actions
    les plus prometteuses sur la dernière date disponible.
    C'est la liste qu'on va optimiser.
    """
    # Dernière date disponible dans le dataset
    last_date = dataset["date"].max()
    snapshot  = dataset[dataset["date"] == last_date].copy()

    # Probabilité de surperformance
    snapshot["proba"] = model.predict_proba(
        snapshot[FEATURES]
    )[:, 1]

    # On garde les actions au dessus du seuil
    selected = snapshot[snapshot["proba"] >= THRESHOLD].copy()
    selected = selected.sort_values("proba", ascending=False)

    print(f"  Date d'analyse    : {last_date.date()}")
    print(f"  Actions analysées : {len(snapshot)}")
    print(f"  Actions retenues  : {len(selected)}")
    print(f"\n  Actions sélectionnées par le modèle :")
    print(f"  {'Ticker':<12} {'Probabilité':>12}")
    print(f"  {'─'*25}")
    for _, row in selected.iterrows():
        print(f"  {row['ticker']:<12} {row['proba']:>11.1%}")

    return selected["ticker"].tolist()


# ─────────────────────────────────────────────
# 2. Optimisation du portefeuille
# ─────────────────────────────────────────────

def optimise_portfolio(tickers: list,
                        prices: pd.DataFrame) -> dict:
    """
    Applique la théorie de Markowitz pour trouver
    l'allocation optimale entre les actions sélectionnées.

    On cherche le portefeuille qui maximise le Sharpe ratio —
    c'est à dire le meilleur rendement possible pour
    le niveau de risque pris.
    """
    # On garde uniquement les prix des actions sélectionnées
    prices_selected = prices[tickers].dropna()

    print(f"\n  Période d'analyse : {prices_selected.index[0].date()}"
          f" → {prices_selected.index[-1].date()}")
    print(f"  Jours de trading  : {len(prices_selected)}")

    # Calcul des rendements espérés (moyenne historique)
    mu = expected_returns.mean_historical_return(
        prices_selected, frequency=252
    )

    # Calcul de la matrice de covariance (risque et corrélations)
    S = risk_models.sample_cov(prices_selected, frequency=252)

    # Optimisation : maximiser le Sharpe ratio
    ef = EfficientFrontier(mu, S)

    # Contraintes : chaque action entre 5% et 40% du portefeuille
    ef.add_constraint(lambda w: w >= 0.05)
    ef.add_constraint(lambda w: w <= 0.40)

    ef.max_sharpe(risk_free_rate=0.08)  # Taux sans risque JSE ~8%
    weights = ef.clean_weights()

    # Métriques du portefeuille optimisé
    perf = ef.portfolio_performance(
        verbose=False, risk_free_rate=0.08
    )

    print(f"\n  Performance du portefeuille optimisé :")
    print(f"  Rendement espéré  : {perf[0]:.1%}")
    print(f"  Volatilité        : {perf[1]:.1%}")
    print(f"  Sharpe ratio      : {perf[2]:.3f}")

    return weights, perf


# ─────────────────────────────────────────────
# 3. Allocation concrète en actions
# ─────────────────────────────────────────────

def allocate_capital(weights: dict,
                      prices: pd.DataFrame,
                      capital: float) -> pd.DataFrame:
    """
    Traduit les pourcentages en nombre concret d'actions
    à acheter avec notre capital disponible.
    """
    # Prix les plus récents
    latest_prices = get_latest_prices(prices[list(weights.keys())])

    # Allocation discrète (nombre entier d'actions)
    da = DiscreteAllocation(
        weights, latest_prices, total_portfolio_value=capital
    )
    allocation, leftover = da.greedy_portfolio()

    # Construction du tableau récapitulatif
    rows = []
    for ticker, nb_actions in allocation.items():
        prix      = latest_prices[ticker]
        montant   = nb_actions * prix
        poids     = weights.get(ticker, 0)
        rows.append({
            "ticker"    : ticker,
            "poids_%"   : round(poids * 100, 1),
            "nb_actions": nb_actions,
            "prix_ZAR"  : round(prix, 2),
            "montant_ZAR": round(montant, 2),
        })

    df = pd.DataFrame(rows).sort_values(
        "poids_%", ascending=False
    ).reset_index(drop=True)

    print(f"\n  Allocation du capital ({capital:,.0f} ZAR) :")
    print(f"\n  {'Ticker':<12} {'Poids':>8} {'Actions':>10}"
          f" {'Prix ZAR':>12} {'Montant ZAR':>14}")
    print(f"  {'─'*60}")
    for _, row in df.iterrows():
        print(f"  {row['ticker']:<12} {row['poids_%']:>7.1f}%"
              f" {row['nb_actions']:>10}"
              f" {row['prix_ZAR']:>12,.2f}"
              f" {row['montant_ZAR']:>14,.2f}")

    print(f"\n  Capital investi   : {df['montant_ZAR'].sum():>12,.2f} ZAR")
    print(f"  Capital restant   : {leftover:>12,.2f} ZAR")

    return df


# ─────────────────────────────────────────────
# Programme principal
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Optimisation du portefeuille JSE ===\n")

    # Chargement
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    dataset = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    prices  = pd.read_csv(PRICES_PATH,
                           index_col=0, parse_dates=True)

    # Sélection ML
    print("─── Étape 1 : Sélection ML ───")
    selected_tickers = select_stocks(model, dataset)

    if len(selected_tickers) < 2:
        print("\n  Pas assez d'actions sélectionnées (minimum 2).")
        print("  Essaie de baisser le THRESHOLD.")
    else:
        # Optimisation Markowitz
        print("\n─── Étape 2 : Optimisation Markowitz ───")
        weights, perf = optimise_portfolio(selected_tickers, prices)

        # Allocation concrète
        print("\n─── Étape 3 : Allocation du capital ───")
        allocation_df = allocate_capital(weights, prices, CAPITAL)

        # Sauvegarde
        os.makedirs(RESULTS_PATH, exist_ok=True)
        allocation_df.to_csv(
            f"{RESULTS_PATH}portfolio_allocation.csv", index=False
        )
        print(f"\n  Fichier sauvegardé : {RESULTS_PATH}portfolio_allocation.csv")

        print(f"\n{'='*50}")
        print(f"  Pipeline complet terminé ✓")
        print(f"  ML → sélection → Markowitz → allocation")
        print(f"{'='*50}")