# src/backtesting.py
# Objectif : simuler une stratégie d'investissement basée sur les prédictions ML

import pandas as pd
import numpy as np
import pickle
import os

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

DATASET_PATH  = "data/dataset.csv"
PRICES_PATH   = "data/prices.csv"
MODEL_PATH    = "results/gradient_boosting.pkl"
RESULTS_PATH  = "results/"

FEATURES = [
    "trailingPE", "priceToBook", "debtToEquity",
    "returnOnEquity", "returnOnAssets", "profitMargins",
    "currentRatio", "floatShares", "momentum_1m",
    "momentum_3m", "momentum_6m", "volatility", "rel_strength",
]

# Capital de départ simulé
CAPITAL_INITIAL = 100_000  # 100 000 ZAR

# Seuil de confiance pour investir
THRESHOLD = 0.55


# ─────────────────────────────────────────────
# 1. Chargement du modèle et des données
# ─────────────────────────────────────────────

def load_model_and_data():
    """
    Charge le modèle entraîné et le dataset.
    """
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    dataset = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    prices  = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)

    print(f"  Modèle chargé    : {MODEL_PATH}")
    print(f"  Dataset          : {dataset.shape}")
    print(f"  Prix             : {prices.shape}")

    return model, dataset, prices


# ─────────────────────────────────────────────
# 2. Génération des signaux d'achat
# ─────────────────────────────────────────────

def generate_signals(model, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Utilise le modèle pour générer des signaux d'achat.
    On prédit la probabilité de surperformance pour chaque
    ticker à chaque date — on garde seulement les plus confiants.
    """
    # On prédit sur toute la période de test (2022-2023)
    test = dataset[
        (dataset["date"] >= "2022-01-01") &
        (dataset["date"] < "2023-01-01")
    ].copy()

    # Probabilité de surperformance
    test["proba"] = model.predict_proba(test[FEATURES])[:, 1]

    # Signal = 1 si le modèle est assez confiant
    test["signal"] = (test["proba"] >= THRESHOLD).astype(int)

    n_signals = test["signal"].sum()
    print(f"\n  Période de test  : 2022")
    print(f"  Signaux générés  : {n_signals} achats potentiels")
    print(f"  Tickers concernés: {test[test['signal']==1]['ticker'].nunique()}")

    return test


# ─────────────────────────────────────────────
# 3. Simulation de la stratégie
# ─────────────────────────────────────────────

def simulate_strategy(signals: pd.DataFrame,
                       prices: pd.DataFrame) -> pd.DataFrame:
    """
    Simule deux stratégies sur la période de test :

    Stratégie ML   : on investit uniquement dans les actions
                     que le modèle recommande (signal = 1)

    Stratégie Buy & Hold : on investit également dans toutes
                           les actions sans sélection — sert
                           de point de comparaison
    """
    results = []

    # On prend un snapshot par mois (premier jour ouvrable)
    signals["month"] = signals["date"].dt.to_period("M")
    monthly = signals.groupby("month").first().reset_index()

    for _, row in monthly.iterrows():
        date     = row["date"]
        month    = row["month"]

        # Actions sélectionnées par le modèle ce mois
        month_signals = signals[
            (signals["date"] == date) &
            (signals["signal"] == 1)
        ]

        # Toutes les actions disponibles ce mois
        month_all = signals[signals["date"] == date]

        # Calcul du rendement moyen sur les 21 jours suivants
        future_date_idx = prices.index.searchsorted(date)
        horizon = min(future_date_idx + 21, len(prices) - 1)
        future_date = prices.index[horizon]

        def avg_return(tickers):
            rets = []
            for t in tickers:
                if t in prices.columns:
                    p_start = prices.loc[date, t] if date in prices.index else np.nan
                    p_end   = prices.loc[future_date, t] if future_date in prices.index else np.nan
                    if pd.notna(p_start) and pd.notna(p_end) and p_start > 0:
                        rets.append((p_end - p_start) / p_start)
            return np.mean(rets) if rets else 0.0

        ml_return  = avg_return(month_signals["ticker"].tolist())
        bh_return  = avg_return(month_all["ticker"].tolist())

        results.append({
            "month"        : str(month),
            "date"         : date,
            "ml_return"    : ml_return,
            "bh_return"    : bh_return,
            "n_selected"   : len(month_signals),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 4. Calcul de la performance cumulée
# ─────────────────────────────────────────────

def compute_cumulative(results: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la valeur du portefeuille mois par mois
    en partant de 100 000 ZAR.
    """
    results = results.copy()

    results["ml_cumulative"] = CAPITAL_INITIAL * (
        1 + results["ml_return"]
    ).cumprod()

    results["bh_cumulative"] = CAPITAL_INITIAL * (
        1 + results["bh_return"]
    ).cumprod()

    return results


# ─────────────────────────────────────────────
# 5. Résumé de la performance
# ─────────────────────────────────────────────

def print_summary(results: pd.DataFrame):
    """
    Affiche un résumé clair de la performance des deux stratégies.
    """
    ml_final = results["ml_cumulative"].iloc[-1]
    bh_final = results["bh_cumulative"].iloc[-1]

    ml_total = (ml_final - CAPITAL_INITIAL) / CAPITAL_INITIAL
    bh_total = (bh_final - CAPITAL_INITIAL) / CAPITAL_INITIAL

    ml_monthly_avg = results["ml_return"].mean()
    bh_monthly_avg = results["bh_return"].mean()

    print(f"\n{'='*50}")
    print(f"  Résultats du Backtesting (2022)")
    print(f"{'='*50}")
    print(f"\n  Capital initial       : {CAPITAL_INITIAL:>12,.0f} ZAR")
    print(f"\n  Stratégie ML :")
    print(f"    Capital final       : {ml_final:>12,.0f} ZAR")
    print(f"    Rendement total     : {ml_total:>11.1%}")
    print(f"    Rendement mensuel   : {ml_monthly_avg:>11.1%}")
    print(f"\n  Stratégie Buy & Hold :")
    print(f"    Capital final       : {bh_final:>12,.0f} ZAR")
    print(f"    Rendement total     : {bh_total:>11.1%}")
    print(f"    Rendement mensuel   : {bh_monthly_avg:>11.1%}")

    print(f"\n  Avantage ML vs B&H   : {ml_total - bh_total:>+11.1%}")

    if ml_total > bh_total:
        print(f"\n  La stratégie ML bat le Buy & Hold sur 2022 ✓")
    else:
        print(f"\n  La stratégie ML ne bat pas le Buy & Hold sur 2022")
        print(f"  → Résultat honnête, cohérent avec l'AUC de 0.46")


# ─────────────────────────────────────────────
# Programme principal
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Backtesting de la stratégie ML ===\n")

    # Chargement
    model, dataset, prices = load_model_and_data()

    # Signaux
    signals = generate_signals(model, dataset)

    # Simulation
    print("\n  Simulation en cours...")
    results = simulate_strategy(signals, prices)
    results = compute_cumulative(results)

    # Sauvegarde
    os.makedirs(RESULTS_PATH, exist_ok=True)
    results.to_csv(f"{RESULTS_PATH}backtesting_results.csv", index=False)
    print(f"  Résultats sauvegardés : {RESULTS_PATH}backtesting_results.csv")

    # Résumé
    print_summary(results)

    print(f"\n  Détail mois par mois :")
    print(results[[
        "month", "n_selected", "ml_return", "bh_return"
    ]].to_string(index=False))