# src/data_collection.py
# Objectif : télécharger prix historiques et fondamentaux des 23 tickers JSE

import pandas as pd
import yfinance as yf
import os
import time

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

TICKERS_PATH  = "data/jse_tickers.csv"
PRICES_PATH   = "data/prices.csv"
FUNDAMENTALS_PATH = "data/fundamentals.csv"

# Période historique : 5 ans de données
START_DATE = "2019-01-01"
END_DATE   = "2024-01-01"

# Fondamentaux qu'on veut récupérer
FUNDAMENTALS_KEYS = [
    "trailingPE",        # P/E ratio
    "priceToBook",       # P/B ratio
    "debtToEquity",      # D/E ratio
    "returnOnEquity",    # ROE
    "returnOnAssets",    # ROA
    "earningsPerShare",  # EPS
    "profitMargins",     # Marge nette
    "currentRatio",      # Liquidité
    "floatShares",       # Float
]


# ─────────────────────────────────────────────
# 1. Téléchargement des prix historiques
# ─────────────────────────────────────────────

def download_prices(tickers: list) -> pd.DataFrame:
    """
    Télécharge les prix de clôture ajustés pour tous les tickers.
    Retourne un DataFrame avec une colonne par ticker.
    """
    print("Téléchargement des prix historiques...\n")

    prices = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=True
    )["Close"]

    # Si un seul ticker, yf retourne une Series — on la convertit
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    print(f"\n  {prices.shape[1]} tickers  |  {prices.shape[0]} jours de trading")
    return prices


# ─────────────────────────────────────────────
# 2. Téléchargement des fondamentaux
# ─────────────────────────────────────────────

def download_fundamentals(tickers: list) -> pd.DataFrame:
    """
    Récupère les ratios fondamentaux pour chaque ticker.
    Retourne un DataFrame avec une ligne par ticker.
    """
    print("\nTéléchargement des fondamentaux...\n")

    rows = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            row = {"ticker": ticker}

            for key in FUNDAMENTALS_KEYS:
                row[key] = info.get(key, None)

            rows.append(row)
            print(f"  ✓  {ticker:12s}  PE={row.get('trailingPE', 'N/A')}")

        except Exception as e:
            print(f"  ✗  {ticker:12s}  Erreur : {e}")

        # Pause pour éviter de surcharger l'API Yahoo Finance
        time.sleep(0.5)

    df = pd.DataFrame(rows).set_index("ticker")
    print(f"\n  {len(df)} entreprises  |  {len(FUNDAMENTALS_KEYS)} ratios")
    return df


# ─────────────────────────────────────────────
# 3. Sauvegarde
# ─────────────────────────────────────────────

def save_data(prices: pd.DataFrame, fundamentals: pd.DataFrame):
    """
    Sauvegarde les deux DataFrames en CSV dans le dossier data/.
    """
    os.makedirs("data", exist_ok=True)

    prices.to_csv(PRICES_PATH)
    fundamentals.to_csv(FUNDAMENTALS_PATH)

    print(f"\n  Prix sauvegardés        → {PRICES_PATH}")
    print(f"  Fondamentaux sauvegardés → {FUNDAMENTALS_PATH}")


# ─────────────────────────────────────────────
# Programme principal
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Collecte des données JSE ===\n")

    # Chargement de la liste des tickers
    tickers_df = pd.read_csv(TICKERS_PATH)
    tickers = tickers_df["ticker"].tolist()
    print(f"  {len(tickers)} tickers chargés depuis {TICKERS_PATH}\n")

    # Téléchargement
    prices       = download_prices(tickers)
    fundamentals = download_fundamentals(tickers)

    # Sauvegarde
    save_data(prices, fundamentals)

    print("\n=== Collecte terminée avec succès ===")
    print("\nAperçu des prix (5 dernières lignes) :")
    print(prices.tail())
    print("\nAperçu des fondamentaux :")
    print(fundamentals.head())