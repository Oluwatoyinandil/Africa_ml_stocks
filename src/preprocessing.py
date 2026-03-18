# src/preprocessing.py — 
# Nouvelles features : momentum, volatilité, performance relative

import pandas as pd
import numpy as np
import yfinance as yf
import os

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

PRICES_PATH       = "data/prices.csv"
FUNDAMENTALS_PATH = "data/fundamentals.csv"
TICKERS_PATH      = "data/jse_tickers.csv"
OUTPUT_PATH       = "data/dataset.csv"
BENCHMARK_TICKER  = "^J203.JO"
HOLDING_PERIOD    = 252


# ─────────────────────────────────────────────
# 1. Chargement des données
# ─────────────────────────────────────────────

def load_data():
    prices       = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    fundamentals = pd.read_csv(FUNDAMENTALS_PATH, index_col=0)
    tickers      = pd.read_csv(TICKERS_PATH)

    print(f"  Prix chargés       : {prices.shape}")
    print(f"  Fondamentaux       : {fundamentals.shape}")
    print(f"  Tickers            : {len(tickers)}")

    return prices, fundamentals, tickers


# ─────────────────────────────────────────────
# 2. Benchmark
# ─────────────────────────────────────────────

def load_benchmark(start: str, end: str) -> pd.Series:
    import yfinance as yf
    print(f"\n  Téléchargement du benchmark {BENCHMARK_TICKER}...")
    benchmark = yf.download(
        BENCHMARK_TICKER,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )["Close"].squeeze()
    benchmark.name = "benchmark"
    print(f"  Benchmark chargé   : {len(benchmark)} jours")
    return benchmark


# ─────────────────────────────────────────────
# 3. Nouvelles features dynamiques (basées sur les prix)
# ─────────────────────────────────────────────

def compute_price_features(prices: pd.DataFrame,
                            benchmark: pd.Series) -> pd.DataFrame:
    """
    Calcule pour chaque ticker et chaque date :
    - momentum_1m  : performance du dernier mois (21 jours)
    - momentum_3m  : performance des 3 derniers mois (63 jours)
    - momentum_6m  : performance des 6 derniers mois (126 jours)
    - volatility   : volatilité sur 63 jours (écart-type des rendements)
    - rel_strength : performance relative vs le benchmark sur 63 jours
    """
    print("\n  Calcul des features dynamiques...")

    all_features = []

    for ticker in prices.columns:
        serie = prices[ticker].dropna()

        # Rendements journaliers
        daily_returns = serie.pct_change()

        # Momentum
        mom_1m = serie.pct_change(periods=21)
        mom_3m = serie.pct_change(periods=63)
        mom_6m = serie.pct_change(periods=126)

        # Volatilité (écart-type sur 63 jours glissants)
        volatility = daily_returns.rolling(window=63).std()

        # Force relative vs benchmark
        bench_aligned = benchmark.reindex(serie.index)
        rel_strength  = mom_3m - bench_aligned.pct_change(periods=63)

        # On rassemble tout dans un DataFrame
        df_ticker = pd.DataFrame({
            "ticker"      : ticker,
            "momentum_1m" : mom_1m,
            "momentum_3m" : mom_3m,
            "momentum_6m" : mom_6m,
            "volatility"  : volatility,
            "rel_strength": rel_strength,
        })

        all_features.append(df_ticker)

    result = pd.concat(all_features)
    print(f"  Features dynamiques calculées ✓")
    return result


# ─────────────────────────────────────────────
# 4. Calcul des rendements et labels
# ─────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame,
                    benchmark: pd.Series,
                    holding_period: int = 252) -> pd.DataFrame:
    returns = prices.pct_change(
        periods=holding_period).shift(-holding_period)
    bench_returns = benchmark.pct_change(
        periods=holding_period).shift(-holding_period)
    bench_returns.name = "benchmark_return"
    returns = returns.join(bench_returns)
    return returns


def create_labels(returns: pd.DataFrame) -> pd.DataFrame:
    labels = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        if col == "benchmark_return":
            continue
        labels[col] = (
            returns[col] > returns["benchmark_return"]
        ).astype(int)
    return labels


# ─────────────────────────────────────────────
# 5. Nettoyage des fondamentaux
# ─────────────────────────────────────────────

def clean_fundamentals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    df = fundamentals.drop(columns=["earningsPerShare"], errors="ignore")
    for col in df.columns:
        lower = df[col].quantile(0.05)
        upper = df[col].quantile(0.95)
        df[col] = df[col].clip(lower=lower, upper=upper)
    df = df.fillna(df.median())
    print(f"\n  Fondamentaux nettoyés : {df.shape}")
    return df


# ─────────────────────────────────────────────
# 6. Construction du dataset final
# ─────────────────────────────────────────────

def build_dataset(labels: pd.DataFrame,
                  fundamentals: pd.DataFrame,
                  price_features: pd.DataFrame) -> pd.DataFrame:
    """
    Combine labels + fondamentaux + features dynamiques
    en un seul dataset propre.
    """
    rows = []

    # On remet l'index date sur price_features
    price_features = price_features.reset_index().rename(
        columns={"index": "date"})

    for ticker in labels.columns:
        if ticker not in fundamentals.index:
            continue

        ticker_labels   = labels[ticker].dropna()
        features_static = fundamentals.loc[ticker].to_dict()

        # Features dynamiques pour ce ticker
        pf_ticker = price_features[
            price_features["ticker"] == ticker
        ].set_index("Date")

        for date, label in ticker_labels.items():
            row = {"ticker": ticker, "date": date, "label": int(label)}

            # Fondamentaux statiques
            row.update(features_static)

            # Features dynamiques si disponibles à cette date
            if date in pf_ticker.index:
                for col in ["momentum_1m", "momentum_3m",
                            "momentum_6m", "volatility", "rel_strength"]:
                    row[col] = pf_ticker.loc[date, col]
            else:
                for col in ["momentum_1m", "momentum_3m",
                            "momentum_6m", "volatility", "rel_strength"]:
                    row[col] = np.nan

            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.dropna()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# Programme principal
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Preprocessing amélioré ===\n")

    prices, fundamentals, tickers = load_data()

    start = str(prices.index.min().date())
    end   = str(prices.index.max().date())
    benchmark = load_benchmark(start, end)

    # Nouvelles features dynamiques
    price_features = compute_price_features(prices, benchmark)

    # Labels
    print("\n  Calcul des labels...")
    returns = compute_returns(prices, benchmark)
    labels  = create_labels(returns)

    # Fondamentaux
    fundamentals_clean = clean_fundamentals(fundamentals)

    # Dataset final
    print("\n  Construction du dataset final...")
    dataset = build_dataset(labels, fundamentals_clean, price_features)

    os.makedirs("data", exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"\n  Dataset sauvegardé  : {OUTPUT_PATH}")
    print(f"  Shape               : {dataset.shape}")
    print(f"\n  Colonnes disponibles :")
    print(f"  {list(dataset.columns)}")
    print(f"\n  Distribution du label :")
    print(dataset["label"].value_counts())
    print(f"\n  Taux de surperformance : {dataset['label'].mean():.1%}")
    print("\nAperçu :")
    print(dataset.head())