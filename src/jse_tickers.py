# src/jse_tickers.py
# Objectif : construire et sauvegarder la liste des tickers JSE

import pandas as pd
import yfinance as yf
import os

# ─────────────────────────────────────────────
# Liste des principales entreprises de la JSE
# Format Yahoo Finance : NOM.JO
# ─────────────────────────────────────────────

JSE_TICKERS = [
    # Technologie & Médias
    ("NPN.JO",  "Naspers"),
    ("PRX.JO",  "Prosus"),

    # Ressources & Mines
    ("AGL.JO",  "Anglo American"),
    ("AMS.JO",  "Anglo American Platinum"),
    ("SOL.JO",  "Sasol"),
    ("GFI.JO",  "Gold Fields"),
    ("ANG.JO",  "AngloGold Ashanti"),
    ("IMP.JO",  "Impala Platinum"),
    ("FSR.JO",  "FirstRand"),

    # Finance & Banques
    ("SBK.JO",  "Standard Bank"),
    ("ABG.JO",  "Absa Group"),
    ("NED.JO",  "Nedbank"),
    ("DSY.JO",  "Discovery"),
    ("SLM.JO",  "Sanlam"),

    # Distribution & Consommation
    ("WHL.JO",  "Woolworths"),
    ("PIK.JO",  "Pick n Pay"),
    ("SPP.JO",  "Shoprite"),
    ("TFG.JO",  "The Foschini Group"),
    ("MRP.JO",  "Mr Price"),

    # Industrie & Énergie
    ("MTN.JO",  "MTN Group"),
    ("VOD.JO",  "Vodacom"),
    ("TKG.JO",  "Telkom"),
    ("REM.JO",  "Remgro"),
]

def build_tickers_df(tickers: list) -> pd.DataFrame:
    """
    Convertit la liste de tuples (ticker, nom) en DataFrame pandas.
    """
    df = pd.DataFrame(tickers, columns=["ticker", "name"])
    return df


def validate_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vérifie que chaque ticker est bien reconnu par Yahoo Finance.
    Garde uniquement ceux qui retournent des données.
    """
    valid = []

    for _, row in df.iterrows():
        ticker = row["ticker"]
        try:
            data = yf.Ticker(ticker).history(period="5d")
            if not data.empty:
                valid.append(row)
                print(f"  ✓  {ticker:12s}  {row['name']}")
            else:
                print(f"  ✗  {ticker:12s}  {row['name']}  — aucune donnée")
        except Exception as e:
            print(f"  ✗  {ticker:12s}  Erreur : {e}")

    return pd.DataFrame(valid).reset_index(drop=True)


def save_tickers(df: pd.DataFrame, path: str = "data/jse_tickers.csv"):
    """
    Sauvegarde le DataFrame dans un fichier CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n  Fichier sauvegardé : {path} ({len(df)} tickers valides)")


if __name__ == "__main__":
    print("=== Construction de la liste des tickers JSE ===\n")

    df = build_tickers_df(JSE_TICKERS)

    print("Validation des tickers sur Yahoo Finance...\n")
    df_valid = validate_tickers(df)

    save_tickers(df_valid)

    print("\nAperçu du fichier final :")
    print(df_valid)