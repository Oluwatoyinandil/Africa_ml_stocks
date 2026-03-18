# src/ml_model.py
# Objectif : entraîner un modèle ML pour prédire la surperformance

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import pickle
import os

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

DATASET_PATH = "data/dataset.csv"
RESULTS_PATH = "results/"

# Les colonnes qu'on donne au modèle
FEATURES = [
    "trailingPE",
    "priceToBook",
    "debtToEquity",
    "returnOnEquity",
    "returnOnAssets",
    "profitMargins",
    "currentRatio",
    "floatShares",
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "volatility",
    "rel_strength",
]

# ─────────────────────────────────────────────
# 1. Chargement et séparation des données
# ─────────────────────────────────────────────

def load_and_split(path: str):
    """
    Charge le dataset et le sépare en deux parties :
    - Train : données avant 2023 (pour apprendre)
    - Test  : données de 2023   (pour évaluer)

    On sépare par DATE et non aléatoirement — c'est
    crucial en finance pour ne pas tricher !
    """
    df = pd.read_csv(path, parse_dates=["date"])

    train = df[df["date"] < "2022-01-01"]
    test  = df[(df["date"] >= "2022-01-01") & (df["date"] < "2023-01-01")]

    X_train = train[FEATURES]
    y_train = train["label"]

    X_test  = test[FEATURES]
    y_test  = test["label"]

    print(f"  Données d'entraînement : {len(train)} lignes")
    print(f"  Données de test        : {len(test)} lignes")
    print(f"  Taux surperformance train : {y_train.mean():.1%}")
    print(f"  Taux surperformance test  : {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 2. Entraînement des modèles
# ─────────────────────────────────────────────

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Entraîne un Random Forest.
    C'est notre modèle principal — il combine plusieurs
    arbres de décision pour faire une prédiction robuste.
    """
    print("\n  Entraînement du Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,   # 100 arbres de décision
        max_depth=5,        # Profondeur max de chaque arbre
        random_state=42,    # Pour reproduire les mêmes résultats
        class_weight="balanced"  # Compense le déséquilibre 66/33
    )
    model.fit(X_train, y_train)
    print("  Random Forest entraîné ✓")
    return model


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Entraîne une Régression Logistique.
    C'est un modèle plus simple qu'on utilise comme
    point de comparaison avec le Random Forest.
    """
    print("\n  Entraînement de la Régression Logistique...")

    # La régression logistique a besoin que les données
    # soient sur la même échelle — on normalise
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        random_state=42,
        class_weight="balanced",
        max_iter=1000
    )
    model.fit(X_train_scaled, y_train)
    print("  Régression Logistique entraînée ✓")
    return model, scaler


# ─────────────────────────────────────────────
# 3. Évaluation des modèles
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, name: str, scaler=None):
    """
    Évalue le modèle sur les données de test et affiche
    les résultats de façon lisible.
    """
    print(f"\n{'='*50}")
    print(f"  Résultats : {name}")
    print(f"{'='*50}")

    # Normalisation si nécessaire
    X = scaler.transform(X_test) if scaler else X_test

    # Prédictions
    y_prob = model.predict_proba(X)[:, 1]
    # Seuil plus sélectif : on prédit 1 seulement si on est très confiant
    threshold = 0.60
    y_pred = (y_prob >= threshold).astype(int)

    # Métriques
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n  AUC Score : {auc:.3f}")
    print(f"\n  Rapport de classification :")
    print(classification_report(y_test, y_pred,
          target_names=["Sous-performe", "Surperforme"]))

    print(f"  Matrice de confusion :")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {cm}")

    return auc, y_pred, y_prob


# ─────────────────────────────────────────────
# 4. Importance des features
# ─────────────────────────────────────────────

def show_feature_importance(model: RandomForestClassifier):
    """
    Affiche quelles features sont les plus utiles
    pour le Random Forest.
    """
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES
    ).sort_values(ascending=False)

    print("\n  Importance des features (Random Forest) :")
    for feat, score in importance.items():
        bar = "█" * int(score * 100)
        print(f"  {feat:20s} {score:.3f}  {bar}")

    return importance


# ─────────────────────────────────────────────
# 5. Sauvegarde du modèle
# ─────────────────────────────────────────────

def save_model(model, filename: str):
    """
    Sauvegarde le modèle entraîné dans results/
    pour pouvoir le réutiliser sans réentraîner.
    """
    os.makedirs(RESULTS_PATH, exist_ok=True)
    path = os.path.join(RESULTS_PATH, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  Modèle sauvegardé : {path}")


# ─────────────────────────────────────────────
# Programme principal
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Entraînement du modèle ML — Version améliorée ===\n")

    X_train, X_test, y_train, y_test = load_and_split(DATASET_PATH)

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_auc, rf_pred, rf_prob = evaluate_model(
        rf_model, X_test, y_test, "Random Forest"
    )
    show_feature_importance(rf_model)
    save_model(rf_model, "random_forest.pkl")

    # Régression Logistique
    lr_model, scaler = train_logistic_regression(X_train, y_train)
    lr_auc, lr_pred, lr_prob = evaluate_model(
        lr_model, X_test, y_test,
        "Régression Logistique", scaler
    )
    save_model((lr_model, scaler), "logistic_regression.pkl")

    # Gradient Boosting — nouveau modèle
    from sklearn.ensemble import GradientBoostingClassifier
    print("\n  Entraînement du Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    print("  Gradient Boosting entraîné ✓")
    gb_auc, gb_pred, gb_prob = evaluate_model(
        gb_model, X_test, y_test, "Gradient Boosting"
    )
    save_model(gb_model, "gradient_boosting.pkl")

    # Comparaison finale
    print(f"\n{'='*50}")
    print(f"  Comparaison finale")
    print(f"{'='*50}")
    print(f"  Random Forest         AUC : {rf_auc:.3f}")
    print(f"  Régression Logistique AUC : {lr_auc:.3f}")
    print(f"  Gradient Boosting     AUC : {gb_auc:.3f}")

    best_auc = max(rf_auc, lr_auc, gb_auc)
    if best_auc == rf_auc:
        winner = "Random Forest"
    elif best_auc == lr_auc:
        winner = "Régression Logistique"
    else:
        winner = "Gradient Boosting"

    print(f"\n  Meilleur modèle : {winner} 🏆")
    print(f"  AUC             : {best_auc:.3f}")