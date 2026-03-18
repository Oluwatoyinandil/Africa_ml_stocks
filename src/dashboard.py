# src/dashboard.py
# Dashboard interactif Streamlit pour le projet Africa ML Stocks

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

# ─────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Africa ML Stocks — JSE",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Chargement des données (mis en cache)
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    prices       = pd.read_csv("data/prices.csv",
                                index_col=0, parse_dates=True)
    fundamentals = pd.read_csv("data/fundamentals.csv",
                                index_col=0)
    dataset      = pd.read_csv("data/dataset.csv",
                                parse_dates=["date"])
    tickers      = pd.read_csv("data/jse_tickers.csv")
    results      = pd.read_csv("results/backtesting_results.csv",
                                parse_dates=["date"])
    allocation   = pd.read_csv("results/portfolio_allocation.csv")
    return prices, fundamentals, dataset, tickers, results, allocation


@st.cache_resource
def load_model():
    with open("results/gradient_boosting.pkl", "rb") as f:
        return pickle.load(f)


prices, fundamentals, dataset, tickers, bt_results, allocation = load_data()
model = load_model()

FEATURES = [
    "trailingPE", "priceToBook", "debtToEquity",
    "returnOnEquity", "returnOnAssets", "profitMargins",
    "currentRatio", "floatShares", "momentum_1m",
    "momentum_3m", "momentum_6m", "volatility", "rel_strength",
]

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/"
    "Flag_of_South_Africa.svg/320px-Flag_of_South_Africa.svg.png",
    width=100
)

st.sidebar.title("Africa ML Stocks")
st.sidebar.markdown("**Johannesburg Stock Exchange**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Accueil",
        "📊 Exploration des données",
        "🤖 Modèle ML",
        "📈 Backtesting",
        "💼 Portefeuille optimal",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Projet : Machine Learning appliqué "
    "aux actions de la JSE"
)

# ─────────────────────────────────────────────
# Page 1 : Accueil
# ─────────────────────────────────────────────

if page == "🏠 Accueil":
    st.title("📈 Africa ML Stocks — JSE")
    st.markdown(
        "Application de prédiction de surperformance boursière "
        "sur la **Bourse de Johannesburg (JSE)**"
    )

    st.markdown("---")

    # Métriques clés
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Actions analysées", "22")
    with col2:
        st.metric("Période", "2019 — 2024")
    with col3:
        ml_return  = (bt_results["ml_cumulative"].iloc[-1] - 100_000) / 100_000
        bh_return  = (bt_results["bh_cumulative"].iloc[-1] - 100_000) / 100_000
        st.metric(
            "Rendement ML 2022",
            f"{ml_return:.1%}",
            delta=f"{ml_return - bh_return:+.1%} vs B&H"
        )
    with col4:
        st.metric("Sharpe ratio", "0.443")

    st.markdown("---")

    # Pipeline
    st.subheader("Pipeline du projet")
    st.markdown("""
```
    Données JSE (yfinance)
           ↓
    Preprocessing + features dynamiques
           ↓
    Modèle ML (Gradient Boosting)
           ↓
    Backtesting (2022)
           ↓
    Optimisation Markowitz
```
    """)

    st.markdown("---")

    # Résumé des étapes
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Technologies utilisées")
        st.markdown("""
        - **Python 3.11**
        - **yfinance** — collecte des données
        - **pandas / numpy** — manipulation des données
        - **scikit-learn** — modèles ML
        - **PyPortfolioOpt** — optimisation Markowitz
        - **Plotly / Streamlit** — visualisation
        """)

    with col2:
        st.subheader("Résultats clés")
        st.markdown("""
        - 22 actions JSE analysées sur 5 ans
        - 13 features (8 fondamentaux + 5 dynamiques)
        - AUC de 0.46 — honnête pour un marché émergent
        - +6.5% ML vs +1.3% Buy & Hold en 2022
        - 5 actions dans le portefeuille final
        """)


# ─────────────────────────────────────────────
# Page 2 : Exploration des données
# ─────────────────────────────────────────────

elif page == "📊 Exploration des données":
    st.title("📊 Exploration des données JSE")

    # Prix normalisés
    st.subheader("Performance des actions (base 100)")

    selected_tickers = st.multiselect(
        "Sélectionne les actions à afficher",
        options=prices.columns.tolist(),
        default=["GFI.JO", "NPN.JO", "MTN.JO", "SBK.JO"]
    )

    if selected_tickers:
        prices_norm = prices[selected_tickers] / \
                      prices[selected_tickers].iloc[0] * 100
        fig = px.line(
            prices_norm,
            title="Performance normalisée (base 100)",
            labels={"value": "Performance", "Date": "Date"},
            template="plotly_white"
        )
        fig.update_layout(legend_title="Ticker")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Fondamentaux
    st.subheader("Ratios financiers")

    ratio = st.selectbox(
        "Sélectionne un ratio",
        options=[
            "trailingPE", "priceToBook", "debtToEquity",
            "returnOnEquity", "returnOnAssets",
            "profitMargins", "currentRatio"
        ],
        format_func=lambda x: {
            "trailingPE"    : "P/E Ratio",
            "priceToBook"   : "Price/Book",
            "debtToEquity"  : "Debt/Equity",
            "returnOnEquity": "Return on Equity",
            "returnOnAssets": "Return on Assets",
            "profitMargins" : "Profit Margin",
            "currentRatio"  : "Current Ratio"
        }.get(x, x)
    )

    fund_plot = fundamentals[[ratio]].reset_index()
    fund_plot.columns = ["ticker", "valeur"]
    fund_plot = fund_plot.sort_values("valeur", ascending=True)

    fig = px.bar(
        fund_plot,
        x="valeur", y="ticker",
        orientation="h",
        title=f"Comparaison des actions — {ratio}",
        template="plotly_white",
        color="valeur",
        color_continuous_scale=["#D85A30", "#F5C4B3", "#1D9E75"]
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# Page 3 : Modèle ML
# ─────────────────────────────────────────────

elif page == "🤖 Modèle ML":
    st.title("🤖 Modèle de Machine Learning")

    # Importance des features
    st.subheader("Importance des features")

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES
    ).sort_values(ascending=True)

    fig = px.bar(
        x=importance.values,
        y=importance.index,
        orientation="h",
        title="Importance des features — Gradient Boosting",
        labels={"x": "Importance", "y": "Feature"},
        color=importance.values,
        color_continuous_scale=["#F5C4B3", "#1D9E75"],
        template="plotly_white"
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Scores ML en temps réel
    st.subheader("Scores ML — Dernière date disponible")

    last_date = dataset["date"].max()
    snapshot  = dataset[dataset["date"] == last_date].copy()
    snapshot["proba"] = model.predict_proba(
        snapshot[FEATURES]
    )[:, 1]
    snapshot = snapshot.sort_values("proba", ascending=False)

    threshold = st.slider(
        "Seuil de sélection",
        min_value=0.40,
        max_value=0.80,
        value=0.55,
        step=0.05
    )

    snapshot["sélectionnée"] = snapshot["proba"] >= threshold

    fig = px.bar(
        snapshot,
        x="proba", y="ticker",
        orientation="h",
        color="sélectionnée",
        color_discrete_map={True: "#1D9E75", False: "#D85A30"},
        title=f"Scores ML par action (seuil = {threshold:.0%})",
        labels={
            "proba" : "Probabilité de surperformance",
            "ticker": "Action"
        },
        template="plotly_white"
    )
    fig.add_vline(
        x=threshold, line_dash="dash", line_color="gray"
    )
    st.plotly_chart(fig, use_container_width=True)

    n_selected = snapshot["sélectionnée"].sum()
    st.info(f"**{n_selected} actions sélectionnées** "
            f"avec un seuil de {threshold:.0%}")


# ─────────────────────────────────────────────
# Page 4 : Backtesting
# ─────────────────────────────────────────────

elif page == "📈 Backtesting":
    st.title("📈 Backtesting — 2022")

    # Métriques
    ml_final = bt_results["ml_cumulative"].iloc[-1]
    bh_final = bt_results["bh_cumulative"].iloc[-1]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Capital final ML",
            f"{ml_final:,.0f} ZAR",
            delta=f"+{ml_final - 100_000:,.0f} ZAR"
        )
    with col2:
        st.metric(
            "Capital final B&H",
            f"{bh_final:,.0f} ZAR",
            delta=f"+{bh_final - 100_000:,.0f} ZAR"
        )
    with col3:
        advantage = (ml_final - bh_final) / 100_000
        st.metric(
            "Avantage ML",
            f"{advantage:+.1%}"
        )

    st.markdown("---")

    # Performance cumulée
    st.subheader("Performance cumulée")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt_results["date"],
        y=bt_results["ml_cumulative"],
        mode="lines+markers",
        name="Stratégie ML",
        line=dict(color="#1D9E75", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=bt_results["date"],
        y=bt_results["bh_cumulative"],
        mode="lines+markers",
        name="Buy & Hold",
        line=dict(color="#D85A30", width=2.5)
    ))
    fig.add_hline(
        y=100_000, line_dash="dash",
        line_color="gray",
        annotation_text="Capital initial"
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Valeur (ZAR)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Rendements mensuels
    st.subheader("Rendements mensuels")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bt_results["month"],
        y=bt_results["ml_return"] * 100,
        name="ML", marker_color="#1D9E75", opacity=0.8
    ))
    fig.add_trace(go.Bar(
        x=bt_results["month"],
        y=bt_results["bh_return"] * 100,
        name="B&H", marker_color="#D85A30", opacity=0.8
    ))
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        xaxis_title="Mois",
        yaxis_title="Rendement (%)"
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# Page 5 : Portefeuille optimal
# ─────────────────────────────────────────────

elif page == "💼 Portefeuille optimal":
    st.title("💼 Portefeuille optimal — Markowitz")

    st.markdown(
        "Allocation optimale calculée par la théorie de Markowitz "
        "sur les actions sélectionnées par le modèle ML."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Graphique en donut
        fig = px.pie(
            allocation,
            values="poids_%",
            names="ticker",
            title="Répartition du portefeuille",
            color_discrete_sequence=[
                "#1D9E75", "#534AB7", "#D85A30",
                "#BA7517", "#185FA5", "#888780"
            ],
            hole=0.3,
            template="plotly_white"
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Tableau détaillé
        st.subheader("Détail de l'allocation")
        display = allocation.copy()
        display.columns = [
            "Ticker", "Poids (%)",
            "Nb actions", "Prix (ZAR)", "Montant (ZAR)"
        ]
        st.dataframe(display, use_container_width=True)

        st.markdown("---")
        st.metric(
            "Capital investi",
            f"{allocation['montant_ZAR'].sum():,.0f} ZAR"
        )
        st.metric(
            "Capital restant",
            f"{100_000 - allocation['montant_ZAR'].sum():,.0f} ZAR"
        )

    st.markdown("---")

    # Performance historique
    st.subheader("Performance historique du portefeuille")

    weights_dict = dict(zip(
        allocation["ticker"],
        allocation["poids_%"] / 100
    ))

    prices_port = prices[list(weights_dict.keys())].dropna()
    daily_ret   = prices_port.pct_change().dropna()
    weighted    = daily_ret.multiply(
        pd.Series(weights_dict), axis=1
    ).sum(axis=1)
    port_value  = 100_000 * (1 + weighted).cumprod()

    bench_value = 100_000 * (
        1 + prices.mean(axis=1).pct_change().dropna()
    ).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=port_value.index,
        y=port_value.values,
        name="Portefeuille optimisé",
        line=dict(color="#1D9E75", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=bench_value.index,
        y=bench_value.values,
        name="Benchmark JSE",
        line=dict(color="#D85A30", width=2, dash="dash")
    ))
    fig.add_hline(
        y=100_000, line_dash="dot",
        line_color="gray",
        annotation_text="Capital initial"
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Valeur (ZAR)"
    )
    st.plotly_chart(fig, use_container_width=True)