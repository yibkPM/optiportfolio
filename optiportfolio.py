# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 22:06:15 2023

@author: ibour
"""

import streamlit as st
import yfinance as yf
import pandas as pd 
import numpy as np 
import requests
from PIL import Image
import matplotlib.pyplot as plt
plt.style.use('default')

st.title('Portfolio Optimization - OUISTITI CAPITAL')
st.write('La singerie au coeur de la finance')

image = Image.open("360_F_568271862_YRGdNXT2ft0Xzjw8vf6CmkE8nQXbTfKJ.jpg")

st.image(image, caption='Ouistiti Capital Analyst')

assets = st.text_input("Renseigner les actifs (séparation par une virgule)", "AAPL, MSFT, GOOGL, AMZN, AMD, TSL, NFLX")

start = st.date_input("Mettre une date initiale pour votre analyse", value=pd.to_datetime('2022-06-01'))

data = yf.download(assets,start=start)['Adj Close']

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

st.subheader("Rendements Moyens et Matrice de Covariance")

moy = expected_returns.mean_historical_return(data)
S = risk_models.sample_cov(data)

st.write("Données des prix de clôture ajustés (Adj Close) :")
st.write(data)

# Afficher le rendement moyen
st.write("Rendement moyen des actifs :")
st.write(moy)

# Afficher la matrice de covariance
st.write("Matrice de covariance des rendements des actifs :")
st.write(S)


#Portfolio Optimization II

st.subheader('Portfolio Optimization - Part2')

# Obtenez les actifs de l'utilisateur, séparés par des virgules
#assets_input = st.text_input(
 #   "Renseigner les actifs (séparation par une virgule)", 
  #  "AAPL, MSFT, GOOGL, AMZN, AMD, TSLA, NFLX"
#)
assets_input=assets
# Divisez la chaîne de caractères pour créer une liste d'actifs
assets = [asset.strip() for asset in assets_input.split(',')]

def manage_portfolio_weights(assets):
    total_weight = 100
    st.write("Gestion des poids des actifs")

    weights = {}
    remaining_weight = total_weight
    
    for asset in assets:
        unique_key = f"slider_{asset}"
        max_weight = remaining_weight if remaining_weight > 0 else 0
        current_weight = 0 if remaining_weight == total_weight else int(remaining_weight / (len(assets) - len(weights)))
        weights[asset] = st.slider(f"Poids de {asset}", 0, max_weight, current_weight, key=unique_key)
        remaining_weight -= weights[asset]

    if sum(weights.values()) != total_weight:
        st.error("La somme des poids doit être égale à 100%")
    else:
        st.success("La répartition des poids est valide.")
    
    return weights

# Affichez les actifs sélectionnés et gérez les poids des actifs si la liste n'est pas vide
if assets:
    st.write(f"Actifs sélectionnés: {assets}")
    st.line_chart(data) # Cette ligne doit être décommentée si 'data' est défini
    weights = manage_portfolio_weights(assets)
    st.write("Poids des actifs:", weights)

    weights_array = np.array(list(weights.values())) / 100
    
    # Simulation de données pour les rendements quotidiens (à remplacer par vos données réelles)
    returns = data.pct_change()
    st.write("Rendements quotidiens de chaque actif")
    st.write(returns)
    st.write("Matrice de covariance annualisée")
    cov_matrix_annual = returns.cov() * 252
    st.write(cov_matrix_annual)

    # Calcul de la variance et volatilité du portefeuille
    port_variance = np.dot(weights_array.T, np.dot(cov_matrix_annual, weights_array))
    port_volatility = np.sqrt(port_variance)
    # Calcul du rendement annuel du portefeuille
    portfolio_simple_annual_returns = np.sum(returns.mean()*weights_array)*252
    
    percent_var = str(round(port_variance,2)*100)+'%'
    percent_vola = str(round(port_volatility,2)*100)+'%'
    percent_ret=str(round(portfolio_simple_annual_returns,2)*100)+'%'
    st.write(f"La variance du portefeuille est égale à : {percent_var}")
    st.write(f"La volatilité du portefeuille est égale à : {percent_vola}")
    st.write(f"Le rendement annuel du portefeuille est égal à : {percent_ret}")
    
    #Portfolio Optimization
    
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import risk_models
    from pypfopt import expected_returns
    from pypfopt import plotting
    moy = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    # Optimization of the Sharpe Ratio
    ef = EfficientFrontier(moy,S,weight_bounds=(None,None))
    weights = ef.max_sharpe()
    cleaned_weights=ef.clean_weights()
    st.write(cleaned_weights)
    # Obtenez le rendement attendu
    expected_return = ef.portfolio_performance(verbose=True)[0]
    Annual_Volatility = ef.portfolio_performance(verbose=True)[1]
    Sharpe_Ratio = ef.portfolio_performance(verbose=True)[2]

    # Formatez le rendement attendu avec 2 décimales
    expected_return_formatted = "{:.2f}%".format(expected_return * 100)
    Annual_Voaltility_formatted = "{:.2f}%".format(Annual_Volatility * 100)
    Sharpe_Ratio_formatted = "{:.2f}".format(Sharpe_Ratio)

    st.subheader("Optimisation du portefeuille avec Maximisation du Ratio de Sharpe :")
    st.write(f"Expected annual return: {expected_return_formatted}")
    st.write(f"Annual Volatility: {Annual_Voaltility_formatted}")
    st.write(f"Sharpe Ratio: {Sharpe_Ratio_formatted}")
    
    ef_for_plotting = EfficientFrontier(moy, S, weight_bounds=(None, None))
    fig, ax = plt.subplots()
    plotting.plot_efficient_frontier(ef_for_plotting, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Pour trouver le portefeuille Tangent (Markowitz Theory) ")
    # Optimization of the Sharpe Ratio
    ef_tangent = EfficientFrontier(moy,S,weight_bounds=(None,None))
    fig2, ax2 = plt.subplots()
    plotting.plot_efficient_frontier(ef_tangent, ax=ax2)
    # Ptf Tangent
    ret_tangent, std_tangent, _=ef_tangent.portfolio_performance()
    ax2.scatter(std_tangent,ret_tangent,marker="*",s=100,c="r",label="Max Ratio Sharpe")
    #Random portfolios
    n_samples = 10000
    w = np.random.dirichlet(np.ones(len(moy)),n_samples)
    rets = w.dot(moy)
    stds = np.sqrt(np.diag(w @ S @ w.T))
    sharpes = rets/stds
    ax2.scatter(stds,rets,marker=".",c=sharpes,cmap="viridis_r")
    #En sortie
    ax2.set_title("Frontière efficiente avec génération aléatoire de portefeuilles")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("ef_scatter.png",dpi=200)
    st.pyplot(fig2)
    
    
     
    #Allocation 

    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    st.subheader("Optimisation de l'Allocation d'actifs pour chaque part d'actions :")
    portfolio_val = st.number_input(" Insérer le montant de votre portefeuille: ", 10000)
    latest_prices = get_latest_prices(data)
    weights = cleaned_weights
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = portfolio_val)
    allocation, leftover = da.lp_portfolio()
    st.write(f" Allocation de votre portefeuille (en nombre de titres) ': {allocation}")
    st.write(f" Fonds restant': {leftover}")
    
    import matplotlib.pyplot as plt
    import datetime as dt
    from pandas_datareader import data as pdr
    def get_data(stocks, start, end):
        stockData =  yf.download(assets,start=start)['Adj Close']
        returns = stockData.pct_change()
        meanReturns = returns.mean()
        covMatrix = returns.cov()
        return meanReturns, covMatrix
    stockList = assets
    stocks = [stock for stock in stockList]
    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days=300)
    meanReturns, covMatrix = get_data(stocks, startDate, endDate)
    weights = np.random.random(len(meanReturns))
    weights /= np.sum(weights)

    mc_sims = 100 # number of simulations
    T = 100 #timeframe in days
    meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
    meanM = meanM.T
    portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
    initialPortfolio = portfolio_val
    for m in range(0, mc_sims):
        Z = np.random.normal(size=(T, len(weights)))#uncorrelated RV's
        L = np.linalg.cholesky(covMatrix) #Cholesky decomposition to Lower Triangular Matrix
        dailyReturns = meanM + np.inner(L, Z) #Correlated daily returns for individual stocks
        portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio
    plt.plot(portfolio_sims)
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Days')
    plt.title('Monte Carlo simulation of a stock portfolio')
    plt.show()
    st.subheader("Simulation de Monte Carlo:")
    st.line_chart(portfolio_sims)
    st.pyplot(plt)
    
    





    

