import numpy as np
import pandas as pd
from util import get_data, plot_data
import datetime as dt

def evaluate_protfolio(protfolio_values):
    daily_prot_value = protfolio_values.values
    daily_return = np.diff(daily_prot_value) / daily_prot_value[:-1]
    cumulative_return = daily_prot_value[-1] / daily_prot_value[0] - 1
    average_daily_return = np.mean(daily_return)
    volatility = np.std(daily_return)
    sharpe_ratio = average_daily_return / volatility
    sharpe_ratio *= 252 ** 0.5
    return cumulative_return, average_daily_return, volatility, sharpe_ratio

def compute_portvals_updated(
        orders,
        symbol = 'JPM',
        start_val=100000,
        commission=0,
        impact=0,
):
    start_date = orders.index[0]
    end_date = orders.index[-1]

    # getting prices on the
    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices = prices.rename(columns={"SPY": 'CASH'})
    prices.loc[:, 'CASH'] = 1

    # print(prices)

    # filling transaction table
    transactions = prices.copy()
    transactions.loc[:, :] = 0

    transactions.loc[:, symbol] = orders
    transactions.loc[:, 'CASH'] = orders * prices[symbol] * -1
    transactions.loc[orders != 0, 'CASH'] -= commission + (impact*prices[symbol])

    # print(transactions)

    # filling out protfolio position
    holdings = prices.copy()
    holdings.loc[:, :] = 0
    for date, row in transactions.iterrows():
        if date == start_date:
            holdings.loc[date, 'CASH'] = start_val
            holdings.loc[date] = holdings.loc[date] + transactions.loc[date]
        else:
            holdings.loc[date] = holdings.loc[prev_date] + transactions.loc[date]
        prev_date = date

    # filling out values
    values = holdings * prices
    values = values.sum(axis=1)

    return values
def author():
    return "pdawoud37"

def study_group():
    return "pdawoud37"
