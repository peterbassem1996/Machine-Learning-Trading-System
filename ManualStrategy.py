import pandas as pd
import datetime as dt
from util import get_data
from indicators import bollinger_bands, macd, rsi, roc
from marketsimcode import compute_portvals_updated, evaluate_protfolio
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None)  # Show full width of each column
pd.set_option('display.expand_frame_repr', False)  # Do not wrap display


class ManualStrategy:
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        A manual learner that can learn a trading policy using the same indicators used in StrategyLearner.

        Parameters:
            verbose (bool): If True, print out information for debugging. If False, do not generate any output.
            impact (float): The market impact of each transaction, defaults to 0.0.
            commission (float): The commission amount charged, defaults to 0.0.
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def _prepare_signal(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31)):
        prices = get_data([symbol], pd.date_range(sd, ed))
        prices.drop(columns=['SPY'], inplace=True)
        prices = bollinger_bands(prices, window=10, num_std_dev=2, symbol=symbol)
        prices = macd(prices, short_span=6, long_span=13, signal_span=5, symbol=symbol)
        prices = rsi(prices, window=14, symbol=symbol)
        prices = roc(prices, period=5, symbol=symbol)
        prices['BB_S'] = (prices[symbol] - prices['Lower Band']) / (prices['Upper Band'] - prices['Lower Band'])
        prices['MACD_S'] = prices['MACD'] - prices['Signal_Line']
        prices['RSI_S'] = prices['RSI']
        prices['ROC_S'] = prices['ROC']

        return prices[['BB_S', 'MACD_S', 'RSI_S', 'ROC_S']]

    def add_evidence(self, symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        """
        Trains the strategy learner over a given time frame.

        Parameters:
            symbol (str): The stock symbol to train on.
            sd (datetime): The start date, defaults to 1/1/2008.
            ed (datetime): The end date, defaults to 1/1/2009.
            sv (int): The starting value of the portfolio.
        """

        pass

    def testPolicy(self, symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        """
        Tests the learner using data outside of the training data.

        Parameters:
            symbol (str): The stock symbol that was trained on.
            sd (datetime): The start date, defaults to 1/1/2008.
            ed (datetime): The end date, defaults to 1/1/2009.
            sv (int): The starting value of the portfolio.

        Returns:
            pandas.DataFrame: A DataFrame with values representing trades for each day. Legal values are +1000.0 (BUY),
            -1000.0 (SELL), and 0.0 (NOTHING). Values of +2000 and -2000 for trades are also legal when switching from long
            to short or short to long, so long as net holdings are constrained to -1000, 0, and 1000.
        """
        #prepare signals from indicators
        signals = ManualStrategy._prepare_signal(symbol, sd, ed)

        #create voting system
        signals['votes'] = 0
        signals.loc[signals['BB_S'] < 0, 'votes'] += -1
        signals.loc[signals['BB_S'] > 1, 'votes'] += 1
        signals.loc[signals['MACD_S'] < 0, 'votes'] += -1
        signals.loc[signals['MACD_S'] > 0, 'votes'] += 1
        signals.loc[signals['RSI_S'] < 30, 'votes'] += -1
        signals.loc[signals['RSI_S'] > 70, 'votes'] += 1
        signals.loc[signals['ROC_S'] < 0, 'votes'] += -1
        signals.loc[signals['ROC_S'] > 0, 'votes'] += 1

        #fill in trades
        state = 0
        trades = get_data([symbol], pd.date_range(sd, ed))
        trades.drop(columns=['SPY'], inplace=True)
        trades[symbol] = 0.0  # Initialize with no trades
        for index, row in signals.iterrows():
            if row['votes'] >= 3:
                if state == 0: trades.loc[index][symbol] = -1000
                elif state == 1: trades.loc[index][symbol] = -2000
                state = -1
            elif row['votes'] <= -3:
                if state == 0: trades.loc[index][symbol]= 1000
                elif state == -1: trades.loc[index][symbol] = 2000
                state = 1

        #compute protfolio values
        return trades

    def author(self):
        """
        Returns the GT username of the student.

        Returns:
            str: The GT username.
        """
        return "pdawoud3"  # Replace with your GT username

    def study_group(self):
        """
        Returns a comma-separated string of GT_Name of each member of your study group.

        Returns:
            str: A comma-separated string of GT_Name(s).
        """
        return "pdawoud3"  # Replace with actual study group members' GT usernames


# Example usage:
def run():
    strategy = ManualStrategy(verbose=True, impact=0.005, commission=9.95)
    plt.figure(figsize=(10, 6))

    #insample
    trades_manual = strategy.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    trades_manual = trades_manual['JPM']
    trades_bench = trades_manual.copy()
    trades_bench[:] = 0
    trades_bench[trades_bench.index[0]] = 1000
    values_manual = compute_portvals_updated(trades_manual, 'JPM', 100000, strategy.commission, strategy.impact)
    values_manual_normalized = values_manual / values_manual[values_manual.index[0]]
    values_bench = compute_portvals_updated(trades_bench, 'JPM', 100000, strategy.commission, strategy.impact)
    values_bench_normalized = values_bench / values_bench[values_manual.index[0]]
    values_manual_normalized.plot(color = 'red', label='Manual')
    values_bench_normalized.plot(color = 'purple', label='Benchmark')
    plt.vlines(x=trades_manual.index[np.where(trades_manual > 0)],
               ymin=min(values_bench_normalized.min(), values_manual_normalized.min()),
               ymax=max(values_bench_normalized.max(), values_manual_normalized.max()),
               colors='blue',
               label='Long')
    plt.vlines(x=trades_manual.index[np.where(trades_manual < 0)],
               ymin=min(values_bench_normalized.min(), values_manual_normalized.min()),
               ymax=max(values_bench_normalized.max(), values_manual_normalized.max()),
               colors='Black',
               label='Short')
    plt.title('Manual vs Benchmark in-sample portfolios')
    plt.legend(loc='best')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid()
    plt.savefig('./manual_fig1.png')
    plt.clf()

    file = open('./manual_report.txt', 'w')

    cumulative_return_benchmark, average_daily_return_benchmark, volatility_benchmark, sharpe_ratio_benchmark = evaluate_protfolio(values_bench)
    cumulative_return_manual, average_daily_return_manual, volatility_manual, sharpe_ratio_manual = evaluate_protfolio(values_manual)

    file.write('IN-SAMPLE\n')
    file.write(f"Cumulative Return of manual strategy portfolio: {cumulative_return_manual}\n")
    file.write(f"Cumulative Return of benchmark portfolio: {cumulative_return_benchmark}\n")
    file.write('\n')
    file.write(f"Standard Deviation of manual strategy portfolio: {volatility_manual}\n")
    file.write(f"Standard Deviation of benchmark portfolio: {volatility_benchmark}\n")
    file.write('\n')
    file.write(f"Average Daily Return of manual strategy portfolio: {average_daily_return_manual}\n")
    file.write(f"Average Daily Return of benchmark portfolio: {average_daily_return_benchmark}\n")
    file.write('\n')
    file.write(f"Sharpe ratio of manual strategy portfolio: {sharpe_ratio_manual}\n")
    file.write(f"Sharpe ratio of benchmark portfolio: {sharpe_ratio_benchmark}\n")
    file.write('\n')
    file.write('\n')


    # out-sample
    trades_manual = strategy.testPolicy(symbol='JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31),
                                         sv=100000)
    trades_manual = trades_manual['JPM']
    trades_bench = trades_manual.copy()
    trades_bench[:] = 0
    trades_bench[trades_bench.index[0]] = 1000
    values_manual = compute_portvals_updated(trades_manual, 'JPM', 100000, strategy.commission, strategy.impact)
    values_manual_normalized = values_manual / values_manual[values_manual.index[0]]
    values_bench = compute_portvals_updated(trades_bench, 'JPM', 100000, strategy.commission, strategy.impact)
    values_bench_normalized = values_bench / values_bench[values_manual.index[0]]
    values_manual_normalized.plot(color='red', label='Manual')
    values_bench_normalized.plot(color='purple', label='Benchmark')
    plt.vlines(x=trades_manual.index[np.where(trades_manual > 0)],
               ymin=min(values_bench_normalized.min(), values_manual_normalized.min()),
               ymax=max(values_bench_normalized.max(), values_manual_normalized.max()),
               colors='blue',
               label='Long')
    plt.vlines(x=trades_manual.index[np.where(trades_manual < 0)],
               ymin=min(values_bench_normalized.min(), values_manual_normalized.min()),
               ymax=max(values_bench_normalized.max(), values_manual_normalized.max()),
               colors='Black',
               label='Short')
    plt.title('Manual vs Benchmark out-sample portfolios')
    plt.legend(loc='best')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid()
    plt.savefig('./manual_fig2.png')
    plt.clf()

    cumulative_return_benchmark, average_daily_return_benchmark, volatility_benchmark, sharpe_ratio_benchmark = evaluate_protfolio(
        values_bench)
    cumulative_return_manual, average_daily_return_manual, volatility_manual, sharpe_ratio_manual = evaluate_protfolio(
        values_manual)

    file.write('OUT-SAMPLE\n')
    file.write(f"Cumulative Return of manual strategy portfolio: {cumulative_return_manual}\n")
    file.write(f"Cumulative Return of benchmark portfolio: {cumulative_return_benchmark}\n")
    file.write('\n')
    file.write(f"Standard Deviation of manual strategy portfolio: {volatility_manual}\n")
    file.write(f"Standard Deviation of benchmark portfolio: {volatility_benchmark}\n")
    file.write('\n')
    file.write(f"Average Daily Return of manual strategy portfolio: {average_daily_return_manual}\n")
    file.write(f"Average Daily Return of benchmark portfolio: {average_daily_return_benchmark}\n")
    file.write('\n')
    file.write(f"Sharpe ratio of manual strategy portfolio: {sharpe_ratio_manual}\n")
    file.write(f"Sharpe ratio of benchmark portfolio: {sharpe_ratio_benchmark}\n")

    file.close()
    # print(strategy.author())
    # print(strategy.study_group())

if __name__ == "__main__":
    run()