""""""  		  	   		 	   			  		 			 	 	 		 		 	
"""  		  	   		 	   			  		 			 	 	 		 		 	
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	   			  		 			 	 	 		 		 	
  		  	   		 	   			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		 	   			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		 	   			  		 			 	 	 		 		 	
  		  	   		 	   			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		 	   			  		 			 	 	 		 		 	
  		  	   		 	   			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		 	   			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   			  		 			 	 	 		 		 	
or edited.  		  	   		 	   			  		 			 	 	 		 		 	
  		  	   		 	   			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		 	   			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		 	   			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		 	   			  		 			 	 	 		 		 	
  		  	   		 	   			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		 	   			  		 			 	 	 		 		 	
  		  	   		 	   			  		 			 	 	 		 		 	
Student Name: Tucker Balch (replace with your name)  		  	   		 	   			  		 			 	 	 		 		 	
GT User ID: tb34 (replace with your User ID)  		  	   		 	   			  		 			 	 	 		 		 	
GT ID: 900897987 (replace with your GT ID)  		  	   		 	   			  		 			 	 	 		 		 	
"""  		  	   		 	   			  		 			 	 	 		 		 	
  		  	   		 	   			  		 			 	 	 		 		 	
import datetime as dt

import pandas as pd  		  	   		 	   			  		 			 	 	 		 		 	
import util as ut
import DTLearner as dtl
import BagLearner as bll
import numpy as np

from indicators import bollinger_bands, macd, rsi, roc

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None)  # Show full width of each column
pd.set_option('display.expand_frame_repr', False)  # Do not wrap display

look_froward = 7
variance_long = 0.02
variance_short = -0.02
leaves = 26
bags = 30

  		  	   		 	   			  		 			 	 	 		 		 	
class StrategyLearner(object):  		  	   		 	   			  		 			 	 	 		 		 	
    """  		  	   		 	   			  		 			 	 	 		 		 	
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	   			  		 			 	 	 		 		 	
  		  	   		 	   			  		 			 	 	 		 		 	
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   			  		 			 	 	 		 		 	
        If verbose = False your code should not generate ANY output.  		  	   		 	   			  		 			 	 	 		 		 	
    :type verbose: bool  		  	   		 	   			  		 			 	 	 		 		 	
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	   			  		 			 	 	 		 		 	
    :type impact: float  		  	   		 	   			  		 			 	 	 		 		 	
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	   			  		 			 	 	 		 		 	
    :type commission: float  		  	   		 	   			  		 			 	 	 		 		 	
    """  		  	   		 	   			  		 			 	 	 		 		 	
    # constructor

    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		 	   			  		 			 	 	 		 		 	
        """  		  	   		 	   			  		 			 	 	 		 		 	
        Constructor method  		  	   		 	   			  		 			 	 	 		 		 	
        """  		  	   		 	   			  		 			 	 	 		 		 	
        self.verbose = verbose  		  	   		 	   			  		 			 	 	 		 		 	
        self.impact = impact  		  	   		 	   			  		 			 	 	 		 		 	
        self.commission = commission
        self.learner = bll.BagLearner(dtl.DTLearner, kwargs={"leaf_size": leaves}, bags=bags, boost=False, verbose=False)

    def create_confusion_matrix_df(actual, predicted):
        actual = np.array(actual)
        predicted = np.array(predicted)

        # Get unique classes
        classes = np.unique(np.concatenate([actual, predicted]))

        # Initialize confusion matrix
        cm = pd.DataFrame(index=classes, columns=classes, data=0)

        # Populate confusion matrix
        for a, p in zip(actual, predicted):
            cm.loc[a, p] += 1

        # Create a list to store rows for the DataFrame
        data = []

        # Convert the confusion matrix to a DataFrame
        for actual_label in classes:
            for pred_label in classes:
                count = cm.loc[actual_label, pred_label]
                if count > 0:
                    data.append({'Actual': actual_label, 'Predicted': pred_label, 'Count': count})

        # Create the DataFrame
        cm_df = pd.DataFrame(data)

        return cm_df

    def _prepare_signal(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), impact=0):
        prices = ut.get_data([symbol], pd.date_range(sd, ed))
        prices.drop(columns=['SPY'], inplace=True)
        prices = bollinger_bands(prices, window=10, num_std_dev=2, symbol=symbol)
        prices = macd(prices, short_span=6, long_span=13, signal_span=5, symbol=symbol)
        prices = rsi(prices, window=5, symbol=symbol)
        prices = roc(prices, period=5, symbol=symbol)
        prices['BB_S'] = (prices[symbol] - prices['Lower Band']) / (prices['Upper Band'] - prices['Lower Band'])
        prices['MACD_S'] = prices['MACD'] - prices['Signal_Line']
        prices['RSI_S'] = prices['RSI']
        prices['ROC_S'] = prices['ROC']
        prices['future'] = prices[symbol].shift(-look_froward)
        prices['GAIN'] = (prices['future'] - prices[symbol]) / prices[symbol]
        prices['ACTION'] = 0
        prices.loc[prices['GAIN'] >= variance_long + impact, 'ACTION'] = 1
        prices.loc[prices['GAIN'] <= variance_short - impact, 'ACTION'] = -1
        prices.dropna(inplace=True)

        # print(prices)

        return prices[['BB_S', 'MACD_S', 'RSI_S', 'ROC_S', 'ACTION']]

    # this method should create a QLearner, and train it for trading
    def add_evidence(  		  	   		 	   			  		 			 	 	 		 		 	
        self,  		  	   		 	   			  		 			 	 	 		 		 	
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),  		  	   		 	   			  		 			 	 	 		 		 	
        ed=dt.datetime(2009, 12, 31),
        sv=10000,  		  	   		 	   			  		 			 	 	 		 		 	
    ):  		  	   		 	   			  		 			 	 	 		 		 	
        """  		  	   		 	   			  		 			 	 	 		 		 	
        Trains your strategy learner over a given time frame.  		  	   		 	   			  		 			 	 	 		 		 	
  		  	   		 	   			  		 			 	 	 		 		 	
        :param symbol: The stock symbol to train on  		  	   		 	   			  		 			 	 	 		 		 	
        :type symbol: str  		  	   		 	   			  		 			 	 	 		 		 	
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   			  		 			 	 	 		 		 	
        :type sd: datetime  		  	   		 	   			  		 			 	 	 		 		 	
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   			  		 			 	 	 		 		 	
        :type ed: datetime  		  	   		 	   			  		 			 	 	 		 		 	
        :param sv: The starting value of the portfolio  		  	   		 	   			  		 			 	 	 		 		 	
        :type sv: int  		  	   		 	   			  		 			 	 	 		 		 	
        """  		  	   		 	   			  		 			 	 	 		 		 	


        data = StrategyLearner._prepare_signal(symbol, sd, ed, self.impact)
        x = data.copy()
        x.drop(columns=['ACTION'], inplace=True)
        y = data['ACTION'].copy()
        self.learner.add_evidence(x, y)
  		  	   		 	   			  		 			 	 	 		 		 	
    # this method should use the existing policy and test it against new data  		  	   		 	   			  		 			 	 	 		 		 	
    def testPolicy(  		  	   		 	   			  		 			 	 	 		 		 	
        self,  		  	   		 	   			  		 			 	 	 		 		 	
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=10000,  		  	   		 	   			  		 			 	 	 		 		 	
    ):  		  	   		 	   			  		 			 	 	 		 		 	
        """  		  	   		 	   			  		 			 	 	 		 		 	
        Tests your learner using data outside of the training data  		  	   		 	   			  		 			 	 	 		 		 	
  		  	   		 	   			  		 			 	 	 		 		 	
        :param symbol: The stock symbol that you trained on on  		  	   		 	   			  		 			 	 	 		 		 	
        :type symbol: str  		  	   		 	   			  		 			 	 	 		 		 	
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   			  		 			 	 	 		 		 	
        :type sd: datetime  		  	   		 	   			  		 			 	 	 		 		 	
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   			  		 			 	 	 		 		 	
        :type ed: datetime  		  	   		 	   			  		 			 	 	 		 		 	
        :param sv: The starting value of the portfolio  		  	   		 	   			  		 			 	 	 		 		 	
        :type sv: int  		  	   		 	   			  		 			 	 	 		 		 	
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	   			  		 			 	 	 		 		 	
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	   			  		 			 	 	 		 		 	
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	   			  		 			 	 	 		 		 	
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	   			  		 			 	 	 		 		 	
        :rtype: pandas.DataFrame  		  	   		 	   			  		 			 	 	 		 		 	
        """  		  	   		 	   			  		 			 	 	 		 		 	
  		  	   		 	   			  		 			 	 	 		 		 	

        signals = StrategyLearner._prepare_signal(symbol, sd, ed, self.impact)
        x = signals.copy()
        x.drop(columns=['ACTION'], inplace=True)
        signals['pred'] = self.learner.query(x)
        # print(data)

        # print(StrategyLearner.create_confusion_matrix_df(signals['ACTION'], signals['pred']))

        # fill in trades
        state = 0
        trades = ut.get_data([symbol], pd.date_range(sd, ed))
        trades.drop(columns=['SPY'], inplace=True)
        trades[symbol] = 0.0  # Initialize with no trades
        for index, row in signals.iterrows():
            if row['pred'] == -1:
                if state == 0:
                    trades.loc[index][symbol] = -1000
                elif state == 1:
                    trades.loc[index][symbol] = -2000
                state = -1
            elif row['pred'] == 1:
                if state == 0:
                    trades.loc[index][symbol] = 1000
                elif state == -1:
                    trades.loc[index][symbol] = 2000
                state = 1

        return trades

  		  	   		 	   			  		 			 	 	 		 		 	
  		  	   		 	   			  		 			 	 	 		 		 	
if __name__ == "__main__":
    print('Hello World!')
