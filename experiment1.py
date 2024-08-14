import datetime as dt
import random

import pandas as pd
import util as ut
import numpy as np
from marketsimcode import compute_portvals_updated, evaluate_protfolio
import matplotlib.pyplot as plt
import StrategyLearner as SL
import ManualStrategy as MS


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

def run():
    strategy = SL.StrategyLearner(verbose=True, impact=0.005, commission=9.95)
    manual_strategy = MS.ManualStrategy(verbose=True, impact=0.005, commission=9.95)
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    strategy.add_evidence('JPM', sd=sd, ed=ed)
    plt.figure(figsize=(10, 6))

    # insample
    trades_learner = strategy.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)['JPM']
    trades_manual = manual_strategy.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)['JPM']

    trades_bench = trades_learner.copy()
    trades_bench[:] = 0
    trades_bench[trades_bench.index[0]] = 1000
    values_learner = compute_portvals_updated(trades_learner, 'JPM', 100000, strategy.commission, strategy.impact)
    values_learner_normalized = values_learner / values_learner[values_learner.index[0]]
    values_manual = compute_portvals_updated(trades_manual, 'JPM', 100000, strategy.commission, strategy.impact)
    values_manual_normalized = values_manual / values_manual[values_manual.index[0]]
    values_bench = compute_portvals_updated(trades_bench, 'JPM', 100000, strategy.commission, strategy.impact)
    values_bench_normalized = values_bench / values_bench[values_learner.index[0]]
    values_learner_normalized.plot(color='red', label='Learner')
    values_bench_normalized.plot(color='purple', label='Benchmark')
    values_manual_normalized.plot(color='blue', label='Manual')
    plt.title('Learner vs manual vs Benchmark in-sample portfolios')
    plt.legend(loc='best')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid()
    plt.savefig('./exp1_fig1.png')
    plt.clf()

    # out-sample
    trades_learner = strategy.testPolicy(symbol='JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31),
                                         sv=100000)['JPM']
    trades_manual = manual_strategy.testPolicy(symbol='JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31),
                                               sv=100000)['JPM']

    trades_bench = trades_learner.copy()
    trades_bench[:] = 0
    trades_bench[trades_bench.index[0]] = 1000
    values_learner = compute_portvals_updated(trades_learner, 'JPM', 100000, strategy.commission, strategy.impact)
    values_learner_normalized = values_learner / values_learner[values_learner.index[0]]
    values_manual = compute_portvals_updated(trades_manual, 'JPM', 100000, strategy.commission, strategy.impact)
    values_manual_normalized = values_manual / values_manual[values_manual.index[0]]
    values_bench = compute_portvals_updated(trades_bench, 'JPM', 100000, strategy.commission, strategy.impact)
    values_bench_normalized = values_bench / values_bench[values_learner.index[0]]
    values_learner_normalized.plot(color='red', label='Learner')
    values_bench_normalized.plot(color='purple', label='Benchmark')
    values_manual_normalized.plot(color='blue', label='Manual')
    plt.title('Learner vs manual vs Benchmark out-sample portfolios')
    plt.legend(loc='best')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid()
    plt.savefig('./exp1_fig2.png')
    plt.clf()

if __name__ == "__main__":
    run()