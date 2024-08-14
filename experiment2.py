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
    impacts = [0.0005, 0.005, 0.05]
    plt.figure(figsize=(10, 6))
    file = open('./experiment2_report.txt', 'w')
    for impact in impacts:
        strategy = SL.StrategyLearner(verbose=True, impact=impact, commission=0)
        sd = dt.datetime(2008, 1, 1)
        ed = dt.datetime(2009, 12, 31)
        strategy.add_evidence('JPM', sd=sd, ed=ed)
        trades_learner = strategy.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)['JPM']
        values_learner = compute_portvals_updated(trades_learner, 'JPM', 100000, strategy.commission, strategy.impact)
        values_learner_normalized = values_learner / values_learner[values_learner.index[0]]
        values_learner_normalized.plot(label=f'impact = {impact}')
        cumulative_return_learner, average_daily_return_learner, volatility_learner, sharpe_ratio_learner = evaluate_protfolio(
            values_learner)
        file.write(f"Cumulative Return of impact {impact}: {cumulative_return_learner}\n")
        file.write(f"Long entrances of impact {impact}: {(trades_learner > 0).sum()}\n")
        file.write(f"short entrances of impact {impact}: {(trades_learner < 0).sum()}\n")
        file.write('\n')
    plt.title('Learner vs manual vs Benchmark in-sample portfolios')
    plt.legend(loc='best')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid()
    plt.savefig('./exp2_fig1.png')
    plt.clf()
    file.close()

if __name__ == "__main__":
    run()