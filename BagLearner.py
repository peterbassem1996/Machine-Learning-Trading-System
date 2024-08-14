import numpy as np
import pandas as pd

class BagLearner:
    def __init__(self, learner, kwargs={"argument1": 1, "argument2": 2}, bags=20, boost=False, verbose=False):
        self.boost = boost
        self.verbose = verbose
        self.learners = [learner(**kwargs) for _ in range(bags)]

    def add_evidence(self, data_x, data_y):
        # print(data_x)
        # print(data_y)
        n = data_x.shape[0]
        for learner in self.learners:
            indices = np.random.choice(n, 260, replace=True)
            learner.add_evidence(data_x.iloc[indices], data_y.iloc[indices])
            if self.verbose:
                print("trained using {}".format(indices))

    def query(self, points):
        output = points.copy()
        output.drop(columns=output.columns, inplace=True)
        i = 1
        for learner in self.learners:
            output[f'{i}'] = learner.query(points)
            i += 1

        # print(output)
        output = output.mode(axis=1)[0]
        return output


    def author(self):
        return "pdawoud37"

    def study_group(self):
        return "pdawoud37"