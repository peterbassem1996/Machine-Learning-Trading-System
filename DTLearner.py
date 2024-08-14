import numpy as np
import scipy.stats as st
import pandas as pd

class DTLearner:
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def find_best_feature(self, data_x, data_y):
        best_index = None
        best_correlation = -1
        for i in data_x.columns:
            if np.std(data_x[i]) == 0:
                continue
            correlation_matrix = np.corrcoef(data_x[i], data_y)
            correlation = abs(correlation_matrix[0, 1])
            if correlation > best_correlation:
                best_correlation = correlation
                best_index = i

        return best_index

    def build_tree(self, data_x, data_y):
        if data_x.shape[0] <= self.leaf_size or len(set(data_y)) == 1:
            return np.array((data_y.mode()[0], ))

        best_feat = self.find_best_feature(data_x, data_y)
        split_val = np.median(data_x[best_feat])

        left_tree_indices = np.where(data_x[best_feat] <= split_val)[0]
        right_tree_indices = np.where(data_x[best_feat] > split_val)[0]

        # print(left_tree_indices, right_tree_indices)

        if left_tree_indices.shape[0] == 0 or right_tree_indices.shape[0] == 0:
            return np.array((data_y.mode()[0], ))

        left_tree = self.build_tree(data_x.iloc[left_tree_indices], data_y.iloc[left_tree_indices])
        right_tree = self.build_tree(data_x.iloc[right_tree_indices], data_y.iloc[right_tree_indices])

        return np.array((best_feat, split_val, left_tree, right_tree), dtype=object)

    def add_evidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)
        if self.verbose:
            print("tree built:", self.tree)

    # def query_point(self, point):
    #     temp_tree = self.tree
    #     print(point)
    #     while len(temp_tree) > 1:
    #         if point[temp_tree[0]] <= temp_tree[1]:
    #             temp_tree = temp_tree[2]
    #         else:
    #             temp_tree = temp_tree[3]
    #
    #     # print('test')
    #     return temp_tree[0]

    def query(self, points):
        # return np.array([self.query_point(point) for point in points])
        # print(self.tree)
        points['PRED'] = 0
        for index, row in points.iterrows():
            temp_tree = self.tree
            while len(temp_tree) > 1:
                if row[temp_tree[0]] <= temp_tree[1]:
                    temp_tree = temp_tree[2]
                else:
                    temp_tree = temp_tree[3]
            # print(temp_tree[0])
            points.loc[index, 'PRED'] = temp_tree[0]
        return points['PRED']


    @staticmethod
    def get_tree_depth(tree):
        if len(tree) == 1: return 1
        left = DTLearner.get_tree_depth(tree[2])
        right = DTLearner.get_tree_depth(tree[3])
        return 1 + max(left, right)

    def tree_depth(self):
        return DTLearner.get_tree_depth(self.tree)

    def author(self):
        return "pdawoud37"

    def study_group(self):
        return "pdawoud37"