import numpy as np
from .Node import Node

class DecisionTreeClassifier:

    def __init__(self, max_depth=10, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fillThreshold(self, X, Y, method):
        m = X.shape[1]
        Node.thresholds = [None] * m
        for i in range(m):
            bestValue = DecisionTreeClassifier.best_Value(X, Y, i, method)
            Node.thresholds[i] = bestValue

    @staticmethod
    def compute_impurity(Y):
        impurity = .0
        if len(Y) == 0: return 0
        if (len(np.unique(Y)) == 1): return 0

        p1 = len(Y[Y == 1]) / len(Y)
        p2 = 1 - p1
        impurity = np.log2(p1) * (-p1) - np.log2(p2) * p2
        return impurity

    @staticmethod
    def best_Value(X, Y, feature_no, method):
        values = method(X[:, feature_no])
        maxGain = 0
        bestValue = -1
        for i in values:
            if (DecisionTreeClassifier.info_gain(X, Y, feature_no, i)[0] > maxGain):
                maxGain = DecisionTreeClassifier.info_gain(X, Y, feature_no, i)[0]
                bestValue = i
        return bestValue

    def one_hot_encoding(X, Y, method):
        for feature in range(X.shape[1]):
            threshold = DecisionTreeClassifierbest_Value(X, Y, feature, method)
            X[:, feature] = (X[:, feature] > threshold)

    def quintile_threshold(X):
        X_sorted = np.sort(X)
        quintiles = np.linspace(0, 100, 10)[1:-1]
        values = np.ones(len(quintiles))
        a = 0
        for i in quintiles:
            values[a] = np.round(np.percentile(X_sorted, i), 2)
            a = a + 1
        return values

    def unique_threshold(X):
        X_sorted = np.sort(X)
        X_uniqued = np.unique(X_sorted)
        thresholds = (X_uniqued[1:] + X_uniqued[:-1]) / 2
        return thresholds

    @staticmethod
    def info_gain(X, Y, feature, threshold):
        if len(Y) <= 1: return 0
        base_impurity = DecisionTreeClassifier.compute_impurity(Y)
        feature_values = X[:, feature]

        mask_left = feature_values > threshold
        mask_right = feature_values <= threshold

        X_left = X[mask_left]
        X_right = X[mask_right]
        Y_left = Y[mask_left]
        Y_right = Y[mask_right]

        p_left = DecisionTreeClassifier.compute_impurity(Y_left)
        p_right = DecisionTreeClassifier.compute_impurity(Y_right)

        w_left = len(Y_left) / len(Y)
        w_right = len(Y_right) / len(Y)

        gain = base_impurity - (p_left * w_left + p_right * w_right)

        return [gain, X_left, X_right, Y_left, Y_right]


    def fit(self, X, Y):
        self.root = Node()

        self.root = self.splitTree(
            self.root,
            X,
            Y,
            depth=0
        )

        return self

    def splitTree(self, node, X, Y, depth):

        if len(Y) < self.min_samples_split:
            node.value = np.bincount(Y).argmax()
            return node

        if self.compute_impurity(Y) == 0:
            node.value = np.bincount(Y).argmax()
            return node

        if depth >= self.max_depth:
            node.value = np.bincount(Y).argmax()
            return node


        bestFauture = None
        bestThreshold = None

        node.left = self.splitTree(
            Node(),
            X_left,
            Y_left,
            depth + 1
        )

        node.right = self.splitTree(
            Node(),
            X_right,
            Y_right,
            depth + 1
        )

        return node