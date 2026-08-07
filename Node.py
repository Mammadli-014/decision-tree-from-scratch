import numpy as np


class Node:
    thresholds = []

    def __init__(self, feature=None, left=None, right=None, value=None, threshold=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def fillThreshold(self, X, Y, method):
        m = X.shape[1]
        Node.thresholds = [None] * m
        for i in range(m):
            bestValue = best_Value(X, Y, i, method)
            Node.thresholds[i] = bestValue

    def createNode(self, left=None, right=None, threshold=None):
        return Node(left=None, right=None, threshold=None)


def compute_impurity(Y):
    impurity = .0
    if (len(Y) == 0): return 0
    if (len(np.unique(Y)) == 1): return 0

    p1 = len(Y[Y == 1]) / len(Y)
    p2 = 1 - p1
    impurity = np.log2(p1) * (-p1) - np.log2(p2) * p2
    return impurity


def one_hot_encoding(X, Y, method):
    for feature in range(X.shape[1]):
        threshold = best_Value(X, Y, feature, method)
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


def info_gain(X, Y, feature_no, threshold):
    if len(Y) <= 1: return 0
    base_impurity = compute_impurity(Y)
    feature_values = X[:, feature_no]

    mask_left = feature_values > threshold
    mask_right = feature_values < threshold

    X_left = X[mask_left]
    X_right = X[mask_right]
    Y_left = Y[mask_left]
    Y_right = Y[mask_right]

    p_left = compute_impurity(Y_left)
    p_right = compute_impurity(Y_right)

    w_left = len(Y_left) / len(Y)
    w_right = len(Y_right) / len(Y)

    return [base_impurity - (p_left * w_left + p_right * w_right), X_left, X_right, Y_left, Y_right]

def best_Value(X, Y, feature_no, method):
    values = method(X[:, feature_no])
    maxGain = 0
    bestValue = -1
    for i in values:
        if (info_gain(X, Y, feature_no, i) > maxGain):
            maxGain = info_gain(X, Y, feature_no, i)
            bestValue = i
    return bestValue


def splitTree(node, X, Y, dept):
    if (len(Y) < 10): return None
    if (compute_impurity(Y) == 0): return node
    if (dept >= Node.maxDept): return node
    if (node is None): node = Node()

    threshold = 0
    bestGain = [0]
    for i in range(len(Node.thresholds)):
        gain = info_gain(X, Y, i, Node.thresholds[i])
        if (bestGain[0] < gain[0]):
            bestGain = gain
            threshold = i
    node.threshold = threshold

    node.left = splitTree(Node(), bestGain[1], bestGain[3], dept + 1)
    node.right = splitTree(Node(), bestGain[2], bestGain[4], dept + 1)

    return node
