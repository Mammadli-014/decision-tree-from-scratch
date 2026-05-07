import numpy as np
from scipy import stats as st

class Node:
    def __init__(self, feature=None, left=None, right=None, value=None):
        self.feature = feature
        self.left = left
        self.right = right
        self.value = value

def compute_impurity(Y):
    impurity=.0
    if(len(Y) == 0) : return 0
    if(len(np.unique(Y)) == 1) : return 0

    p1= len(Y[Y==1]) / len(Y)
    p2=1-p1
    impurity = np.log2(p1) * (-p1) - np.log2(p2) * p2
    return impurity


def info_gain(X, Y, feature_no):
    base_impurity = compute_impurity(Y)

    mask_left = X[:, feature_no] == 1
    mask_right = X[:, feature_no] == 0
    Y_left = Y[mask_left]
    Y_right = Y[mask_right]

    p_left = compute_impurity(Y_left)
    p_right = compute_impurity(Y_right)
    w_left = len(Y_left) / len(Y)
    w_right = len(Y_right) / len(Y)

    return base_impurity - (p_left * w_left + p_right * w_right)


def recursive_create_tree(X, Y, depth):
    if (depth == 0 or len(np.unique(Y)) == 1): return Node(value=st.mode(Y[0]))

    maxInfoGain = .0
    bestFeature = -1

    for i in range(X.shape[1]):
        temp = info_gain(X, Y, i)
        if (temp > maxInfoGain):
            bestFeature = i
            maxInfoGain = temp
    if (bestFeature == -1): return Node(value=st.mode(Y[0]))

    mask_left = X[:, bestFeature] == 1
    mask_right = X[:, bestFeature] == 0
    X_left = X[mask_left]
    X_right = X[mask_right]
    Y_left = Y[mask_left]
    Y_right = Y[mask_right]

    left = recursive_create_tree(X_left, Y_left, depth - 1)
    right = recursive_create_tree(X_right, Y_right, depth - 1)
    return Node(bestFeature, left, right)

def build_tree(X, Y, depth = 4):
    return recursive_create_tree(X, Y, depth)
