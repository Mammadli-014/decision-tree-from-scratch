import numpy as np
from Node import Node

class MyTree:

    thresholds=[]
    def __init__(self, max_depth=10, min_samples_split=10,min_gain=0.01):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.root = None

    def fillThreshold(self, X, Y, method):
        m = X.shape[1]
        self.thresholds = np.empty(m,dtype=int)
        for i in range(m):
            bestValue = MyTree.best_Value(X, Y, i, method)
            self.thresholds[i] = bestValue

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
            gain = MyTree.info_gain(X, Y, feature_no, i)
            if (gain[0] > maxGain):
                maxGain =gain[0]
                bestValue = i
        return bestValue

    @staticmethod
    def one_hot_encoding(X, Y, method):
        for feature in range(X.shape[1]):
            threshold = MyTree.best_Value(X, Y, feature, method)
            X[:, feature] = (X[:, feature] > threshold)

    @staticmethod
    def quintile_threshold(X):
        X_sorted = np.sort(X)
        quintiles = np.linspace(0, 100, 10)[1:-1]
        values = np.ones(len(quintiles))
        a = 0
        for i in quintiles:
            values[a] = np.round(np.percentile(X_sorted, i), 2)
            a = a + 1
        return values

    @staticmethod
    def unique_threshold(X):
        X_sorted = np.sort(X)
        X_uniqued = np.unique(X_sorted)
        thresholds = (X_uniqued[1:] + X_uniqued[:-1]) / 2
        return thresholds

    @staticmethod
    def info_gain(X, Y, feature, threshold):
        if len(Y) <= 1: return 0
        base_impurity = MyTree.compute_impurity(Y)
        feature_values = X[:, feature]

        mask_left = feature_values <= threshold
        mask_right = feature_values > threshold

        X_left = X[mask_left]
        X_right = X[mask_right]
        Y_left = Y[mask_left]
        Y_right = Y[mask_right]

        p_left = MyTree.compute_impurity(Y_left)
        p_right = MyTree.compute_impurity(Y_right)

        w_left = len(Y_left) / len(Y)
        w_right = len(Y_right) / len(Y)

        gain = base_impurity - (p_left * w_left + p_right * w_right)

        return [gain, X_left, X_right, Y_left, Y_right]

    def splitTree(self,node, X, Y, dept):
        if (node is None): node = Node()

        node.dept = dept

        if (len(Y) < self.min_samples_split):
            node.value = np.bincount(Y).argmax()
            return node

        if (MyTree.compute_impurity(Y) == 0):
            node.value = np.bincount(Y).argmax()
            return node

        if (node.dept >= self.max_depth):
            node.value = np.bincount(Y).argmax()
            return node

        feature = None
        bestGain = [0]

        for i in range(len(self.thresholds)):
            gain = MyTree.info_gain(X, Y, i, self.thresholds[i])
            if (bestGain[0] < gain[0]):
                bestGain = gain
                feature = i

        if (bestGain[0] < self.min_gain):
            node.value = np.bincount(Y).argmax()
            return node

        node.feature = feature
        node.threshold = self.thresholds[feature]
        node.dept = dept

        print(
            "Depth:", node.dept,
            "Samples:", len(Y),
            "Feature:", node.feature,
            "Gain:", "{:.4f}".format(bestGain[0]),
            "Threshold:", "{:.2f}".format(node.threshold)
        )

        node.left = self.splitTree(Node(), bestGain[1], bestGain[3], dept + 1)
        node.right = self.splitTree(Node(), bestGain[2], bestGain[4], dept + 1)

        return node


    def fit(self, X, Y):
        self.fillThreshold(X, Y, method = self.unique_threshold)
        self.root = Node()
        self.splitTree(self.root,X,Y,dept=0)

        return self


    def predict(self,X):

        prediction = np.empty(len(X),dtype=int)
        indices = np.arange(len(X))

        self._predict_recursive(self.root,X, indices,prediction)

        return prediction


    def _predict_recursive(self,node,X,indices,prediction):
        if node.value is not None:
            prediction[indices] = node.value
            return
        mask = X[indices,node.feature] > node.threshold

        right=indices[mask]
        left=indices[~mask]

        self._predict_recursive(node.left, X, left, prediction)
        self._predict_recursive(node.right, X, right, prediction)




    def calculateAcc(self,y_pre, y):
        return np.mean(y_pre == y)