class Node:

    def __init__(self, left=None, right=None, value = None, dept=None, threshold = None,feature = None,):
        self.threshold=threshold
        self.left = left
        self.right = right
        self.value=value
        self.feature=feature
        self.dept=dept