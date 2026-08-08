class Node:

    def __init__(
        self,
        left=None,
        right=None,
        value=None,
        feature=None,
        threshold=None
    ):
        self.left = left
        self.right = right
        self.value = value
        self.feature = feature
        self.threshold = threshold