import math
from typing import List


class TreeNode:
    def __init__(self, feature_idx=None, split_value=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.split_value = split_value
        self.left = left
        self.right = right
        self.value = value  # only set for leaf nodes


class SimpleDecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, x: List[List[float]], residual: List[float]):
        self.root = self._build_tree(x, residual, depth=0)

    def _build_tree(self, x, y, depth):
        if len(y) < self.min_samples_split or depth >= self.max_depth:
            value = sum(y) / len(y)
            return TreeNode(value=value)

        best_feature, best_split, best_left_idx, best_right_idx, best_mse = None, None, None, None, float("inf")
        n_features = len(x[0])

        for feature_idx in range(n_features):
            values = set(row[feature_idx] for row in x)
            for split_value in values:
                left_idx = [i for i in range(len(x)) if x[i][feature_idx] <= split_value]
                right_idx = [i for i in range(len(x)) if x[i][feature_idx] > split_value]
                if not left_idx or not right_idx:
                    continue
                left_y = [y[i] for i in left_idx]
                right_y = [y[i] for i in right_idx]
                mse = self._mse(left_y) * len(left_y) + self._mse(right_y) * len(right_y)
                if mse < best_mse:
                    best_feature = feature_idx
                    best_split = split_value
                    best_left_idx = left_idx
                    best_right_idx = right_idx
                    best_mse = mse

        if best_feature is None:
            return TreeNode(value=sum(y) / len(y))

        left_x = [x[i] for i in best_left_idx]
        left_y = [y[i] for i in best_left_idx]
        right_x = [x[i] for i in best_right_idx]
        right_y = [y[i] for i in best_right_idx]

        left_node = self._build_tree(left_x, left_y, depth + 1)
        right_node = self._build_tree(right_x, right_y, depth + 1)
        return TreeNode(feature_idx=best_feature, split_value=best_split, left=left_node, right=right_node)

    def _mse(self, y):
        mean = sum(y) / len(y)
        return sum((val - mean) ** 2 for val in y) / len(y)

    def predict_one(self, x_row, node=None):
        node = node if node else self.root
        while node.value is None:
            if x_row[node.feature_idx] <= node.split_value:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, x):
        return [self.predict_one(row) for row in x]


class SimpleXGBoost:
    def __init__(self, n_estimators=3, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, x, y):
        y_pred = [0.0] * len(y)
        for _ in range(self.n_estimators):
            residual = [y[i] - y_pred[i] for i in range(len(y))]
            tree = SimpleDecisionTree(max_depth=self.max_depth)
            tree.fit(x, residual)
            update = tree.predict(x)
            y_pred = [y_pred[i] + self.learning_rate * update[i] for i in range(len(y))]
            self.trees.append(tree)

    def predict(self, x):
        y_pred = [0.0] * len(x)
        for tree in self.trees:
            update = tree.predict(x)
            y_pred = [y_pred[i] + self.learning_rate * update[i] for i in range(len(x))]
        return y_pred


# 간단한 테스트 케이스
x_train = [[1], [2], [3], [4], [5]]
y_train = [1, 2, 3, 4, 5]
x_test = [[1.5], [3.5], [5.0]]

model = SimpleXGBoost(n_estimators=3, learning_rate=0.5, max_depth=2)
model.fit(x_train, y_train)

# 테스트 케이스 출력 (pandas 없이 표로 보기)
x_test = [[1.5], [3.5], [5.0]]
preds = model.predict(x_test)

print("x\tpredicted_y")
for x_val, y_pred in zip(x_test, preds):
    print(f"{x_val[0]:.1f}\t{y_pred:.4f}")
