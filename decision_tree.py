import math
from typing import NamedTuple

# -------------------------------
# 트리 노드 구조 정의
# -------------------------------


class Node:
    """Base class for tree nodes"""

    pass


class InnerNode(Node):
    """내부 노드: 분기 기준 포함"""

    def __init__(self, feature_idx: int, split_value: float, left: Node, right: Node) -> None:
        self.feature_idx = feature_idx
        self.split_value = split_value
        self.left = left
        self.right = right


class LeafNode(Node):
    """리프 노드: 클래스 라벨 포함"""

    def __init__(self, label: int) -> None:
        self.label = label


class BestSplit(NamedTuple):
    ig: float = -math.inf
    feature_idx: int = 0
    split_value: float = 0.0
    left_indices: list[int] = []
    right_indices: list[int] = []


# -------------------------------
# 결정 트리 구현
# -------------------------------


class DecisionTreeClassifier:
    def __init__(self):
        self.max_depth = 10

    def fit(self, x: list[list[float]], y: list[int]):
        """트리 학습 시작"""
        self.root = self._build_node(x, y, depth=0)

    def _build_node(self, x: list[list[float]], y: list[int], depth: int) -> Node:
        n_samples = len(x)
        if n_samples > 2 and depth < self.max_depth:
            best_split = self._get_best_split(x, y)
            if best_split.ig > 0:
                left_x = [x[i] for i in best_split.left_indices]
                left_y = [y[i] for i in best_split.left_indices]
                right_x = [x[i] for i in best_split.right_indices]
                right_y = [y[i] for i in best_split.right_indices]
                left_child = self._build_node(left_x, left_y, depth + 1)
                right_child = self._build_node(right_x, right_y, depth + 1)
                return InnerNode(best_split.feature_idx, best_split.split_value, left_child, right_child)
        return LeafNode(label=max(sorted(y), key=y.count))

    def _get_best_split(self, x: list[list[float]], y: list[int]) -> BestSplit:
        n_samples = len(x)
        n_features = len(x[0])
        best_split = BestSplit()

        for feature in range(n_features):
            feature_values = [row[feature] for row in x]
            for split_value in feature_values:
                left_indices = [i for i in range(n_samples) if x[i][feature] <= split_value]
                right_indices = [i for i in range(n_samples) if x[i][feature] > split_value]

                if left_indices and right_indices:
                    left_y = [y[i] for i in left_indices]
                    right_y = [y[i] for i in right_indices]
                    ig = self._calculate_ig(y, left_y, right_y)

                    if ig > best_split.ig:
                        best_split = BestSplit(ig, feature, split_value, left_indices, right_indices)
        return best_split

    @staticmethod
    def _calculate_entropy(y: list[int]) -> float:
        total = len(y)
        if total == 0:
            return 0.0
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def _calculate_ig(y_parent: list[int], y_left: list[int], y_right: list[int]) -> float:
        n = len(y_parent)
        l = len(y_left)
        r = len(y_right)
        e_parent = DecisionTreeClassifier._calculate_entropy(y_parent)
        e_left = DecisionTreeClassifier._calculate_entropy(y_left)
        e_right = DecisionTreeClassifier._calculate_entropy(y_right)
        return e_parent - ((l / n) * e_left + (r / n) * e_right)

    def predict(self, x: list[list[float]]) -> list[int]:
        """
        입력 샘플 x에 대해 트리를 따라가서 리프 노드의 라벨을 예측
        """
        preds = []

        for sample in x:
            node = self.root
            while isinstance(node, InnerNode):
                if sample[node.feature_idx] <= node.split_value:
                    node = node.left
                else:
                    node = node.right
            preds.append(node.label)

        return preds


# -------------------------------
# 테스트 케이스
# -------------------------------


def solution(x_train: list[list[float]], y_train: list[int], x_test: list[list[float]]) -> list[int]:
    tree = DecisionTreeClassifier()
    tree.fit(x_train, y_train)
    return tree.predict(x_test)


if __name__ == "__main__":
    # 아주 간단한 1D 데이터 예시
    x_train = [[2.0], [3.0], [4.5], [6.0], [7.0], [8.5]]  # label 0  # label 1
    y_train = [0, 0, 0, 1, 1, 1]

    x_test = [[1.0], [4.0], [5.0], [7.5]]

    y_pred = solution(x_train, y_train, x_test)
    print("✅ 예측 결과:", y_pred)  # 예상 출력: [0, 0, 1, 1]
