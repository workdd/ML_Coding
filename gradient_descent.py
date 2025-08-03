import random

def random_bias_theta(n_features: int) -> tuple[float, list[float]]:
    """
    랜덤하게 bias와 weight(theta)를 초기화
    """
    b = random.random()
    theta = [random.random() for _ in range(n_features)]
    return b, theta


def calculate_y_hat(b: float, theta: list[float], x: list[float]) -> float:
    """
    예측값 y_hat 계산: y_hat = b + Σ(theta_i * x_i)
    """
    y_hat = 0
    for idx in range(len(x)):
        y_hat += theta[idx] * x[idx]  # ✅ 누적 곱셈
    y_hat += b
    return y_hat


def calculate_d_b(Y: list[float], Y_hat: list[float]) -> float:
    """
    bias에 대한 gradient 계산 (∂L/∂b)
    """
    d_b = 0
    for idx in range(len(Y)):
        d_b += Y_hat[idx] - Y[idx]  # 오차 누적
    d_b = d_b / len(Y)  # 평균 오차
    return d_b


def calculate_d_theta(X: list[list[float]], Y: list[float], Y_hat: list[float]) -> list[float]:
    """
    각 theta_j에 대한 gradient 계산 (∂L/∂θ_j)
    """
    feature_num = len(X[0])
    sample_num = len(Y)

    d_theta_list = []

    for f_idx in range(feature_num):
        d_theta = 0.0
        for i in range(sample_num):
            error = Y_hat[i] - Y[i]
            d_theta += error * X[i][f_idx]
        d_theta /= sample_num  # 평균
        d_theta_list.append(d_theta)

    return d_theta_list


def update(X: list[list[float]], Y: list[float], Y_hat: list[float], b_prev: float, theta_prev: list[float], learning_rate: float) -> tuple[float, list[float]]:
    """
    경사 하강법으로 파라미터 업데이트
    """
    d_theta = calculate_d_theta(X, Y, Y_hat)
    d_b = calculate_d_b(Y, Y_hat)

    b_new = b_prev - learning_rate * d_b
    theta_new = []
    for k in range(len(theta_prev)):
        theta_val = theta_prev[k] - learning_rate * d_theta[k]
        theta_new.append(theta_val)

    return b_new, theta_new


def fit(X: list[list[float]], Y: list[float], num_iterations: int, learning_rate: float = 0.2) -> tuple[float, list[float]]:
    """
    학습을 위한 전체 흐름 구성
    """
    b, theta = random_bias_theta(len(X[0]))
    for _ in range(num_iterations):
        Y_hat = []
        for x in X:
            Y_hat.append(calculate_y_hat(b, theta, x))
        b, theta = update(X, Y, Y_hat, b, theta, learning_rate)

    return b, theta


def solution(x_train: list[list[float]], y_train: list[float], x_test: list[list[float]], iterations: int = 1000) -> list[float]:
    random.seed(42)  # 결과 고정
    b, theta = fit(x_train, y_train, iterations)
    return [round(calculate_y_hat(b, theta, x), 2) for x in x_test]

if __name__ == "__main__":
    print("테스트 케이스 1 — y = 2x")
    x_train1 = [[1], [2], [3], [4]]
    y_train1 = [2, 4, 6, 8]
    x_test1 = [[5], [6]]
    print("예측:", solution(x_train1, y_train1, x_test1))  # [10.0, 12.0]

    print("\n테스트 케이스 2 — y = x1 + 2x2")
    x_train2 = [[1, 1], [2, 2], [3, 3]]
    y_train2 = [3, 6, 9]
    x_test2 = [[4, 1], [1, 4]]
    print("예측:", solution(x_train2, y_train2, x_test2))  # 근사적으로 [6.0, 9.0]

    print("\n테스트 케이스 3 — 상수값 y = 5")
    x_train3 = [[0], [0], [0]]
    y_train3 = [5, 5, 5]
    x_test3 = [[0], [0]]
    print("예측:", solution(x_train3, y_train3, x_test3))  # [5.0, 5.0]

    print("\n테스트 케이스 4 — y = 3x1 - x2 + 1")
    x_train4 = [[1, 2], [2, 1], [3, 3]]
    y_train4 = [3*1 - 2 + 1, 3*2 - 1 + 1, 3*3 - 3 + 1]  # [2, 6, 7]
    x_test4 = [[4, 2], [0, 5]]
    print("예측:", solution(x_train4, y_train4, x_test4))  # [~9, ~-4]
