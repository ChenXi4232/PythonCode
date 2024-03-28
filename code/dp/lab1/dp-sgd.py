import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.optimize import root_scalar
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RANDOM_STATE = 1


class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, num_iterations=100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.tau = 1e-6  # small value to prevent log(0)
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        # np.clip(z, -500, 500) limits the range of z to avoid extremely
        # large or small values that could lead to overflow.
        return 1 / (1 + np.exp(-np.clip(z, -700, 700)))

    def fit(self, X, y):
        # Initialize weights and bias
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0.0

        # Gradient descent optimization
        for _ in range(self.num_iterations):
            # Compute predictions of the model
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            # Compute loss and gradients
            '''loss = -np.mean(
                y * np.log(predictions + self.tau)
                + (1 - y) * np.log(1 - predictions + self.tau)
            )'''
            # dz = predictions - y
            # cross entropy
            '''dz = -(y * np.log(predictions + self.tau) +
                   (1 - y) * np.log(1 - predictions + self.tau))'''
            dz = -(y / (predictions + self.tau) -
                   (1 - y) / (1 - predictions + self.tau))
            dw = np.dot(X.T, dz) / num_samples
            db = np.sum(dz) / num_samples

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def dp_fit(self, X, y, epsilon, delta, C=1):
        # Initialize weights and bias
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # TODO: Calculate epsilon_u, delta_u based epsilon, delta and epochs here.
        epsilon_u, delta_u = compute_DP_args(
            epsilon, delta, self.num_iterations)
        print("epsilon_u:", epsilon_u)
        if epsilon_u >= 1 or epsilon_u <= 0:
            print("epsilon_u is not valid.")
            return
        # Gradient descent optimization
        for _ in range(self.num_iterations):
            # Compute predictions of the model
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            # Compute loss and gradients
            '''loss = -np.mean(y * np.log(predictions) + (1 - y)
                            * np.log(1 - predictions))'''
            # dz = predictions - y
            # cross entropy
            '''dz = -(y * np.log(predictions + self.tau) +
                   (1 - y) * np.log(1 - predictions + self.tau))'''
            dz = -(y / (predictions + self.tau) -
                   (1 - y) / (1 - predictions + self.tau))

            # TODO: Clip gradient here.
            clip_dz = clip_gradients(dz, C)
            # Add noise to gradients
            # TODO: Calculate epsilon_u, delta_u based epsilon, delta and epochs here.
            noisy_dz = add_gaussian_noise_to_gradients(
                clip_dz, epsilon_u, delta_u, C)
            dw = np.dot(X.T, noisy_dz) / num_samples
            db = np.sum(noisy_dz) / num_samples

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_probability(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(linear_model)
        return probabilities

    def predict(self, X):
        probabilities = self.predict_probability(X)
        # Convert probabilities to classes
        return np.round(probabilities)


def get_train_data(dataset_name=None):
    if dataset_name is None:
        # Generate simulated data
        # np.random.seed(RANDOM_STATE)
        X, y = make_classification(
            n_samples=100000, n_features=20, n_classes=2, random_state=RANDOM_STATE
        )
    elif dataset_name == "cancer":
        # Load the breast cancer dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
    else:
        raise ValueError("Not supported dataset_name.")

    # Standardize features
    X = (X - np.mean(X, axis=0)) / X.std(axis=0)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def clip_gradients(gradients, C):
    # TODO: Clip gradients.
    # L2 范数
    L2_norm = np.linalg.norm(gradients, ord=2)
    # 算法中的梯度裁剪公式
    if L2_norm > C:
        # 如果梯度的 L2 范数大于 C，则进行裁剪
        return C * (gradients / L2_norm)
    else:
        # 希望保留小于 C 的梯度
        return gradients


def add_gaussian_noise_to_gradients(gradients, epsilon, delta, C):
    # TODO: add gaussian noise to gradients.
    # 利用 C 进行裁剪，对于每一行数据输出结果，其敏感度为 C，因此噪声的方差为 2C^2 * ln(1.25/delta) / epsilon^2
    # 计算噪声的方差
    variance = 2 * C**2 * np.log(1.25/delta) / epsilon**2
    # 生成噪声
    noisy_gradients = gradients + \
        np.random.normal(0, np.sqrt(variance), gradients.shape)
    return noisy_gradients


def compute_DP_args(total_epsilon, total_delta, num_iterations=100):
    # 利用高级组合性求解每次迭代应该满足的差分隐私参数
    single_delta = total_delta / (num_iterations + 1)
    x = sp.symbols("x")
    # 高级组合性计算表达式
    expr = sp.sqrt(2*num_iterations*sp.log(1/single_delta)) * \
        x+num_iterations*x*(sp.exp(x)-1) - total_epsilon
    # 公式简化
    lam_f = sp.lambdify(x, sp.simplify(expr))
    # 求解数值解
    sol = root_scalar(lam_f, bracket=(0, 500))
    # solutions = sp.nsolve(expr, x, 0)
    return sol.root, single_delta


if __name__ == "__main__":
    # Prepare datasets.
    dataset_name = "cancer"
    X_train, X_test, y_train, y_test = get_train_data(dataset_name)

    # Training the normal model
    normal_model = LogisticRegressionCustom(
        learning_rate=0.01, num_iterations=10000)
    normal_model.fit(X_train, y_train)
    y_pred = normal_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Normal accuracy:", accuracy)

    # Training the differentially private model
    '''dp_model = [LogisticRegressionCustom(
        learning_rate=0.01, num_iterations=i) for i in [
        100, 200, 500, 1000, 2000, 5000, 10000]]'''
    dp_model = LogisticRegressionCustom(
        learning_rate=0.01, num_iterations=10000)
    epsilon = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    # epsilon = [0.1, 1, 10, 100, 1000, 10000, 100000, 1e6, 1e7, 1e8, 1e9]
    delta = 1e-3
    accs = []
    '''for i in range(len(dp_model)):
        print("num_iterations:", dp_model[i].num_iterations)
        dp_model[i].dp_fit(X_train, y_train, epsilon=epsilon,
                           delta=delta, C=1)
        y_pred = dp_model[i].predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("DP accuracy:", accuracy)
        accs.append(accuracy)'''
    for eps in epsilon:
        print("epsilon:", eps)
        dp_model.dp_fit(X_train, y_train, epsilon=eps, delta=delta, C=1)
        y_pred = dp_model.predict(X_test)
        '''if len(set(y_pred)) == 1:
            # 如果结果全0或全1，则认为噪声太大。结果不可信，准确率为0
            accuracy = 0
        else:'''
        accuracy = accuracy_score(y_test, y_pred)
        print("DP accuracy:", accuracy)
        accs.append(accuracy)
    # 作图观察隐私预算与准确率的关系
    plt.title('accuracy-epsilon')
    plt.xlabel("epsilon")
    plt.ylabel("accuracy")
    # plt.xticks(epsilon)
    plt.plot(epsilon, accs)
    plt.show()
