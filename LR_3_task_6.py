import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.pipeline import make_pipeline

# Завантаження даних
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Функція для побудови кривих навчання
def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    train_errors = -train_scores.mean(axis=1)
    test_errors = -test_scores.mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_errors, "r-+", label="Навчальні дані")
    plt.plot(train_sizes, test_errors, "b-", label="Тестові дані")
    plt.title(title)
    plt.xlabel("Розмір навчального набору")
    plt.ylabel("Помилка")
    plt.legend()
    plt.grid()
    plt.show()

# Розділення даних
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Лінійна регресія
linear_model = LinearRegression()
plot_learning_curve(linear_model, X_train, y_train, "Криві навчання для лінійної регресії")

# Поліноміальна регресія 2-го ступеня
polynomial_2_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
plot_learning_curve(polynomial_2_model, X_train, y_train, "Криві навчання для поліноміальної регресії (2-го ступеня)")

# Поліноміальна регресія 10-го ступеня
polynomial_10_model = make_pipeline(PolynomialFeatures(degree=10), LinearRegression())
plot_learning_curve(polynomial_10_model, X_train, y_train, "Криві навчання для поліноміальної регресії (10-го ступеня)")
