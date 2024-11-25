from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Генерація даних для варіанта 3
m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# Лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred = lin_reg.predict(X)

# Поліноміальна регресія
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)

# Графік
plt.scatter(X, y, color='blue', label='Дані')
plt.plot(X, y_pred, color='green', label='Лінійна регресія')
plt.plot(X, y_poly_pred, color='red', label='Поліноміальна регресія')
plt.legend()
plt.show()
