import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Завантаження даних
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Розбиття даних на навчальну і тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Створення моделі лінійної регресії
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Прогноз по тестовій вибірці
y_pred = regr.predict(X_test)

# Розрахунок метрик
print("Коефіцієнти регресії:", regr.coef_)
print("Перехоплення (intercept):", regr.intercept_)
print("R2 score:", r2_score(y_test, y_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))

# Побудова графіка
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0), label='Передбачення')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4, label='Ідеальна лінія')
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
ax.legend()
plt.show()
