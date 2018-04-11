# Importing regression modules
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn import svm
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.preprocessing import StandardScaler

import pandas as pd


df = pd.read_csv('data.csv')

X, y = df.drop("current_result", axis = 1).values, df[["current_result"]].values

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True)

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

kf = KFold(n_splits=10, shuffle=True)

array_of_accuracies = []
array_of_balances = []

# Units in satoshis
start = 1
step = 1
end = 150


for i in range(start, end, step):

    if ((end - i) % 100 == 0):
        print((end - i) / (end - start) * 100, "%")

    # print("TESTING FOR multiplier =", 1 + i / 100)
    m = KNeighborsRegressor(n_neighbors=i)
    # m = linear_model.RANSACRegressor()
    # m = KernelRidge(alpha=1.0)
    # m = linear_model.Ridge(alpha=0.5)    # best one so far
    # m = svm.SVR()
    # m = linear_model.RANSACRegressor()
    # m = linear_model.ElasticNet()
    # m = linear_model.Lasso()

    accuracies = []
    balance_array = []
    y_pred = []
    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        m.fit(X_train, y_train)
#         y_pred = np.full(len(y_test), 1 + i / 100)
        y_pred = m.predict(X_test)
        count = 0
        balance = 100000.00
        for k in range(len(y_test)):
            if y_pred[k] <= y_test[k] and y_pred[k] >= 1:
                # print(y_test[k])
                # print( 1 + i / 100)
                count += 1
                balance += float(y_pred[k]) * 100 - 100
            else:
                balance -= 100

        accuracy = count / len(y_pred)
        accuracies = np.append(accuracies, accuracy)
        balance_array = np.append(balance_array, balance)

    average_ending_balance = np.mean(balance_array)
    average_accuracy = np.mean(accuracies)

    # print("Multiplier: ", 1 + i / 100)
    # print("Ending balance: ", average_ending_balance)
    # print("Average Accuracy: ", average_accuracy)
    # print()

    # print(len(array_of_accuracies))
    array_of_accuracies = np.append(array_of_accuracies, average_accuracy)
    array_of_balances = np.append(array_of_balances, average_ending_balance)

# numbers_list = list(frange(1 + start / 100, 1 + end / 100 - 0.01, round(step / 100, 2)))
# numbers_array = [round(elem, 2) for elem in numbers_list]
numbers_list = list(range(start, end, step))
numbers_array = [elem for elem in numbers_list]
# print(array_of_accuracies)
# print(numbers_array)
plt.plot(numbers_array, array_of_accuracies)
plt.title("Accuracy")
plt.show()
plt.plot(numbers_array, array_of_balances)
plt.plot(numbers_array, np.full(int((end - start) / step), 100000))
plt.title("Balance")
plt.show()

print(np.max(array_of_balances))

print("Done")