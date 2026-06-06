import numpy as np

class AdalineGD(object):

    def __init__(self, mu=0.01, epochs=50):
        self.mu = mu
        self.epochs = epochs

    def train(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.epochs):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.mu * X.T.dot(errors)
            self.w_[0] += self.mu * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def activation(self, X):
        return self.net_input(X)
    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

data = pd.read_excel("class-data.xlsx")

# inputs
X = data.iloc[0:1999, [0,1]].values
X = X.astype(float)

# class labels
y = data.iloc[0:1999, 2].values
#y = np.where(y == 1, y, -1)

# standardise data (only X)
Xstd = np.copy(X)
Xstd[:,0] = (X[:,0] - X[:,0].mean())/(X[:,0].std())
Xstd[:,1] = (X[:,1] - X[:,1].mean())/(X[:,1].std())

ada = AdalineGD(epochs=100, mu=0.0001)

ada.train(Xstd, y)
plot_decision_regions(Xstd, y, clf=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('attribute 1 [standardized]')
plt.ylabel('attribute 2 [standardized]')
plt.show()

plt.plot(range(1, len( ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Sum-squared-error')
plt.show()