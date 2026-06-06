import numpy as np
from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions
from mlxtend.classifier import Adaline
import matplotlib.pyplot as plt

# Loading and printing iris data
iris, y = iris_data()
print('Dimensions: %s x %s' % (iris.shape[0], iris.shape[1]))
print('\nHeader: %s' % ['sepal length', 'sepal width', 'petal length', 'petal width'])
print('1st row', iris[0])
print('Classes: Setosa, Versicolor, Virginica')
print('Class distribution: %s' % np.bincount(y))
X = iris[:, [0, 3]] # choose features `sepal length' and `petal width'
X = X[0:100] # only classes 0 and 1
y = y[0:100] # only classes 0 and 1

# standardize the data
X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

# ADALINE neural network
ada = Adaline(epochs=30, eta=0.01, minibatches=None,  random_seed=1)
ada.fit(X, y)
plot_decision_regions(X, y, clf=ada)
plt.title('Adaline for iris data')
plt.show()