from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
import numpy as np
import time
import scipy.stats as sps

from scitime import Estimator

# example for kmeans clustering
estimator = Estimator(meta_algo='RF', verbose=3)
nn = MLPRegressor(hidden_layer_sizes=(1, 1))

# generating inputs for this example
# X = np.random.rand(100000, 10)

X, y = make_regression(n_samples=int(5e4), n_features=8, random_state=0)

# run the estimation

n = 30
tlst = []


for _ in range(n):
    t0 = time.time()
    nn.fit(X, y)
    tlst.append(time.time()-t0)

    print('\r{!s}'.format(sps.describe(tlst)), end='')

print('Average time: {!s}s'.format(round(np.average(tlst))))


y = nn.predict(X)

print(y.shape)
