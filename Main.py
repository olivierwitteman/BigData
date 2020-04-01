from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import numpy as np
import time
import scipy.stats as sps

from scitime import Estimator

# example for kmeans clustering
estimator = Estimator(meta_algo='RF', verbose=3)
rfr = RandomForestRegressor()

# generating inputs for this example
# X = np.random.rand(100000, 10)

X, y = make_regression(n_samples=int(5e4), n_features=8, random_state=0)

# run the estimation
estimation, lower_bound, upper_bound = estimator.time(rfr, X, y=y)

n = 30
tlst = []


for _ in range(n):
    t0 = time.time()
    rfr.fit(X, y)
    tlst.append(time.time()-t0)

print('Average time: {!s}s'.format(round(np.average(tlst))))

print(sps.describe(tlst))
y = rfr.predict(X)

print(y.shape)
