from sklearn.cluster import KMeans
import numpy as np
import time
import scipy.stats as sps

from scitime import Estimator

# example for kmeans clustering
estimator = Estimator(meta_algo='RF', verbose=3)
km = KMeans()

# generating inputs for this example
X = np.random.rand(100000, 10)
# run the estimation
estimation, lower_bound, upper_bound = estimator.time(km, X)

n = 30
tlst = []


for _ in range(n):
    t0 = time.time()
    km.fit(X)
    tlst.append(time.time()-t0)

print('Average time: {!s}s'.format(round(np.average(tlst))))

print(sps.describe(tlst))
y = km.predict(X)

print(y)
