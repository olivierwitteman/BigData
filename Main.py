from sklearn.cluster import KMeans
import numpy as np

from scitime import Estimator

# example for kmeans clustering
estimator = Estimator(meta_algo='RF', verbose=3)
km = KMeans()

# generating inputs for this example
X = np.random.rand(100000, 10)
# run the estimation
estimation, lower_bound, upper_bound = estimator.time(km, X)
