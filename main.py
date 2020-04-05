#%%

__author__ = "Olivier Witteman"
__email__ = "olivier@2001.net"

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
import numpy as np
import time
import scipy.stats as sps
from scitime import Estimator
import time_complexity as tcom

#%%

# General parameters
n = 2

# Dataset parameters
features = 300
set_size = int(1e5)

X, y = make_regression(n_samples=set_size, n_features=features, random_state=0)

# Random forest parameters
max_trees = 100
tries_tree = set_size

# Neural network parameters
output_nodes = 1
iterations = 30
hidden_layers = (8, 100, 100, 100, 100, 1)
batch_size = 200

#%%

estimator = Estimator(meta_algo='RF', verbose=2)
rfr = RandomForestRegressor(n_estimators=max_trees)

# Estimation by scitime
estimation, lower_bound, upper_bound = estimator.time(rfr, X, y=y)

tlst_rf = []

# for _ in range(n):
#     t0 = time.time()
#     rfr.fit(X, y)
#     tlst_rf.append(time.time()-t0)
#
#     print('\r{!s}'.format(sps.describe(tlst_rf)), end='')

#%%

trees = len(rfr.estimators_)
t_b = tcom.big_o_rfr(n_base=np.array([estimation, lower_bound, upper_bound]), n_tree=trees, m_try=tries_tree,
                     n=min(set_size, int(2e4)), inv=True)

nn_est = tcom.big_o_nn(n_base=t_b, m=features, o=output_nodes, i=iterations, nodes=hidden_layers, t=set_size, inv=False)
print('Neural network training time estimate (95% confidence interval):\n\nEstimate:    {!s}\nLower bound: '
      '{!s}\nUpper bound: {!s}'.format(nn_est[0], nn_est[1], nn_est[2]))

#%%

nn = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=iterations, batch_size=batch_size)

tlst_nn = []

for _ in range(n):
    t0 = time.time()
    nn.fit(X, y)
    tlst_nn.append(time.time()-t0)

    print('\r{!s}'.format(sps.describe(tlst_nn)), end='')
