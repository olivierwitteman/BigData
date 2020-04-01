import numpy as np


def big_o(n_time, algo='NN', nodes=None, t=None):
    """
    Calculates the expected computation effort compared to n_time

    :param n_time: Calculation time for baseline n
    :param algo: algorithm to calculate computation effort for
    :param nodes: list of node sizes (ie. [a, b, c, d] for a 4 layer network)
    :param t: training examples
    :return: Calculation time extrapolated to parameters that will be used
    """

    if algo == 'KMeans':
        # return n_time ** (d * k + 1)
        return None
    elif algo == 'SVC':
        return None
    elif algo == 'RandomForestClassifier':
        return None
    elif algo == 'RandomForestRegressor':
        return None
    elif algo == 'NN':
        # https://ai.stackexchange.com/questions/5728/what-is-the-time-complexity-for-training-a-neural-network-using-back-propagation
        nodecomplexity = 0
        for q in range(len(nodes) - 1):
            nodecomplexity += nodes[q] * nodes[q + 1]
        return n_time * t * nodecomplexity


def big_o_inv(time, algo='RandomForestRegressor', n=1, t=1):
    """
    Calculates baseline n from a sample training with parameters used

    :param time: Actual calculation time using given parameters
    :param algo: algorithm used
    :param n: algorithm parameter sample size
    :param t: algorithm parameter trees
    :return: Baseline n calculation time
    """
    if algo == 'KMeans':
        # return time ** -(d * k + 1)
        return None
    elif algo == 'SVC':
        return None
    elif algo == 'RandomForestClassifier':
        return None
    elif algo == 'RandomForestRegressor':
        return np.sqrt(time / (n * t))
    else:
        return None
