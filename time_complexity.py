import numpy as np


def big_o_nn(n_base, m=1, o=1, i=1, nodes=(100, 8), t=1, method='scikit', inv=False):
    """
    Calculates the expected computation effort compared to n_time

    :param n_base: Calculation time for baseline n
    :param algo: algorithm to calculate computation effort for
    :param m: features
    :param o: output neurons
    :param i: iterations
    :param nodes: list of node sizes (ie. [a, b, c, d] for a 4 layer network)
    :param t: training examples
    :param method: method for complexity calculation
    :return: Calculation time extrapolated to parameters that will be used
    """

    nodecomplexity = 0
    for q in range(len(nodes) - 1):
        nodecomplexity += nodes[q] * nodes[q + 1]

    if method == 'stack':
        # https://ai.stackexchange.com/questions/5728/what-is-the-time-complexity-for-training-a-neural-network-using-back-propagation
        if inv:
            return n_base / (t * nodecomplexity)
        else:
            return n_base * t * nodecomplexity
    elif method == 'scikit':
        # https://scikit-learn.org/stable/modules/neural_networks_supervised.html
        if inv:
            return n_base / (t * m * nodecomplexity * o * i)
        else:
            return n_base * t * m * nodecomplexity * o * i * t


def big_o_rfr(n_base, n_tree, m_try, n, inv=False):
    """

    :param n_base:
    :param n_tree:
    :param m_try:
    :param n:
    :return:
    """
    if inv:
        return n_base / (n_tree * m_try * n * np.log(n))
    else:
        return n_base * n_tree * m_try * n * np.log(n)


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


# t_b = big_o_rfr(n_base=6, n_tree=100, m_try=int(1e4), n=int(1e4), inv=True)
# print(t_b)
# print(big_o_nn(n_base=t_b, m=8, o=1, i=30, nodes=(100, 100, 100, 100), t=int(200), inv=False))
