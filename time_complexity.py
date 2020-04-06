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
            return n_base * t * nodecomplexity * o * i # * m


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

