def big_o(n_time, algo='NN', d=1, k=1, nodes=None, t=None):
    """
    Calculates the expected computation effort compared to n_time

    :param n_time: Calculation time for baseline n
    :param algo: algorithm to calculate computation effort for
    :param d: algorithm parameter
    :param k: algorithm parameter
    :param i: algorithm parameter
    :param nodes: list of nodes (ie. [a, b, c, d] for a 4 layer network)
    :param t: training examples
    :return: Calculation time extrapolated to parameters that will be used
    """

    if algo == 'KMeans':
        return n_time ** (d * k + 1)
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


def big_o_inv(time, algo='KMeans', d=1, k=1):
    """
    Calculates baseline n from a sample training with parameters used

    :param time: Actual calculation time using given parameters
    :param algo: algorithm used
    :param d: algorithm parameter
    :param k: algorithm parameter
    :param i: algorithm parameter
    :return: Baseline n calculation time
    """
    if algo == 'KMeans':
        return time ** -(d * k + 1)
    elif algo == 'SVC':
        return None
    elif algo == 'RandomForestClassifier':
        return None
    elif algo == 'RandomForestRegressor':
        return None
    else:
        return None
