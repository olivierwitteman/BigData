def big_o(n_time, algo='KMeans', d=1, k=1, i=1):
    """
    Calculates the expected computation effort compared to n_time

    :param n_time: Calculation time for baseline n
    :param algo: algorithm to calculate computation effort for
    :param d: algorithm parameter
    :param k: algorithm parameter
    :param i: algorithm parameter
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
    else:
        return None


def big_o_inv(time, algo='KMeans', d=1, k=1, i=1):
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
