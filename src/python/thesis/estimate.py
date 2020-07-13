import numpy as np
import matplotlib.pyplot as plt
from common import utils
import operator as op
from functools import reduce
import sympy
import basic_influence_model.choosing_model as influence_model
import time

N = 1000
b = 1
J = 0.5
optimal_value = 1

estimate_cache = {}


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def estimate_iteratively(m, sum_m):
    E = 0
    for i in range(N + 1):
        p_i = 1

        for j in range(i):
            p_i *= (m + j) / (sum_m + j)

        for j in range(i, N, 1):
            p_i *= (1 - ((m + i) / (sum_m + j)))

        p_i *= ncr(N, i)
        E += (i * p_i)
    return E


def estimate_prob_recursively(m, sum_m, k, j, n):
    if (j, n) in estimate_cache:
        return estimate_cache[(j, n)]
    if n == 0:
        if j == k:
            value = 1
        else:
            value = 0
    else:
        q = (m + j) / (sum_m + N - n)
        value = q * estimate_prob_recursively(m, sum_m, k, j + 1, n - 1) + (1 - q) * estimate_prob_recursively(m, sum_m, k, j, n - 1)
    estimate_cache[(j, n)] = value
    return value


def estimate_recursively(m, sum_m):
    E = 0
    for i in range(N + 1):
        p_i = estimate_prob_recursively(m, sum_m, i, 0, N)
        E += (i * p_i)
        estimate_cache.clear()
    return E


def get_population_estimations(biased_indicators_):
    estimations_ = estimate_iteratively(biased_indicators_, biased_indicators_.sum())
    # return np.ceil(estimations)
    # return np.floor(estimations)
    return estimations_


def plot_estimations(biased_indicators_, estimations_, style, label):
    plt.plot(biased_indicators_ / biased_indicators_.sum(), estimations_, style, label=label)


def write_data_to_file(file_path, biased_indicators_, estimations_):
    print("writing...")
    f = open(file_path, 'w+')
    biased_indicators_ = biased_indicators_ / biased_indicators_.sum()
    for A in biased_indicators_:
        f.write(str(A) + " ")

    f.write('\n')

    for estimation in estimations_:
        f.write(str(estimation) + " ")
    f.close()


indicators = np.random.normal(size=N)
biased_indicators = utils.bias_function(b, J, optimal_value, indicators)

model_results_mean = np.array([influence_model.choose_role_model(indicators, b, J, optimal_value) for i in range(1000)]).mean(axis=0)
# estimations_r = estimate_recursively(biased_indicators, biased_indicators.sum())
estimations = estimate_iteratively(biased_indicators, biased_indicators.sum())

plt.plot(biased_indicators, model_results_mean, 'o', label="model simulation")
plt.plot(biased_indicators, estimations, 'o', label="estimations simulation")

plt.legend()
plt.show()
print("done")
