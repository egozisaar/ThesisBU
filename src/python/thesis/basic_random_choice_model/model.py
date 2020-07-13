from generators import generate
import numpy as np
from common import utils
from numba import njit
import time

bias_J_indicator = 2
bias_b_indicator = 1
mean_social_weight = 1  # the mean relative value the social rank will have. will be trimmed between 0 and 2


def run_model(params, models_indicators, models_indirects):
    data = []
    for i in range(params.number_of_generations):
        data.append([np.mean(models_indicators), np.std(models_indicators), np.mean(models_indirects), np.std(models_indirects)])

        indicator_social_ranks = np.random.normal(mean_social_weight, 1, params.generation_size)
        # indicator_social_ranks = \
        trim(indicator_social_ranks)

        errs_indicators, errs_indirects = generate.matrix_2d_with_corr(params.err_correlation, params.generation_size).T  # divide here
        errs_indicators /= params.error_scale
        errs_indirects /= params.error_scale

        models_indicators = models_indicators.reshape(1, -1) + errs_indicators.reshape(-1, 1)  # add errors
        models_indirects = models_indirects.reshape(1, -1) + errs_indirects.reshape(-1, 1)  # add errors

        biased_indicators = utils.bias_function(bias_b_indicator, bias_J_indicator, params.optimal_indicator, models_indicators)  # calculate bias
        biased_indicators = (indicator_social_ranks.reshape(1, -1) * biased_indicators) + biased_indicators
        biased_indicators /= biased_indicators.sum(axis=1, keepdims=True)#.reshape(-1, 1)
        t = time.time()
        models_indicators, models_indirects = inform_copiers(biased_indicators, models_indicators, models_indirects)
        print(time.time() - t)
        models_indicators = np.array(models_indicators)
        models_indirects = np.array(models_indirects)
    return np.array(data)


@njit()
def inform_copiers(biased_indicators, models_indicators, models_indirects):
    new_indicators, new_indirects = [], []
    for i in range(biased_indicators.shape[0]):
        models = biased_indicators[i]
        chosen_model = np.searchsorted(np.cumsum(models), np.random.rand(1))
        new_indicators.append(models_indicators[i][chosen_model])
        new_indirects.append(models_indirects[i][chosen_model])
    return new_indicators, new_indirects


@njit()
def trim(A):
    A[A < 0] = 0
    A[A > 2] = 2
    # return A
