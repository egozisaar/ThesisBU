from generators import generate
import numpy as np
from common import utils
from numba import njit
from time import time

bias_J_indicator = 2
bias_b_indicator = 1
bias_J_indirect = 4
bias_b_indirect = 0.5
social_rank_correlation = 0.5
mean_social_weight = 1  # the mean relative value the social rank will have. will be trimmed between 0 and 2


def run_model(params, models_indicators, models_indirects):
    data = []
    for i in range(params.number_of_generations):
        t = time()
        data.append([np.mean(models_indicators), np.std(models_indicators), np.mean(models_indirects), np.std(models_indirects)])

        indicator_social_ranks, indirect_social_ranks = generate.matrix_2d_with_corr(social_rank_correlation, params.generation_size,
                                                                                     mean_social_weight).T
        indicator_social_ranks = trim(indicator_social_ranks)
        indirect_social_ranks = trim(indirect_social_ranks)

        errs_indicators, errs_indirects = generate.matrix_2d_with_corr(params.err_correlation, params.generation_size).T
        errs_indicators /= params.error_scale
        errs_indirects /= params.error_scale

        models_indicators = models_indicators.reshape(1, -1) + errs_indicators.reshape(-1, 1)  # add errors
        models_indirects = models_indirects.reshape(1, -1) + errs_indirects.reshape(-1, 1)  # add errors

        biased_indicators = utils.bias_function(bias_b_indicator, bias_J_indicator, params.optimal_indicator, models_indicators)  # calculate bias
        biased_indirects = utils.bias_function(bias_b_indirect, bias_J_indirect, params.optimal_indicator, models_indicators)  # calculate bias
        biased_indicators = (indicator_social_ranks.reshape(1, -1) * biased_indicators) + biased_indicators
        biased_indirects = (indirect_social_ranks.reshape(1, -1) * biased_indirects) + biased_indirects

        models_indicators = np.average(models_indicators, axis=1, weights=biased_indicators)
        models_indirects = np.average(models_indirects, axis=1, weights=biased_indirects)

    return np.array(data)


@njit
def trim(A):
    A[A < 0] = 0
    A[A > 2] = 2
    return A
