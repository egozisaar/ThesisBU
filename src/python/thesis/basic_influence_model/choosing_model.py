import numpy as np
from common import utils

err_scale = 0.01
indicator_weight = 0.5


def choose_role_model(models_indicators, b=1, J=0.5, optimal_value=1):
    influences = np.zeros_like(models_indicators)
    copiers_err = np.random.normal(size=models_indicators.size, scale=err_scale)
    reshaped_indicators = models_indicators.reshape(1, -1)
    reshaped_copier_errs = copiers_err.reshape(-1, 1)
    perceived_indicators_matrix = reshaped_indicators + reshaped_copier_errs
    biased_indicators_matrix = utils.bias_function(bias_b=b, bias_J=J, optimal_value=optimal_value,
                                                   perceived_traits=perceived_indicators_matrix) * indicator_weight
    random_indices = np.random.rand(models_indicators.size)
    for i, copier_scores in enumerate(biased_indicators_matrix):
        copier_scores += (influences * (1 - indicator_weight))
        copier_scores /= copier_scores.sum()
        copier_scores = copier_scores.cumsum()
        chosen_role_model_index = np.searchsorted(copier_scores, random_indices[i])
        influences[chosen_role_model_index] += 1
    return influences
