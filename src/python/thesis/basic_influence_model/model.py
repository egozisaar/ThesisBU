import generators.generate as generators
import numpy as np
from common import utils
from common import classes as common
from numba import njit

bias_J = 0.5
bias_b = 1

indicator_const_weight = 0.5
weights_scale = 10


def run_model(params, models_indicators, models_indirects, weights_var):
    data = []
    if weights_var == common.WeightsType.Corr:
        indicators_weights = np.power(models_indicators - params.optimal_indicator, 2)
        indicators_weights = 1 / (np.power(weights_scale, indicators_weights))

    for i in range(params.number_of_generations):
        data.append([np.mean(models_indicators), np.std(models_indicators), np.mean(models_indirects), np.std(models_indirects)])
        models_influence = np.ones(params.generation_size, dtype=int)
        errs_indicators, errs_indirects = generators.matrix_2d_with_corr(params.err_correlation, params.generation_size).T
        errs_indicators /= params.error_scale
        errs_indirects /= params.error_scale
        if weights_var != common.WeightsType.Corr:
            indicators_weights = get_weights(weights_var, params)
        models_indicators = models_indicators.reshape(1, -1) #+ errs_indicators.reshape(-1, 1)
        models_indirects = models_indirects.reshape(1, -1) + errs_indirects.reshape(-1, 1)
        biased_indicators = utils.bias_function(bias_b, bias_J, params.optimal_indicator, models_indicators)
        biased_indicators = biased_indicators * indicators_weights.reshape(-1, 1)
        models_indicators, models_indirects, indicators_weights = inform_copiers(biased_indicators, indicators_weights, models_influence,
                                                                                 models_indicators,

                                                                                 models_indirects)
        # This is a temporary return
        return models_influence

        models_indicators = np.array(models_indicators)
        models_indirects = np.array(models_indirects)
        indicators_weights = np.array(indicators_weights).reshape(-1)
    return np.array(data)


# models in columns, copiers in rows
@njit()
def inform_copiers(biased_indicators_matrix, indicator_weights, models_influence, indicators, indirects):
    new_indicators, new_indirects, new_indicators_weights = [], [], []
    for i in range(biased_indicators_matrix.shape[0]):
        copiers_biased_indicators = biased_indicators_matrix[i]
        weighted_influences = models_influence * (1 - indicator_weights[i])
        copier_scores = weighted_influences + copiers_biased_indicators
        copier_scores /= copier_scores.sum()
        chosen_model_index = np.searchsorted(np.cumsum(copier_scores), np.random.rand(1))
        models_influence[chosen_model_index] += 1
        new_indicators.append(indicators[i][chosen_model_index])
        new_indirects.append(indirects[i][chosen_model_index])
        new_indicators_weights.append(indicator_weights[chosen_model_index])
    return new_indicators, new_indirects, new_indicators_weights


def get_weights(weights_var, params):
    if weights_var == common.WeightsType.Random:
        indicators_weights = generators.random_0_to_1_vector(params.generation_size)
    elif weights_var == common.WeightsType.Const:
        indicators_weights = np.full(params.generation_size, indicator_const_weight)
    return indicators_weights
