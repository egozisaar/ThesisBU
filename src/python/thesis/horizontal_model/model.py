import generators.generate as generators
import numpy as np
from numba import njit
from common import classes as common
from time import time

bias_J = 2
bias_b = 1

indicator_const_weight = 0.5
weights_scale = 10


def run_model(params, models_indicators, models_indirects, weights_var):
    data = []
    for i in range(params.number_of_generations):
        data.append([np.mean(models_indicators), np.std(models_indicators), np.mean(models_indirects), np.std(models_indirects)])
        models_influence = np.ones(params.generation_size, dtype=int)
        errs_indicators, errs_indirects = generators.matrix_2d_with_corr(params.err_correlation, params.generation_size).T
        errs_indicators /= params.error_scale
        errs_indirects /= params.error_scale
        indicators_weights = get_weights(weights_var.value, params.generation_size, errs_indicators)
        models_indicators, models_indirects = inform_copiers(indicators_weights, models_influence,
                                                             models_indicators, models_indirects,
                                                             errs_indicators, errs_indirects, params.generation_size,
                                                             params.optimal_indicator)
        models_indicators = np.array(models_indicators).reshape(-1)
        models_indirects = np.array(models_indirects).reshape(-1)
    return np.array(data)


# models in columns, copiers in rows
@njit()
def inform_copiers(indicators_weights, models_influence, models_indicators, models_indirects, errs_indicators, errs_indirects,
                   gen_size, optimal_indicator):
    new_indicators, new_indirects = [], []
    second_hand_influence = np.full(gen_size, -1)
    for i in range(len(errs_indicators)):
        copier_biased_indicators, copier_indicators, weighted_influences = calculate_biases(errs_indicators, i, indicators_weights, models_indicators,
                                                                                            models_influence, optimal_indicator)

        chosen_model_index = choose_model(copier_biased_indicators, weighted_influences)

        # next generation
        new_indicators.append(copier_indicators[chosen_model_index])
        new_indirects.append(models_indirects[chosen_model_index] + errs_indirects[i])

        models_indicators, models_indirects, models_influence, second_hand_influence = update_current_generation(chosen_model_index,
                                                                                                                 copier_indicators, errs_indirects[i],
                                                                                                                 models_indicators, models_indirects,
                                                                                                                 models_influence,
                                                                                                                 second_hand_influence)

        update_deep_influence(chosen_model_index, models_influence, second_hand_influence)
    return new_indicators, new_indirects


@njit()
def choose_model(copier_biased_indicators, weighted_influences):
    copier_scores = weighted_influences + copier_biased_indicators
    copier_scores /= copier_scores.sum()  # to keep make the sum of the array 1
    chosen_model_index = np.searchsorted(np.cumsum(copier_scores), np.random.rand(1))
    return chosen_model_index


@njit()
def calculate_biases(errs_indicators, i, indicators_weights, models_indicators, models_influence, optimal_indicator):
    copier_indicators = errs_indicators[i] + models_indicators
    copier_biased_indicators = bias_b * np.exp(-(((optimal_indicator - copier_indicators) ** 2) / (2 * bias_J)))
    copier_biased_indicators *= indicators_weights[i]
    weighted_influences = models_influence * (1 - indicators_weights[i])
    return copier_biased_indicators, copier_indicators, weighted_influences


@njit()
def update_deep_influence(chosen_model_index, models_influence, second_hand_influence):
    # updating the influences across the whole array
    models_influence[chosen_model_index] += 1
    parent_index = second_hand_influence[chosen_model_index]
    # choosing first element to support numba
    while parent_index[0] > -1:
        models_influence[parent_index] += 1
        parent_index = second_hand_influence[parent_index]


@njit()
def update_current_generation(chosen_model_index, copier_indicators, err_indirect, models_indicators, models_indirects, models_influence,
                              second_hand_influence):
    models_indicators = np.append(models_indicators, copier_indicators[chosen_model_index])
    models_indirects = np.append(models_indirects, models_indirects[chosen_model_index] + err_indirect)
    models_influence = np.append(models_influence, 1)
    second_hand_influence = np.append(second_hand_influence, chosen_model_index)
    return models_indicators, models_indirects, models_influence, second_hand_influence


@njit()
def get_weights(weights_var, generation_size, copiers_errs):
    if weights_var == 1:
        indicators_weights = np.random.random_sample(generation_size)
    elif weights_var == 2:
        indicators_weights = np.full(generation_size, indicator_const_weight)
    else:
        indicators_weights = 1 - np.abs(copiers_errs)  # closer to 0 should be closer to 1
        indicators_weights = (indicators_weights - np.mean(indicators_weights)) / np.std(indicators_weights)  # standardising the weights
        min_val, max_val = indicators_weights.min(), indicators_weights.max()
        indicators_weights = (indicators_weights - min_val) / (max_val - min_val)  # rescaling
    return indicators_weights
