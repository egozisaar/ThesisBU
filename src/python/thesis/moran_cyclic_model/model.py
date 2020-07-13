import generators.generate as generators
import numpy as np
from numba import njit
from common import classes as common
from time import time

bias_J = 2
bias_b = 1

indicator_const_weight = 0.5
weights_scale = 10

death_cycles = 10
selection_influence = 1


def run_model(params, models_indicators, models_indirects, weights_var):
    models_influences = np.ones(models_indicators.shape)
    models_influencers = np.full(models_indicators.shape, -1)
    cycles_alive = np.full(models_indicators.shape, 0)
    # birth of individuals is independent, hence we can generate them in a matrix
    errs_indicators, errs_indirects = generators.matrix_2d_with_corr(params.err_correlation, params.number_of_generations).T
    errs_indicators /= params.error_scale
    errs_indirects /= params.error_scale
    indicator_weights = get_weights(weights_var.value, errs_indicators.shape[0], errs_indicators)
    data = evolve(cycles_alive, errs_indicators, errs_indirects, indicator_weights, models_indicators, models_indirects, models_influencers,
                  models_influences, params.optimal_indicator)
    return np.array(data)


@njit()
def evolve(cycles_alive, errs_indicators, errs_indirects, indicator_weights, models_indicators, models_indirects, models_influencers,
           models_influences, optimal_indicator):
    data = []
    for i in range(errs_indicators.shape[0]):
        data.append([np.mean(models_indicators), np.std(models_indicators), np.mean(models_indirects), np.std(models_indirects)])

        # bias
        copier_biased_indicators, copier_indicators, weighted_influences = calculate_bias(errs_indicators, i, indicator_weights, models_indicators,
                                                                                          models_influences, optimal_indicator)

        chosen_model_index = calculate_scores(copier_biased_indicators, weighted_influences)

        # arrays updates
        cycles_alive, models_indicators, models_indirects, models_influencers, models_influences = update_arrays(chosen_model_index,
                                                                                                                 copier_indicators, cycles_alive,
                                                                                                                 errs_indirects[i], models_indicators,
                                                                                                                 models_indirects, models_influencers,
                                                                                                                 models_influences)

        update_deep_influence(i, chosen_model_index, models_influencers, models_influences)

        # determine death
        if (i + 1) % death_cycles == 0:
            cycles_alive, models_indicators, models_indirects, models_influencers, models_influences = kill_individuals(cycles_alive,
                                                                                                                        models_indicators,
                                                                                                                        models_indirects,
                                                                                                                        models_influencers,
                                                                                                                        models_influences)
    return data


@njit()
def kill_individuals(cycles_alive, models_indicators, models_indirects, models_influencers, models_influences):
    for i in range(death_cycles):
        random_misfortunes = np.random.randn(cycles_alive.shape[0])
        selection = selection_influence * models_indirects
        death_scores = cycles_alive + random_misfortunes + selection
        # choose who will die
        death_scores /= death_scores.sum()  # to keep make the sum of the array 1
        death_index = np.searchsorted(np.cumsum(death_scores), np.random.rand(1))

        # delete the dead
        models_indicators = np.delete(models_indicators, death_index)
        models_indirects = np.delete(models_indirects, death_index)
        models_influences = np.delete(models_influences, death_index)
        cycles_alive = np.delete(cycles_alive, death_index)

        models_influencers[models_influencers == death_index] = -1
        models_influencers[models_influencers > death_index] -= 1
        models_influencers = np.delete(models_influencers, death_index)
    return cycles_alive, models_indicators, models_indirects, models_influencers, models_influences


@njit()
def update_deep_influence(i, chosen_model_index, models_influencers, models_influences):
    # updating the influences across the whole array
    models_influences[chosen_model_index] += 1
    parent_index = models_influencers[chosen_model_index]
    # choosing first element to support numba
    count = 0
    while parent_index[0] > -1:
        models_influences[parent_index] += 1
        parent_index = models_influencers[parent_index]
        count += 1


@njit()
def update_arrays(chosen_model_index, copier_indicators, cycles_alive, err_indirect, models_indicators, models_indirects, models_influencers,
                  models_influences):
    models_indicators = np.append(models_indicators, copier_indicators[chosen_model_index])
    models_indirects = np.append(models_indirects, models_indirects[chosen_model_index] + err_indirect)
    models_influences = np.append(models_influences, 1)
    models_influencers = np.append(models_influencers, chosen_model_index)
    cycles_alive += 1
    cycles_alive = np.append(cycles_alive, 0)
    return cycles_alive, models_indicators, models_indirects, models_influencers, models_influences


@njit()
def calculate_scores(copier_biased_indicators, weighted_influences):
    # score evaluation
    copier_scores = weighted_influences + copier_biased_indicators
    copier_scores /= copier_scores.sum()  # to keep make the sum of the array 1
    chosen_model_index = np.searchsorted(np.cumsum(copier_scores), np.random.rand(1))
    return chosen_model_index


@njit()
def calculate_bias(errs_indicators, i, indicator_weights, models_indicators, models_influences, optimal_indicator):
    copier_indicators = errs_indicators[i] + models_indicators
    copier_biased_indicators = bias_b * np.exp(-(((optimal_indicator - copier_indicators) ** 2) / (2 * bias_J)))
    copier_biased_indicators *= indicator_weights[i]
    weighted_influences = models_influences * (1 - indicator_weights[i])
    return copier_biased_indicators, copier_indicators, weighted_influences


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
