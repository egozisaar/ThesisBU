import random
import numpy as np
from numba import njit


def calculate_scores_list(models_list, noise, evaluate, copier=None):
    scores = [evaluate(model, copier) for model in models_list]
    normalized_noise = (sum(scores) / len(scores)) * noise
    noisy_scores = [(score + normalized_noise) for score in scores]
    return noisy_scores


def get_random_in_range(minimum, maximum):
    scale = (maximum - minimum)
    seed = random.random()
    return (seed * scale) + minimum


def limit_trait_val(trait_val):
    if trait_val < 0:
        return 0
    return trait_val


def bias_function(bias_b, bias_J, optimal_value, perceived_traits):
    return bias_b * np.exp(-(((optimal_value - perceived_traits) ** 2) / (2 * bias_J)))
