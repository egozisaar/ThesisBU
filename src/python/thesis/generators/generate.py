from common.classes import BasicIndividual
import numpy as np
from io_wrap import writer
from typing import NamedTuple


class GenZero(NamedTuple):
    generation_size: int
    traits_correlation: float
    err_correlation: float
    optimal_indicator: float


def generate(n, traits_correlation, err_correlation, optimal_indicator_val, err_scale):
    means = [0, 0]
    cov = [[1, traits_correlation],
           [traits_correlation, 1]]
    indicators, indirects = np.random.multivariate_normal(means, cov, n).T
    cov = [[1, err_correlation],
           [err_correlation, 1]]
    indicator_errs, indirect_errs = np.random.multivariate_normal(means, cov, n).T
    gen = [BasicIndividual(indicator=indicators[i], indirect=indirects[i],
                           indicator_err=indicator_errs[i] / err_scale, indirect_err=indirect_errs[i] / err_scale)
           for i in range(n)]

    file_name = "n_" + str(n) + "_tCorr_" + str(traits_correlation) + "_eCorr_" + str(err_correlation) + "_eScale_" + str(err_scale)
    title = GenZero(err_correlation=err_correlation, generation_size=n, traits_correlation=traits_correlation,
                    optimal_indicator=optimal_indicator_val)
    data = [[individual.indicator, individual.indirect, individual.indicator_err, individual.indirect_err] for individual in gen]
    writer.write_csv_file(name=file_name,
                          dest="generators/resources",
                          title_row=title,
                          data_rows=data)
    return gen


def matrix_2d_with_corr(corr, size, mean=None):
    means = [0, 0]
    if mean is not None:
        means = [mean, mean]
    cov = [[1, corr],
           [corr, 1]]
    return np.random.multivariate_normal(means, cov, size=size)


def random_0_to_1_vector(size):
    return np.random.random_sample(size)
