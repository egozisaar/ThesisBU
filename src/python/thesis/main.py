from generators import generate
import argparse
from io_wrap import writer, reader
from datetime import datetime
from basic_influence_model import model as basic_influence_model
from horizontal_model import model as horizontal_model
from moran_cyclic_model import model as cyclic_model
from boyd_richerson_model import model as boyd_richerson_model
from basic_random_choice_model import model as random_choice_model
from time import time
from common import classes as common
import estimate
import matplotlib.pyplot as plt
import numpy as np
from common import utils


def main():
    sys_params = parse_custom_arguments()
    model_params = get_model_params(sys_params)
    models_indicators, models_indirects = generate.matrix_2d_with_corr(model_params.traits_correlation, model_params.generation_size).T
    # data = boyd_richerson_model.run_model(model_params, models_indicators, models_indirects)
    # data = random_choice_model.run_model(model_params, models_indicators, models_indirects)
    # data = horizontal_model.run_model(model_params, models_indicators, models_indirects, common.WeightsType.Corr)
    # data = cyclic_model.run_model(model_params, models_indicators, models_indirects, common.WeightsType.Corr)
    iterations = 1000
    running_data = np.zeros(models_indicators.shape)
    for i in range(iterations):
        t = time()
        running_data += basic_influence_model.run_model(model_params, np.copy(models_indicators), models_indirects, common.WeightsType.Const)
        print(f'({i}) - {time() - t}')
    running_data /= iterations
    plt.xlabel(r'$\frac{\beta(A,\^A)}{\sum \beta(A,\^A)}$')
    plt.ylabel('number of copiers')
    plt.title('number of copiers - expected vs average real (1000 iterations)')
    biased_indicators = utils.bias_function(1, 0.5, 1, np.copy(models_indicators))
    estimate.plot_estimations(biased_indicators, running_data, 'bo', "real values")
    estimate.write_data_to_file("mean_real_data.txt", np.copy(biased_indicators), running_data)
    estimations = estimate.get_population_estimations(biased_indicators)
    estimate.plot_estimations(biased_indicators, estimations, 'ro', "expected values")
    estimate.write_data_to_file("expected_data.txt", np.copy(biased_indicators), estimations)

    plt.legend()
    plt.show()

    data_dir_path = "basic_random_choice_model/data/initial_correlation_variable"
    # writer.write_csv_file(name=get_data_file_name(model_params),
    #                       dest=data_dir_path,
    #                       title_row=get_title_from_params(model_params).tuple(),
    #                       data_rows=data)


def parse_custom_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="choose a model to run")
    parser.add_argument("-n", "--gensize", type=int, help="choose generation size")
    parser.add_argument("-k", "--gennum", type=int, help="choose num of generations")
    parser.add_argument("-tc", "--traitscorr", type=float, help="choose the correlation between traits")
    parser.add_argument("-ec", "--errcorr", type=float, help="choose the correlation between errors")
    parser.add_argument("-es", "--errscale", type=float, help="choose the error scale")
    parser.add_argument("-o", "--optimal", type=float, help="choose the optimal indicator value")
    parser.add_argument("-e", "--exist", type=str, help="choose if to use existing generation or not, relative path is required")
    parser.add_argument("-g", "--graph", action="store_true", help="choose if to display graphs after simulation")
    parser.add_argument("-p", "--plot", action="store_true", help="choose if to activate program in plot mode")
    return parser.parse_args()


def get_model_params(sys_params):
    gen_size = 100 if sys_params.gensize is None else int(sys_params.gensize)
    gen_num = 100 if sys_params.gennum is None else int(sys_params.gennum)
    err_scale = 1000 if sys_params.errscale is None else float(sys_params.errscale)
    traits_correlation = 0.7 if sys_params.traitscorr is None else float(sys_params.traitscorr)
    err_correlation = 0 if sys_params.errcorr is None else float(sys_params.errcorr)
    optimal_indicator = 1 if sys_params.optimal is None else float(sys_params.optimal)
    return common.ModelParams(gen_size=gen_size, gen_num=gen_num, err_scale=err_scale, traits_correlation=traits_correlation,
                              err_correlation=err_correlation, optimal_indicator=optimal_indicator)


def get_data_file_name(params):
    date_time = str(datetime.now()).replace(".", "_")
    file_name = "n_" + str(params.generation_size) + "_k_" + str(params.number_of_generations) + "_" + date_time
    return file_name


def get_title_from_params(params):
    return common.StatsTitle(generation_size=params.generation_size,
                             generation_num=params.number_of_generations,
                             traits_correlation=params.traits_correlation,
                             err_correlation=params.err_correlation,
                             err_scale=params.error_scale,
                             optimal_indicator=params.optimal_indicator)


if __name__ == "__main__":
    main()
