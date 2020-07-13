import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as smooth_graph
import numpy as np
import io_wrap.reader as reader
import common.classes as common


def plot_data(data, optimal_val, num_of_gens):
    plt.rcParams.update({'font.size': 15})
    data = np.array(data)
    plt.xlabel("generations")
    plt.ylabel("trait value")
    plot_optimal(optimal_val, num_of_gens)
    indicator_means = data[:, 0]
    indicator_std = data[:, 1]
    smooth_and_plot(indicator_means, indicator_std, "indicator trait")

    indirect_means = data[:, 2]
    indirect_std = data[:, 3]

    smooth_and_plot(indirect_means, indirect_std, "indirect trait")
    plt.legend()
    plt.show()


def smooth_and_plot(means, std_vector, label):
    std_bottom = smooth_graph(means - std_vector, 51, 3)
    std_up = smooth_graph(means + std_vector, 51, 3)
    means = smooth_graph(means, 51, 3)
    x_axis = range(len(means))
    plt.plot(x_axis, means, label=label, linestyle="-")
    plt.fill_between(x_axis, std_bottom, std_up, alpha=0.2)


def plot_optimal(optimal_val, generations):
    plt.plot([i for i in range(generations)],
             [optimal_val for _ in range(generations)],
             label="optimal value", color="black", linestyle="dotted")


def plot_directory_means(dir_path, title):
    raw_data = np.array(reader.read_files_in_directory(dir_path, title))
    if len(raw_data) == 0:
        print("no matching files were found to given title")
        return
    print(f"found {len(raw_data)} files")
    data = []
    for i in range(1, title.generation_num + 1, 1):
        indicators = np.array([float(file[i][0]) for file in raw_data])
        std_indicators = np.array([float(file[i][1]) for file in raw_data])
        indirects = np.array([float(file[i][2]) for file in raw_data])
        std_indirects = np.array([float(file[i][3]) for file in raw_data])
        data.append([indicators.mean(), std_indicators.mean(), indirects.mean(), std_indirects.mean()])
    plot_data(data, title.optimal_indicator, title.generation_num)


def plot_single_file(file_path):
    raw_data = reader.read_csv_file_by_path(file_path)
    return raw_data


dir_path = "basic_influence_model/data"
title = common.StatsTitle(generation_size=5000, generation_num=1000, traits_correlation=0.9,
                          err_correlation=None, err_scale=None, optimal_indicator=1)
plot_directory_means(dir_path, title)
