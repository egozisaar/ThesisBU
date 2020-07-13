import io_wrap.reader as reader
import common.classes as classes
import numpy as np
import matplotlib.pyplot as plt

correlations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
indirects = []
variations = []
files_list = []
path = "/Users/egozi/Documents/IDC/Thesis/src/python/thesis/basic_random_choice_model/data/initial_correlation_variable"

for i in correlations:
    files_list = np.array(reader.read_files_in_directory(path, title=classes.StatsTitle(traits_correlation=i)))
    mean_indirects = []
    for file in files_list:
        mean_indirects.append(np.mean(np.array([float(file[j][2]) for j in range(int(len(file) - len(file) * 0.1), len(file), 1)])))
    indirects.append(np.mean(mean_indirects))
    variations.append(np.var(mean_indirects))

optimal_indicator = float(files_list[0][0][5])
differences = np.abs(np.array([optimal_indicator for j in correlations]) - np.array(indirects))

plt.xlabel(r'$\chi$ - traits correlation')
plt.ylabel(r'$\hat{A} - \bar{B}$')
plt.plot(correlations, differences)

plt.figure()
plt.xlabel(r'$\chi$ - traits correlation')
plt.ylabel('Var(B)')
plt.plot(correlations, variations)

plt.show()
