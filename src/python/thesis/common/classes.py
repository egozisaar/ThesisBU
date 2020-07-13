from typing import NamedTuple
import enum


class Statistics(NamedTuple):
    mean: float
    interval_start: float
    interval_end: float
    var: float
    std: float


class WeightsType(enum.Enum):
    Random = 1
    Const = 2
    Corr = 3


class BasicIndividual:
    def __init__(self, indicator=0, indirect=0, indicator_err=0, indirect_err=0):
        self.indicator = indicator
        self.indirect = indirect
        self.indicator_err = indicator_err
        self.indirect_err = indirect_err


class ModelParams:
    def __init__(self, gen_size, gen_num, err_scale, traits_correlation, optimal_indicator, err_correlation):
        self.generation_size = gen_size
        self.number_of_generations = gen_num
        self.error_scale = err_scale
        self.traits_correlation = traits_correlation
        self.err_correlation = err_correlation
        self.optimal_indicator = optimal_indicator


class StatsTitle:
    def __init__(self, generation_size=None, generation_num=None, traits_correlation=None, err_correlation=None, err_scale=None,
                 optimal_indicator=None):
        self.generation_size = generation_size
        self.generation_num = generation_num
        self.traits_correlation = traits_correlation
        self.err_correlation = err_correlation
        self.err_scale = err_scale
        self.optimal_indicator = optimal_indicator

    def tuple(self):
        return self.generation_size, self.generation_num, self.traits_correlation, self.err_correlation, self.err_scale, self.optimal_indicator

    def compare(self, row_title):
        if (self.generation_size is None or self.generation_size == int(row_title[0])) and (
                self.generation_num is None or self.generation_num == int(row_title[1])) and (
                self.traits_correlation is None or self.traits_correlation == float(row_title[2])) and (
                self.err_correlation is None or self.err_correlation == float(row_title[3])) and (
                self.err_scale is None or self.err_scale == float(row_title[4])) and (
                self.optimal_indicator is None or self.optimal_indicator == float(row_title[5])):
            return True
        return False
