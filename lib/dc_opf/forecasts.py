import os

import numpy as np

from lib.chronics import get_sorted_chronics, is_loads_file, is_prods_file
from lib.data_utils import read_bz2_to_dataframe, load_python_module


def load_forecasts(env, chronic_idx, config):
    chronics_dir, chronics, chronics_sorted = get_sorted_chronics(env=env)

    case_chronics = env.chronics_handler.path
    chronic_name = chronics_sorted[chronic_idx]
    chronic_dir = os.path.join(case_chronics, chronic_name)

    prods_file = [file for file in os.listdir(chronic_dir) if is_prods_file(file)]
    loads_file = [file for file in os.listdir(chronic_dir) if is_loads_file(file)]
    assert len(prods_file) == 1 and len(loads_file) == 1

    prods = read_bz2_to_dataframe(os.path.join(chronic_dir, prods_file[0]), sep=";")
    loads = read_bz2_to_dataframe(os.path.join(chronic_dir, loads_file[0]), sep=";")

    gen_org_to_grid_name = config["names_chronics_to_grid"]["prods"]
    load_org_to_grid_name = config["names_chronics_to_grid"]["loads"]

    prods = prods.rename(columns=gen_org_to_grid_name)
    prods = prods.reindex(sorted(prods.columns, key=lambda x: x.split("_")[-1]), axis=1)

    loads = loads.rename(columns=load_org_to_grid_name)
    loads = loads.reindex(sorted(loads.columns, key=lambda x: x.split("_")[-1]), axis=1)

    return prods.values, loads.values


class Forecasts:
    def __init__(self, env, t=0, horizon=2):
        self.t = t
        self.horizon = horizon
        self.time_steps = np.arange(self.horizon)

        self.env = env
        self.data = self._get_chronic_data()

    @property
    def load_p(self):
        return self.data.load_p[self.t : self.t + self.horizon, :]

    @property
    def prod_p(self):
        return self.data.prod_p[self.t : self.t + self.horizon, :]

    def _get_chronic_data(self):
        return self.env.chronics_handler.real_data.data


class ForecastsPlain:
    def __init__(self, env, t=0, horizon=2):
        self.t = t
        self.horizon = horizon
        self.time_steps = np.arange(self.horizon)

        self.env = env

    def __bool__(self):
        return False


class ForecastsFromFile:
    def __init__(self, env, t=0, horizon=2):
        self.t = t
        self.horizon = horizon
        self.time_steps = np.arange(self.horizon)

        self.env = env
        self.config = self.load_config()
        self.data_prod_p, self.data_load_p = self._get_chronic_data()

    @property
    def load_p(self):
        return self.data_load_p[self.t : self.t + self.horizon, :]

    @property
    def prod_p(self):
        return self.data_prod_p[self.t : self.t + self.horizon, :]

    def load_config(self):
        case_path = os.path.join(self.env.chronics_handler.path, "..")
        module = load_python_module(os.path.join(case_path, "config.py"), name=".")
        return module.config

    def _get_chronic_data(self):
        chronic_idx = int(self.env.chronics_handler.get_name())
        print("chronic loaded", chronic_idx)

        prods, loads = load_forecasts(self.env, chronic_idx, self.config)

        return prods, loads

    def reset(self):
        self.data_prod_p, self.data_load_p = self._get_chronic_data()
        self.t = 1
