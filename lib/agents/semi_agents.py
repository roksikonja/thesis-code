import os

import numpy as np
import tensorflow as tf

from bclassification.dnn import load_model_kwargs, load_data_kwargs, load_dnn
from bclassification.utils_fcn import obs_to_vect, action_to_vect
from lib.action_space import is_do_nothing_action
from lib.dc_opf import TopologyConverter, ForecastsFromFile
from lib.visualizer import pprint


class SemiAgentBase:
    def take_switching_action(self, observation, action):
        pass

    def reset(self):
        pass


class SemiAgentIL(SemiAgentBase):
    def __init__(self, case, model_dir):
        self.name = "-il"

        self.model_dir = model_dir
        self.model_kwargs, self.data_kwargs = self.load_kwargs()
        self.model = self.load_model()

        self.env = case.env
        self.tc = TopologyConverter(self.env)
        self.forecasts = ForecastsFromFile(self.env, t=1, horizon=1)

        self.x_obses = None
        self.x_actions = None
        self.x_forecasts = None

        self.semi_action = None

    def take_switching_action(self, observation, action):
        x = self.create_x(observation, action)
        y_pred = np.squeeze(self.model(x).numpy())
        self.semi_action = y_pred > 0.5

        pprint(self.forecasts.t, self.semi_action)
        self.forecasts.t += 1
        return self.semi_action

    def reset(self):
        self.semi_action = None
        self.forecasts.reset()

    def create_x(self, observation, action):
        action = is_do_nothing_action([action], self.env)

        self.x_obses = obs_to_vect(observation, self.tc, self.data_kwargs["input_mode"])
        self.x_actions = action_to_vect(action)
        self.x_forecasts = np.hstack(
            (self.forecasts.prod_p.flatten(), self.forecasts.load_p.flatten())
        )

        if self.data_kwargs["feature_scaling"]:
            if self.data_kwargs["input_mode"] == "binary":
                n_features = 3 * self.tc.n_gen + self.tc.n_load + self.tc.n_line
                self.x_obses[:n_features] /= self.data_kwargs["p_scaling_std"]

            elif self.data_kwargs["input_mode"] == "structured":
                n_features = 4 * self.tc.n_gen + 2 * self.tc.n_load + 4 * self.tc.n_line
                self.x_obses[:n_features] /= self.data_kwargs["p_scaling_std"]

            self.x_forecasts /= self.data_kwargs["p_scaling_std"]

        x = np.hstack((self.x_obses, self.x_actions, self.x_forecasts))
        x = np.reshape(x, newshape=(1, -1))
        return x

    def load_kwargs(self):
        kwargs_file = os.path.join(self.model_dir, "params.txt")

        model_kwargs = load_model_kwargs(kwargs_file)
        data_kwargs = load_data_kwargs(kwargs_file)

        return model_kwargs, data_kwargs

    def load_model(self):
        model = load_dnn(**self.model_kwargs)

        ckpt_dir = os.path.join(self.model_dir, "ckpts")
        ckpt = tf.train.Checkpoint(model=model, optimizer=tf.keras.optimizers.Adam())
        ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

        pprint("Restoring checkpoint from:", ckpt_manager.latest_checkpoint)
        return model


class SemiAgentMaxRho(SemiAgentBase):
    def __init__(self, max_rho):
        self.name = "-max-rho"
        self.max_rho = max_rho

        self.semi_action = None

    def take_switching_action(self, observation, action):
        max_rho = observation.rho.max()

        self.semi_action = max_rho > self.max_rho
        return self.semi_action

    def reset(self):
        self.semi_action = None


class SemiAgentRandom(SemiAgentBase):
    def __init__(self, probability):
        self.name = "-random"
        self.probability = probability

        self.semi_action = None

    def take_switching_action(self, observation, action):
        # Sample from Bernoulli distribution
        sample = np.random.binomial(1, self.probability)

        self.semi_action = sample == 1
        return self.semi_action

    def reset(self):
        self.semi_action = None


class SemiAgentKSteps(SemiAgentBase):
    def __init__(self, k):
        self.name = "-k-steps"
        self.t = 0
        self.k = k

        self.semi_action = None

    def take_switching_action(self, observation, action):
        remainder = self.t % self.k

        self.t += 1
        self.semi_action = remainder == 0
        return self.semi_action

    def reset(self):
        self.t = 0
        self.semi_action = None
