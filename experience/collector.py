import os

import numpy as np
from grid2op.Converter import ToVect

from lib.chronics import get_sorted_chronics, is_loads_file, is_prods_file
from lib.constants import Constants as Const
from lib.data_utils import read_bz2_to_dataframe, load_python_module
from lib.visualizer import pprint


class ExperienceCollector(object):
    def __init__(self, save_dir):
        self.save_dir = save_dir

        self.config = None

        self.chronic_files = []
        self.chronic_ids = []
        self.chronic_lengths = []

        self.data = None

        self.obses = None
        self.actions = None
        self.rewards = None
        self.dones = None

        self.semi_actions = None

        self.distances = None
        self.distances_line = None
        self.distances_sub = None
        self.total_return = None
        self.duration = None
        self.chronic_len = None
        self.chronic_name = None

        self.reset()

    def full_reset(self):
        self.chronic_files = []
        self.chronic_ids = []
        self.chronic_lengths = []
        self.data = None

    def reset(self):
        self.obses = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.distances = []
        self.distances_line = []
        self.distances_sub = []

        self.semi_actions = []

        self.total_return = 0.0
        self.duration = 0
        self.chronic_len = 0
        self.chronic_name = ""

    @staticmethod
    def print_collector(phase):
        print("\n" + "-" * 80)
        pprint("", f"{phase} Experience")
        print("-" * 80)

    def collect(
        self, env, agent, do_chronics=(), n_chronics=-1, n_steps=-1, verbose=False
    ):
        self.print_collector("Collecting")
        agent.print_agent(default=verbose)

        agent_name = agent.name.replace(" ", "-").lower()
        self._load_chronics(agent_name=agent_name)

        chronics_dir, chronics, chronics_sorted = get_sorted_chronics(env=env)
        pprint("Chronics:", chronics_dir)

        if len(self.chronic_ids):
            pprint(
                "    - Done chronics:",
                ", ".join(map(lambda x: str(x), sorted(self.chronic_ids))),
            )

        if len(do_chronics):
            pprint(
                "    - To do chronics:",
                ", ".join(map(lambda x: str(x), sorted(do_chronics))),
            )

        done_chronic_ids = []
        for chronic_idx, chronic_name in enumerate(chronics_sorted):
            if len(done_chronic_ids) >= n_chronics > 0:
                break

            # If chronic already done
            if chronic_idx in self.chronic_ids:
                continue

            # Environment specific filtering
            if "rte_case5" in env.name:
                if chronic_idx not in do_chronics:
                    continue
            elif "l2rpn_2019" in env.name:
                if chronic_idx not in do_chronics:
                    continue
            elif "l2rpn_wcci_2020" in env.name:
                if chronic_idx not in do_chronics:
                    continue

            chronic_org_idx = chronics.index(chronic_name)
            env.chronics_handler.tell_id(chronic_org_idx - 1)  # Set chronic id

            obs = env.reset()
            agent.reset(obs=obs)

            chronic_len = env.chronics_handler.real_data.data.max_iter
            chronic_path_name = "/".join(
                os.path.normpath(env.chronics_handler.get_id()).split(os.sep)[-3:]
            )
            pprint("    - Chronic:", chronic_path_name)

            t = 0
            done = False
            reward = np.nan

            """
                Collect data.
            """
            while not done and not (t >= n_steps > 0):
                action = agent.act(obs, reward=reward, done=done)
                obs_next, reward, done, info = env.step(action)
                self._add(obs, action, reward, done)

                t = env.chronics_handler.real_data.data.current_index

                if t % 50 == 0:
                    pprint("        - Step:", t)

                if done:
                    pprint("        - Length:", f"{t}/{chronic_len}")

                obs = obs_next

            self.obses.append(obs.to_vect())
            done_chronic_ids.append(chronic_idx)

            self._save_chronic(agent_name, chronic_idx, verbose)
            self.reset()

    def _add(self, obs, action, reward, done):
        self.obses.append(obs.to_vect())
        self.actions.append(action.to_vect())
        self.rewards.append(reward)
        self.dones.append(done)

    def _add_plus(self, distance, distance_line, distance_sub, semi_action):
        self.distances.append(distance)
        self.distances_line.append(distance_line)
        self.distances_sub.append(distance_sub)

        self.semi_actions.append(semi_action)

    def _save_chronic(self, agent_name, chronic_idx, verbose=False):
        obses = np.array(self.obses)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        distances = np.array(self.distances)
        distances_line = np.array(self.distances_line)
        distances_sub = np.array(self.distances_sub)

        semi_actions = np.array(self.semi_actions)

        total_return = np.array(self.total_return)
        duration = np.array(self.duration)
        chronic_len = np.array(self.chronic_len)
        chronic_name = self.chronic_name

        agent_name = agent_name.replace(" ", "-").lower()
        chronic_file = "{}-chronic-{:05}.npz".format(agent_name, chronic_idx)
        pprint("        - Experience saved to:", chronic_file)

        if verbose:
            pprint("        - Observations:", obses.shape)
            pprint("        - Actions:", actions.shape)
            pprint("        - Rewards:", rewards.shape)
            pprint("        - Dones:", dones.shape)
            pprint("        - Distances:", distances.shape)
            pprint("            - Line:", distances_line.shape)
            pprint("            - Substation:", distances_sub.shape)
            pprint("        - Semi-Actions:", semi_actions.shape)
            pprint("        - Return:", total_return)
            pprint("        - Duration:", duration)
            pprint("        - Length:", chronic_len)
        else:
            pprint(
                "        - O A R D Dist TR Dur L:",
                obses.shape,
                actions.shape,
                rewards.shape,
                dones.shape,
                distances.shape,
                distances_line.shape,
                distances_sub.shape,
                semi_actions.shape,
                total_return,
                duration,
                chronic_len,
            )

        np.savez_compressed(
            os.path.join(self.save_dir, chronic_file),
            obses=obses,
            actions=actions,
            rewards=rewards,
            dones=dones,
            distances=distances,
            distances_line=distances_line,
            distances_sub=distances_sub,
            semi_actions=semi_actions,
            total_return=total_return,
            duration=duration,
            chronic_len=chronic_len,
            chronic_name=chronic_name,
        )

        self.chronic_files.append(chronic_file)
        self.chronic_ids.append(chronic_idx)

    def _load_chronics(self, agent_name):
        pprint(
            "    - Loading chronics:",
            self.save_dir.replace("\\", "/") + f"/{agent_name}-chronic-****",
        )
        for chronic_file in os.listdir(self.save_dir):
            if (
                "chronic-" in chronic_file
                and agent_name in chronic_file
                and ".npz" in chronic_file
            ):
                chronic_idx = int(os.path.splitext(chronic_file)[0].split("-")[-1])
                self.chronic_files.append(chronic_file)
                self.chronic_ids.append(chronic_idx)

    def load_data(self, agent_name, env):
        self.full_reset()
        self.print_collector("Loading")
        self._load_chronics(agent_name=agent_name)

        chronic_data = dict()
        for chronic_idx, chronic_file in zip(self.chronic_ids, self.chronic_files):
            chronic_data[chronic_idx] = dict()

            npz_file = np.load(os.path.join(self.save_dir, chronic_file))
            for key in npz_file.keys():
                chronic_data[chronic_idx][key] = npz_file[key]

            self.chronic_lengths.append(len(npz_file["rewards"]))

        self.data = chronic_data
        self.transform_data(env=env)

    def transform_data(self, env):
        converter = ToVect(action_space=env.action_space)

        for chronic_idx in self.data:
            data_chronic = self.data[chronic_idx]

            obses = []
            actions = []
            for obs_vect in data_chronic["obses"]:
                obs = env.observation_space.from_vect(obs_vect)
                obses.append(obs)

            for action_vect in data_chronic["actions"]:
                action = converter.convert_act(action_vect)
                actions.append(action)

            self.data[chronic_idx]["obses"] = obses
            self.data[chronic_idx]["actions"] = actions

    def aggregate_data(self, verbose=False):
        observations = []
        actions = []
        rewards = []
        dones = []
        for chronic_idx in self.data:
            data_chronic = self.data[chronic_idx]
            obses_chronic = data_chronic["obses"]
            actions_chronic = data_chronic["actions"]
            rewards_chronic = data_chronic["rewards"]
            dones_chronic = data_chronic["dones"]

            if verbose:
                pprint("Chronic:", chronic_idx)
                pprint(
                    "        - O A R D:",
                    len(obses_chronic),
                    len(actions_chronic),
                    rewards_chronic.shape,
                    dones_chronic.shape,
                )

            observations.extend(obses_chronic[:-1])
            actions.extend(actions_chronic)
            rewards.extend(rewards_chronic.tolist())
            dones.extend(dones_chronic.tolist())

        return observations, actions, np.array(rewards), np.array(dones)

    def load_forecasts(self, env, chronic_idx):
        datasets_path = Const.DATASET_DIR
        case_path = os.path.join(datasets_path, env.name)

        if not self.config:
            module = load_python_module(os.path.join(case_path, "config.py"), name=".")
            self.config = module.config

        prods, loads = load_forecasts(env, chronic_idx, self.config)
        return prods, loads


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
