import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyPDF2 import PdfFileMerger

from lib.chronics import get_sorted_chronics
from lib.constants import Constants as Const
from lib.rl_utils import compute_returns
from lib.visualizer import pprint
from .experiment_base import ExperimentBase, ExperimentMixin


class ExperimentSwitching(ExperimentBase, ExperimentMixin):
    def analyse(self, case, agent, save_dir=None, verbose=False):
        env = case.env

        self.print_experiment("Switching")
        agent.print_agent(default=verbose)

        file_name = agent.name.replace(" ", "-").lower() + "-chronics"
        chronic_data, done_chronic_indices = self._load_done_chronics(
            file_name=file_name, save_dir=save_dir
        )

        new_chronic_data = self._runner(
            case=case, env=env, agent=agent, done_chronic_indices=done_chronic_indices
        )
        chronic_data = chronic_data.append(new_chronic_data)

        self._save_chronics(
            chronic_data=chronic_data, file_name=file_name, save_dir=save_dir
        )

    def compare_agents(self, case, save_dir=None, delete_file=True):
        case_name = self._get_case_name(case)
        chronic_data = dict()

        agent_names = []
        for file in os.listdir(save_dir):
            if "-chronics.pkl" in file:
                agent_name = file[: -len("-chronics.pkl")]
                agent_name = agent_name.replace("-", " ").capitalize()
                agent_names.append(agent_name)
                chronic_data[agent_name] = pd.read_pickle(os.path.join(save_dir, file))

        chronic_indices_all = pd.Index([], name="chronic_idx")
        for agent_name in agent_names:
            chronic_indices_all = chronic_indices_all.union(
                chronic_data[agent_name].index
            )

        chronic_names_all = pd.DataFrame(
            columns=["chronic_name"], index=chronic_indices_all
        )
        for agent_name in agent_names:
            chronic_names_all["chronic_name"].loc[
                chronic_data[agent_name].index
            ] = chronic_data[agent_name]["chronic_name"]

        for chronic_idx in chronic_indices_all:
            chronic_name = chronic_names_all["chronic_name"].loc[chronic_idx]

            self._plot_rewards(
                chronic_data, case_name, chronic_idx, chronic_name, save_dir
            )

            for dist, ylabel in [
                ("distances", "Unitary action distance to reference topology"),
                ("distances_status", "Line status distance to reference topology"),
                ("distances_sub", "Substation distance distance to reference topology"),
            ]:
                self._plot_distances(
                    chronic_data,
                    dist,
                    ylabel,
                    case_name,
                    chronic_idx,
                    chronic_name,
                    save_dir,
                )

            self._plot_gains(
                chronic_data, case_name, chronic_idx, chronic_name, save_dir
            )

        self.aggregate_by_chronics(save_dir, delete_file=delete_file)

    def aggregate_by_chronics(self, save_dir, delete_file=True):
        for plot_name in [
            "rewards",
            "distances",
            "distances_status",
            "distances_sub",
            "gains",
            "objectives",
            "rhos",
        ]:
            merger = PdfFileMerger()
            chronic_files = []
            for file in os.listdir(save_dir):
                if "agents-chronic-" in file and plot_name + ".pdf" in file:
                    f = open(os.path.join(save_dir, file), "rb")
                    chronic_files.append((file, f))
                    merger.append(f)

            if merger.inputs:
                with open(
                    os.path.join(save_dir, "_" + f"agents-chronics-{plot_name}.pdf"),
                    "wb",
                ) as f:
                    merger.write(f)

                # Reset merger
                merger.pages = []
                merger.inputs = []
                merger.output = None

            self.close_files(chronic_files, save_dir, delete_file=delete_file)

    @staticmethod
    def _plot_gains(chronic_data, case_name, chronic_idx, chronic_name, save_dir):
        colors = Const.COLORS

        fig_obj, ax_obj = plt.subplots(figsize=Const.FIG_SIZE)
        fig_gain, ax_gain = plt.subplots(figsize=Const.FIG_SIZE)
        for agent_id, agent_name in enumerate(chronic_data):
            if chronic_idx in chronic_data[agent_name].index and agent_name != "Do nothing agent":
                color_id = agent_id % len(colors)
                color = colors[color_id]

                t = chronic_data[agent_name].loc[chronic_idx]["time_steps"]
                actions = chronic_data[agent_name].loc[chronic_idx]["actions"]

                obj = np.array(chronic_data[agent_name].loc[chronic_idx]["objectives"])
                obj_dn = np.array(chronic_data[agent_name].loc[chronic_idx]["objectives_dn"])
                mu_max = np.array(chronic_data[agent_name].loc[chronic_idx]["mu_max"])

                ax_obj.plot(t, obj, linewidth=0.5, c=color, linestyle="-", label=agent_name)
                ax_obj.plot(t, obj_dn, linewidth=0.5, c=color, linestyle="--")
                ax_obj.plot(t, mu_max, linewidth=0.5, c=color, linestyle="-.")

                gain = obj_dn - obj
                markerline, stemlines, _ = ax_gain.stem(
                    t, gain, use_line_collection=True, markerfmt=f"C{color_id}o", basefmt=" ", linefmt=f"C{color_id}", label=agent_name,
                )
                plt.setp(markerline, markersize=1)
                plt.setp(stemlines, linewidth=0.5)

                for i in range(len(t)):
                    action_id = actions[i]
                    if action_id != 0:
                        ax_gain.text(t[i], gain[i], str(action_id), fontsize=2)

        ax_obj.set_xlabel("Time step t")
        ax_obj.set_ylabel("Objective value")
        ax_obj.legend()
        ax_obj.set_ylim(bottom=0.0)
        fig_obj.suptitle(f"{case_name} - Chronic {chronic_name}")

        ax_gain.set_xlabel("Time step t")
        ax_gain.set_ylabel("Gain of selected action vs. do-nothing action")
        ax_gain.legend()
        fig_gain.suptitle(f"{case_name} - Chronic {chronic_name}")

        if save_dir:
            file_name = f"agents-chronic-" + "{:05}".format(chronic_idx) + "-"
            fig_obj.savefig(os.path.join(save_dir, file_name + "objectives"))
            fig_gain.savefig(os.path.join(save_dir, file_name + "gains"))
        plt.close(fig_obj)
        plt.close(fig_gain)

    @staticmethod
    def _runner(case, env, agent, done_chronic_indices=()):
        chronics_dir, chronics, chronics_sorted = get_sorted_chronics(
            case=case, env=env
        )
        pprint("Chronics:", chronics_dir)

        np.random.seed(0)
        env.seed(0)

        chronic_data = []
        for chronic_idx, chronic in enumerate(chronics_sorted):
            if chronic_idx in done_chronic_indices:
                continue

            if case.name == "Case RTE 5":
                if chronic_idx != 0:
                    continue
            elif case.name == "Case L2RPN 2019":
                if chronic_idx != 10 and chronic_idx != 20:
                    continue
            elif case.name == "Case L2RPN 2020 WCCI":
                if chronic_idx != 0:
                    continue

            chronic_org_idx = chronics.index(chronic)
            env.chronics_handler.tell_id(chronic_org_idx - 1)  # Set chronic id

            obs = env.reset()
            agent.reset(obs=obs)
            chronic_len = env.chronics_handler.real_data.data.max_iter

            chronic_name = "/".join(
                os.path.normpath(env.chronics_handler.get_id()).split(os.sep)[-3:]
            )

            pprint("    - Chronic:", chronic_name)

            if case.name == "Case L2RPN 2020 WCCI":
                chronic_name = env.chronics_handler.get_name().split("_")
                chronic_name = "-".join([chronic_name[1][:3], chronic_name[2]])
            else:
                chronic_name = env.chronics_handler.get_name()

            done = False
            reward = 0.0
            t = 0
            actions = []
            rewards = []
            distances = []
            distances_status = []
            distances_sub = []
            time_steps = []
            objectives = []
            objectives_dn = []
            mu_max = []
            mu_max_dn = []
            mu_gen = []
            mu_gen_dn = []
            while not done:
                action, info = agent.act_with_objectives(obs, reward, done)
                obs_next, reward, done, _ = env.step(action)
                t = env.chronics_handler.real_data.data.current_index

                if t % 50 == 0:
                    pprint("Step:", t)

                if t > 500:
                    done = True

                if done:
                    pprint("        - Length:", f"{t}/{chronic_len}")

                action_id = [
                    idx
                    for idx, agent_action in enumerate(agent.actions)
                    if action == agent_action
                ]

                if "unitary_action" in agent.model_kwargs:
                    if not agent.model_kwargs["unitary_action"] and len(action_id) != 1:
                        action_id = np.nan
                    else:
                        assert (
                            len(action_id) == 1
                        )  # Exactly one action should be equivalent
                        action_id = int(action_id[0])
                else:
                    assert (
                        len(action_id) == 1
                    )  # Exactly one action should be equivalent
                    action_id = int(action_id[0])

                dist, dist_status, dist_sub = agent.distance_to_ref_topology(
                    obs_next.topo_vect, obs_next.line_status
                )

                obs = obs_next
                actions.append(action_id)
                time_steps.append(t)

                rewards.append(float(reward))
                distances.append(dist)
                distances_status.append(dist_status)
                distances_sub.append(dist_sub)

                if "obj" in info:
                    objectives.append(info["obj"])
                if "obj_dn" in info:
                    objectives_dn.append(info["obj_dn"])

                if "mu_max" in info:
                    mu_max.append(info["mu_max"])
                if "mu_max_dn" in info:
                    mu_max_dn.append(info["mu_max_dn"])

                if "mu_gen" in info:
                    mu_gen.append(info["mu_gen"])
                if "mu_gen_dn" in info:
                    mu_gen_dn.append(info["mu_gen_dn"])

            total_return = compute_returns(rewards)[0]
            chronic_data.append(
                {
                    "chronic_idx": chronic_idx,
                    "chronic_org_idx": chronic_org_idx,
                    "chronic_name": chronic_name,
                    "actions": actions,
                    "time_steps": time_steps,
                    "rewards": rewards,
                    "return": total_return,
                    "chronic_length": chronic_len,
                    "duration": t,
                    "distances": distances,
                    "distances_status": distances_status,
                    "distances_sub": distances_sub,
                    "objectives": objectives,
                    "objectives_dn": objectives_dn,
                    "mu_max": mu_max,
                    "mu_max_dn": mu_max_dn,
                    "mu_gen": mu_gen,
                    "mu_gen_dn": mu_gen_dn,
                }
            )

        if chronic_data:
            chronic_data = pd.DataFrame(chronic_data)
            chronic_data = chronic_data.set_index("chronic_idx")
        else:
            chronic_data = pd.DataFrame()

        return chronic_data
