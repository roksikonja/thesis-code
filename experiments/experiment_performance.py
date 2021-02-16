import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from PyPDF2 import PdfFileMerger

from experience import ExperienceCollector
from lib.chronics import get_sorted_chronics
from lib.constants import Constants as Const
from lib.rl_utils import compute_returns
from lib.agents import get_agent_color
from lib.visualizer import pprint
from .experiment_base import ExperimentBase


class ExperimentPerformance(ExperimentBase):
    def __init__(self, save_dir):
        self.collector = ExperienceCollector(save_dir=save_dir)

    def analyse(
        self,
        case,
        agent,
        do_chronics=(),
        n_chronics=-1,
        n_steps=-1,
        verbose=False,
    ):
        env = case.env

        self.collector.full_reset()
        self._runner(
            env,
            agent,
            do_chronics=do_chronics,
            n_chronics=n_chronics,
            n_steps=n_steps,
            verbose=verbose,
        )

    def compare_agents(self, case, save_dir=None, plot=True, delete_file=True):
        case_name = self._get_case_name(case)
        chronic_data = dict()

        agent_names = np.unique(
            [
                file.split("-chronic-")[0]
                for file in os.listdir(save_dir)
                if "-chronic-" in file and "agents" not in file
            ]
        )

        chronic_indices_all = []
        for agent_name in agent_names:
            self.collector.load_data(agent_name, case.env)
            chronic_indices_all.extend(self.collector.chronic_ids)
            chronic_data[agent_name] = copy.deepcopy(self.collector.data)

        chronic_indices_all = np.unique(chronic_indices_all).tolist()

        if plot:
            for chronic_idx in chronic_indices_all:
                self._plot_rewards(chronic_data, case_name, chronic_idx, save_dir)

                self._plot_max_rho(chronic_data, case_name, chronic_idx, save_dir)
                self._plot_line_loading(chronic_data, case_name, chronic_idx, save_dir)

                for dist, ylabel in [
                    ("distances", r"$d(\tau, \tau^\mathrm{ref})$"),
                    ("distances_line", r"$d_\mathcal{P}(\tau, \tau^\mathrm{ref})$"),
                    ("distances_sub", r"$d_\mathcal{S}(\tau, \tau^\mathrm{ref})$"),
                ]:
                    self._plot_distances(
                        chronic_data,
                        dist,
                        ylabel,
                        case_name,
                        chronic_idx,
                        save_dir,
                    )

            self._plot_durations(chronic_data, chronic_indices_all, case_name, save_dir)

            self._plot_returns(chronic_data, chronic_indices_all, case_name, save_dir)

            self._plot_loading_distribution(chronic_data, case_name, save_dir)

            self.aggregate_by_chronics(save_dir, delete_file=delete_file)

        return chronic_data

    def aggregate_by_chronics(self, save_dir, delete_file=True):
        for plot_name in [
            "rewards",
            "distances",
            "distances_line",
            "distances_sub",
            "advantages",
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

        agent_names = np.unique(
            [
                file.split("-chronic-")[0]
                for file in os.listdir(save_dir)
                if "-chronic-" in file
            ]
        )
        for agent_name in agent_names:
            for plot_name in ["line-loading", "dist-loading"]:
                merger = PdfFileMerger()
                chronic_files = []
                for file in os.listdir(save_dir):
                    if (
                        "-chronic-" in file
                        and agent_name in file
                        and plot_name + ".pdf" in file
                    ):
                        f = open(os.path.join(save_dir, file), "rb")
                        chronic_files.append((file, f))
                        merger.append(f)

                if merger.inputs:
                    with open(
                        os.path.join(
                            save_dir, "_" + f"{agent_name}-chronics-{plot_name}.pdf"
                        ),
                        "wb",
                    ) as f:
                        merger.write(f)

                    # Reset merger
                    merger.pages = []
                    merger.inputs = []
                    merger.output = None

                self.close_files(chronic_files, save_dir, delete_file=delete_file)

    @staticmethod
    def _plot_rewards(chronic_data, case_name, chronic_idx, save_dir=None):
        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)

        chronic_name = str(chronic_idx)
        for agent_name in chronic_data:
            agent_data = chronic_data[agent_name]

            if chronic_idx in agent_data:
                rewards = agent_data[chronic_idx]["rewards"]
                t = np.arange(len(rewards))

                color = get_agent_color(agent_name)
                ax.plot(t, rewards, linewidth=1, label=agent_name, color=color)

                if agent_data[chronic_idx]["chronic_name"]:
                    chronic_name = agent_data[chronic_idx]["chronic_name"]

        ax.set_xlabel("Time step t")
        ax.set_ylabel("Reward")
        ax.legend()
        # fig.suptitle(f"{case_name} - Chronic {chronic_name}")
        fig.tight_layout()
        if save_dir:
            file_name = f"agents-chronic-" + "{:05}".format(chronic_idx) + "-"
            fig.savefig(os.path.join(save_dir, file_name + "rewards"))
        plt.close(fig)

    @staticmethod
    def _plot_max_rho(chronic_data, case_name, chronic_idx, save_dir=None):
        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)

        chronic_name = str(chronic_idx)
        for agent_name in chronic_data:
            agent_data = chronic_data[agent_name]

            if chronic_idx in agent_data:
                obses = agent_data[chronic_idx]["obses"][:-1]
                rho = [np.max(obs.rho) for obs in obses]
                t = np.arange(len(obses))

                color = get_agent_color(agent_name)
                ax.plot(t, rho, linewidth=1, label=agent_name, color=color)

                if agent_data[chronic_idx]["chronic_name"]:
                    chronic_name = agent_data[chronic_idx]["chronic_name"]

        ax.set_xlabel("Time step t")
        ax.set_ylabel(r"Relative flow $\rho$")
        ax.legend()
        ax.set_ylim([0.0, 2.0])
        # fig.suptitle(f"{case_name} - Chronic {chronic_name}")
        fig.tight_layout()
        if save_dir:
            file_name = f"agents-chronic-" + "{:05}".format(chronic_idx) + "-"
            fig.savefig(os.path.join(save_dir, file_name + "rhos"))
        plt.close(fig)

    @staticmethod
    def _plot_line_loading(chronic_data, case_name, chronic_idx, save_dir=None):

        for agent_name in chronic_data:
            agent_data = chronic_data[agent_name]

            if chronic_idx in agent_data:
                fig, ax = plt.subplots(figsize=Const.FIG_SIZE)

                obses = agent_data[chronic_idx]["obses"][:-1]
                rhos = np.vstack([obs.rho for obs in obses])
                t = np.arange(len(obses))

                for line_id in range(rhos.shape[1]):
                    rho = rhos[:, line_id]
                    ax.plot(t, rho, linewidth=0.5)

                chronic_name = agent_data[chronic_idx]["chronic_name"]

                ax.set_xlabel("Time step t")
                ax.set_ylabel(r"$\rho$")
                ax.set_ylim(bottom=0)
                # fig.suptitle(f"{case_name} - Chronic {chronic_name}")
                fig.tight_layout()
                if save_dir:
                    file_name = (
                        f"{agent_name}-chronic-" + "{:05}".format(chronic_idx) + "-"
                    )
                    fig.savefig(os.path.join(save_dir, file_name + "line-loading"))
                plt.close(fig)

    @staticmethod
    def _plot_distances(
        chronic_data, dist, ylabel, case_name, chronic_idx, save_dir=None
    ):
        plot = False
        for agent_name in chronic_data:
            if chronic_idx in chronic_data[agent_name]:
                plot = plot or bool(len(chronic_data[agent_name][chronic_idx][dist]))

        if plot:
            fig, ax = plt.subplots(figsize=Const.FIG_SIZE)

            chronic_name = str(chronic_idx)
            for agent_name in chronic_data:
                agent_data = chronic_data[agent_name]

                if chronic_idx in agent_data:
                    distances = agent_data[chronic_idx][dist]
                    t = np.arange(len(distances))

                    color = get_agent_color(agent_name)
                    ax.plot(t, distances, linewidth=0.5, label=agent_name, color=color)

                    if agent_data[chronic_idx]["chronic_name"]:
                        chronic_name = agent_data[chronic_idx]["chronic_name"]

            ax.set_xlabel("Time step t")
            ax.set_ylabel(ylabel)
            ax.legend()
            # fig.suptitle(f"{case_name} - Chronic {chronic_name}")

            ax.set_xlim(left=-10, right=500)
            fig.tight_layout()
            if save_dir:
                file_name = f"agents-chronic-" + "{:05}".format(chronic_idx) + "-"
                fig.savefig(os.path.join(save_dir, file_name + dist))
            plt.close(fig)

    @staticmethod
    def _plot_durations(chronic_data, chronic_indices_all, case_name, save_dir=None):
        fig, ax = plt.subplots(figsize=(8, 8))
        width = 0.3 / len(chronic_data.keys())

        x_all = np.arange(len(chronic_indices_all))

        chronic_names_all = [""] * len(chronic_indices_all)
        chronic_lengths_all = [0] * len(chronic_indices_all)
        for agent_id, agent_name in enumerate(chronic_data):
            agent_data = chronic_data[agent_name]

            agent_chronic_indices = [chronic_idx for chronic_idx in agent_data]

            for chronic_idx in agent_chronic_indices:
                idx = chronic_indices_all.index(chronic_idx)
                chronic_name = str(agent_data[chronic_idx]["chronic_name"])
                chronic_len = agent_data[chronic_idx]["chronic_len"]
                chronic_names_all[idx] = chronic_name
                chronic_lengths_all[idx] = chronic_len

            x = []
            for x_, chronic_idx in zip(x_all, chronic_indices_all):
                if chronic_idx in agent_data:
                    x.append(x_)
            x = np.array(x)

            y = [agent_data[chronic_idx]["duration"] for chronic_idx in agent_data]

            color = get_agent_color(agent_name)
            ax.barh(
                x + agent_id * width,
                y,
                width,
                left=0.001,
                label=agent_name,
                color=color,
            )

        if len(chronic_indices_all) < 25:
            # ax.scatter(chronic_lengths_all, x_all, marker="|", c="black")
            ax.set_yticks(x_all)

            chronic_names_all = [str(i + 1) for i in range(len(chronic_names_all))]
            ax.set_yticklabels(chronic_names_all)

        ax.invert_yaxis()
        ax.legend()

        ax.set_ylabel("Scenario")
        ax.set_xlabel("Duration")
        # fig.suptitle(f"{case_name} - Chronic durations")

        fig.tight_layout()
        if save_dir:
            file_name = f"_agents-chronics-"
            fig.savefig(os.path.join(save_dir, file_name + "durations"))
        plt.close(fig)

    @staticmethod
    def _plot_returns(chronic_data, chronic_indices_all, case_name, save_dir=None):
        fig, ax = plt.subplots(figsize=(8, 8))
        width = 0.3 / len(chronic_data.keys())

        x_all = np.arange(len(chronic_indices_all))

        chronic_names_all = [""] * len(chronic_indices_all)
        for agent_id, agent_name in enumerate(chronic_data):
            agent_data = chronic_data[agent_name]

            agent_chronic_indices = [chronic_idx for chronic_idx in agent_data]

            for chronic_idx in agent_chronic_indices:
                idx = chronic_indices_all.index(chronic_idx)
                chronic_name = str(agent_data[chronic_idx]["chronic_name"])
                chronic_names_all[idx] = chronic_name

            x = []
            for x_, chronic_idx in zip(x_all, chronic_indices_all):
                if chronic_idx in agent_data:
                    x.append(x_)
            x = np.array(x)

            y = [agent_data[chronic_idx]["total_return"] for chronic_idx in agent_data]

            color = get_agent_color(agent_name)
            ax.barh(
                x + agent_id * width,
                y,
                width,
                left=0.001,
                label=agent_name,
                color=color,
            )

        if len(chronic_indices_all) < 25:
            # ax.scatter(chronic_lengths_all, x_all, marker="|", c="black")
            ax.set_yticks(x_all)

            chronic_names_all = [str(i + 1) for i in range(len(chronic_names_all))]
            ax.set_yticklabels(chronic_names_all)

        ax.set_yticklabels(chronic_names_all)
        ax.invert_yaxis()
        ax.legend()

        ax.set_ylabel("Scenario")
        ax.set_xlabel("Return")
        # fig.suptitle(f"{case_name} - Chronic returns")
        fig.tight_layout()

        if save_dir:
            file_name = f"_agents-chronics-"
            fig.savefig(os.path.join(save_dir, file_name + "returns"))
        plt.close(fig)

    def _plot_loading_distribution(self, chronic_data, case_name, save_dir=None):

        for agent_name in chronic_data:
            agent_data = chronic_data[agent_name]

            rhos = []
            for chronic_idx in agent_data:
                obses_chronic = agent_data[chronic_idx]["obses"][:-1]
                rhos_chronic = np.vstack([obs.rho for obs in obses_chronic])
                rhos.extend(rhos_chronic)

                means_chronic = rhos_chronic.mean(axis=0)
                max_ids_chronic = np.sort(np.argsort(means_chronic)[-4:])
                rhos_chronic_max = pd.DataFrame(
                    rhos_chronic[:, max_ids_chronic],
                    columns=max_ids_chronic.astype(str),
                )

                self.__plot_loading_distribution(
                    rhos_chronic_max, case_name, agent_name, chronic_idx, save_dir
                )

            rhos = np.vstack(rhos)
            means = rhos.mean(axis=0)
            max_ids = np.sort(np.argsort(means)[-4:])
            rhos = pd.DataFrame(rhos[:, max_ids], columns=max_ids.astype(str))

            fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
            # sns.histplot(data=rhos, ax=ax)
            sns.kdeplot(data=rhos, ax=ax)
            ax.set_xlabel(r"$\rho$")
            ax.set_ylabel(r"PDF")
            ax.set_xlim([0.0, 2.0])
            # fig.suptitle(f"{case_name} - {agent_name}")

            fig.tight_layout()
            if save_dir:
                file_name = f"_{agent_name}-chronics-"
                fig.savefig(os.path.join(save_dir, file_name + "dist-all-loading"))
            plt.close(fig)

    @staticmethod
    def __plot_loading_distribution(
        rhos, case_name, agent_name, chronic_idx, save_dir=None
    ):
        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        # sns.histplot(data=rhos, ax=ax)
        sns.kdeplot(data=rhos, ax=ax)
        ax.set_xlabel(r"$\rho$")
        # ax.set_ylabel(r"Counts")
        ax.set_ylabel("PDF")
        ax.set_xlim([0.0, 2.0])
        # ax.set_title(f"Chronic {chronic_idx}")
        # fig.suptitle(f"{case_name} - {agent_name}")

        fig.tight_layout()
        if save_dir:
            file_name = f"{agent_name}-chronic-" + "{:05}".format(chronic_idx) + "-"
            fig.savefig(os.path.join(save_dir, file_name + "dist-loading"))
        plt.close(fig)

    def _runner(
        self, env, agent, do_chronics=(), n_chronics=-1, n_steps=-1, verbose=False
    ):
        self.print_experiment("Performance")
        agent.print_agent(default=verbose)

        agent_name = agent.name.replace(" ", "-").lower()
        self.collector._load_chronics(agent_name=agent_name)

        chronics_dir, chronics, chronics_sorted = get_sorted_chronics(env=env)
        pprint("Chronics:", chronics_dir)

        if len(self.collector.chronic_ids):
            pprint(
                "    - Done chronics:",
                ", ".join(map(lambda x: str(x), sorted(self.collector.chronic_ids))),
            )

        if len(do_chronics):
            pprint(
                "    - To do chronics:",
                ", ".join(map(lambda x: str(x), sorted(do_chronics))),
            )

        done_chronic_ids = []
        for chronic_idx, chronic_name in enumerate(chronics_sorted):
            if len(done_chronic_ids) >= n_chronics >= 0:
                break

            # If chronic already done
            if chronic_idx in self.collector.chronic_ids:
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

            augmentation_info = os.path.join(
                env.chronics_handler.get_id(), "augmentation.json"
            )
            ps = None
            if os.path.isfile(augmentation_info):
                with open(augmentation_info, "r") as f:
                    ps = json.load(f)

            pprint("    - Chronic:", chronic_path_name)
            # if ps:
            #     p = ps["p"]
            #     min_p = ps["min_p"]
            #     max_p = ps["max_p"]
            #     targets = ps["targets"]
            #
            #     pprint("        - Augmentation:", ps["augmentation"])
            #     pprint(
            #         "            - Rate:",
            #         "p = {:.2f} ~ [{:.2f}, {:.2f}]".format(p, min_p, max_p),
            #     )
            #     if targets:
            #         pprint("            - Targets:", str(targets))

            t = 0
            done = False
            reward = np.nan

            """
                Collect data.
            """
            while not done and not (t >= n_steps > 0):
                action = agent.act(obs, reward=reward, done=done)
                obs_next, reward, done, info = env.step(action)
                self.collector._add(obs, action, reward, done)

                semi_action = False
                if agent.semi_agent is not None:
                    semi_action = agent.semi_agent.semi_action

                dist, dist_status, dist_status = agent.distance_to_ref_topology(
                    obs_next.topo_vect, obs_next.line_status
                )
                self.collector._add_plus(dist, dist_status, dist_status, semi_action)

                t = env.chronics_handler.real_data.data.current_index

                if t % 200 == 0:
                    pprint("        - Step:", t)

                if done:
                    pprint("        - Length:", f"{t}/{chronic_len}")

                obs = obs_next

            self.collector.obses.append(obs.to_vect())
            self.collector.total_return = compute_returns(self.collector.rewards)[0]
            self.collector.duration = t
            self.collector.chronic_len = chronic_len
            self.collector.chronic_name = chronic_name

            done_chronic_ids.append(chronic_idx)

            self.collector._save_chronic(agent_name, chronic_idx, verbose)
            self.collector.reset()
