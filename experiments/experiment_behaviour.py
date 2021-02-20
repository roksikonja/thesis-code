import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyPDF2 import PdfFileMerger

from lib.constants import Constants as Const
from lib.visualizer import pprint
from .experiment_base import ExperimentBase


class ExperimentBehaviour(ExperimentBase):
    def evaluate_performance(
        self,
        case,
        agent,
        save_dir=None,
        n_steps=100,
        verbose=False,
    ):
        env = case.env
        case_name = self._get_case_name(case)

        self.print_experiment("Behaviour")

        agent.print_agent(default=verbose)

        measurements = self._runner(env, agent, n_steps=n_steps, verbose=verbose)

        if verbose:
            print(
                "MEASUREMENTS:\n"
                + measurements[
                    [
                        "t",
                        "e",
                        "reward",
                        "reward-est",
                        "rho",
                        "env-rho",
                        "env-gens-p",
                        "env-loads-p",
                    ]
                ].to_string()
            )

        self._plot_and_save(
            measurements,
            env=env,
            title=f"{case_name} - {agent.name}",
            save_dir=save_dir,
            prefix=agent.name.replace(" ", "-").lower(),
        )
        self._save_csv(
            data=measurements,
            save_dir=save_dir,
            prefix=agent.name.replace(" ", "-").lower(),
        )

    @staticmethod
    def aggregate_by_agent(agent, save_dir, delete_file=False):
        merger = PdfFileMerger()
        agent_name = agent.name.replace(" ", "-").lower()
        agent_files = []
        for plot_name in [
            "rewards",
            "distance",
            "rho",
            "generators_p",
            "generators_q",
            "generators_q-p",
            "power-flows",
            "rhos",
            "production-demand",
        ]:
            file = agent_name + "-" + plot_name + ".pdf"
            if file in os.listdir(save_dir):
                f = open(os.path.join(save_dir, file), "rb")
                agent_files.append((file, f))
                merger.append(f)

        with open(
            os.path.join(save_dir, "_" + agent_name + "-performance.pdf"), "wb"
        ) as f:
            merger.write(f)

        for file, f in agent_files:
            f.close()
            try:
                if delete_file:
                    os.remove(os.path.join(save_dir, file))
            except PermissionError as e:
                print(e)

    @staticmethod
    def compare_agents(save_dir):
        measurements = dict()

        for file in os.listdir(save_dir):
            if "measurements.csv" in file:
                agent_name = file[: -len("-measurements.csv")]
                measurements[agent_name] = pd.read_csv(os.path.join(save_dir, file))

        fig_env, ax_env = plt.subplots(figsize=Const.FIG_SIZE)
        fig_est, ax_est = plt.subplots(figsize=Const.FIG_SIZE)
        fig_dist, ax_dist = plt.subplots(figsize=Const.FIG_SIZE)
        for agent_name in measurements:
            name = agent_name.replace("-", " ").capitalize()
            t = measurements[agent_name]["t"]
            ax_env.plot(t, measurements[agent_name]["reward"], label=name)
            ax_est.plot(t, measurements[agent_name]["reward-est"], label=name)
            ax_dist.plot(t, measurements[agent_name]["dist"], label=name)

        ax_env.set_xlabel("Time step t")
        ax_env.set_ylabel("Reward")
        ax_est.set_xlabel("Time step t")
        ax_est.set_ylabel("Reward")
        ax_dist.set_xlabel("Time step t")
        ax_dist.set_ylabel("Unitary action distance to reference topology")
        ax_env.legend()
        ax_est.legend()
        ax_dist.legend()
        # fig_env.suptitle("Agent Comparison - Reward ENV")
        # fig_est.suptitle("Agent Comparison - Reward EST")
        # fig_dist.suptitle("Agent Comparison - Distance to reference topology")

        if save_dir:
            file_name = "agents-"
            fig_env.savefig(os.path.join(save_dir, file_name + "rewards-env"))
            fig_est.savefig(os.path.join(save_dir, file_name + "rewards-est"))
            fig_dist.savefig(os.path.join(save_dir, file_name + "distances"))

    @staticmethod
    def _runner(env, agent, n_steps=100, verbose=False):
        np.random.seed(0)
        env.seed(0)
        env.chronics_handler.tell_id(-1)

        measurements = []

        e = 0  # Episode counter
        done = False
        reward = 0.0
        obs = env.reset()
        pprint("    - Chronic:", env.chronics_handler.get_id())
        agent.reset(obs=obs)
        for t in range(n_steps):
            action = agent.act(obs, reward, done=done)
            obs_next, reward, done, info = env.step(action)

            if t % 50 == 0 or verbose:
                pprint("Step:", env.chronics_handler.real_data.data.current_index)

            reward_est = agent.get_reward()
            res_line, res_gen = agent.compare_with_observation(obs_next, verbose=False)

            dist, dist_status, dist_sub = agent.distance_to_ref_topology(
                obs_next.topo_vect, obs_next.line_status
            )

            measurement = dict()
            measurement["t"] = t
            measurement["e"] = e
            measurement["reward"] = reward
            measurement["reward-est"] = reward_est
            measurement["dist"] = dist
            measurement["dist_status"] = dist_status
            measurement["dist_sub"] = dist_sub
            measurement["rho"] = res_line["rho"].max()
            measurement["env-rho"] = res_line["env_rho"].max()
            measurement["env-gens-p"] = obs.prod_p.sum()
            measurement["env-loads-p"] = obs.load_p.sum()

            for gen_id in res_gen.index:
                measurement[f"gen-{gen_id}"] = res_gen["p_pu"][gen_id]
                measurement[f"env-gen-{gen_id}"] = res_gen["env_p_pu"][gen_id]
                measurement[f"env-gen-{gen_id}-q"] = res_gen["env_q_pu"][gen_id]
                measurement[f"env-q-p-{gen_id}"] = res_gen["env_gen_q_p"][gen_id]

            for line_id in res_line.index:
                measurement[f"line-{line_id}"] = res_line["p_pu"][line_id]
                measurement[f"env-line-{line_id}"] = res_line["env_p_pu"][line_id]

                measurement[f"rho-{line_id}"] = res_line["rho"][line_id]
                measurement[f"env-rho-{line_id}"] = res_line["env_rho"][line_id]

            measurements.append(measurement)

            obs = obs_next
            step = env.chronics_handler.real_data.data.current_index
            if done:
                obs = env.reset()
                pprint(
                    "        - Length:",
                    f"{step}/{env.chronics_handler.real_data.data.max_iter}",
                )
                pprint("    - Done! Next chronic:", env.chronics_handler.get_id())
                agent.reset(obs=obs)
                e = e + 1

        measurements = pd.DataFrame(measurements)
        return measurements

    @staticmethod
    def _plot_and_save(
        measurements,
        env,
        title=None,
        save_dir=None,
        prefix=None,
    ):
        Const.LW = 1
        colors = Const.COLORS
        t = measurements["t"]

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        ax.plot(t, measurements["reward"], label="Reward - ENV")
        ax.plot(t, measurements["reward-est"], label="Reward - EST")
        ax.set_xlabel("Time step t")
        ax.set_ylabel("Reward")
        ax.legend()
        # fig.suptitle(title)
        fig.tight_layout()
        if save_dir:
            file_name = "rewards"
            if prefix:
                file_name = prefix + "-" + file_name

            fig.savefig(os.path.join(save_dir, file_name))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        ax.plot(t, measurements["dist"], label="Overall")
        ax.plot(t, measurements["dist_status"], label="Line")
        ax.plot(t, measurements["dist_sub"], label="Substation")
        ax.set_xlabel("Time step t")
        ax.set_ylabel("Unitary action distance to reference topology")
        ax.legend()
        # fig.suptitle(title)
        fig.tight_layout()
        if save_dir:
            file_name = "distance"
            if prefix:
                file_name = prefix + "-" + file_name

            fig.savefig(os.path.join(save_dir, file_name))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        ax.plot(t, measurements["env-rho"], label="Rho - ENV")
        ax.plot(t, measurements["rho"], label="Rho - EST")
        ax.plot(
            t,
            np.ones_like(t),
            c="tab:red",
            linestyle="-",
            linewidth=Const.LW,
        )
        ax.set_xlabel("Time step t")
        ax.set_ylabel(r"$\rho^\mathrm{max}$")
        ax.set_ylim((0.0, 2.0))
        ax.legend()
        # fig.suptitle(title)
        fig.tight_layout()
        if save_dir:
            file_name = "rho"
            if prefix:
                file_name = prefix + "-" + file_name

            fig.savefig(os.path.join(save_dir, file_name))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        for gen_id in range(env.n_gen):
            color = colors[gen_id % len(colors)]
            ax.plot(
                t,
                measurements[f"gen-{gen_id}"] / env.gen_pmax[gen_id],
                # measurements[f"gen-{gen_id}"],
                label=f"Gen-{gen_id}",
                c=color,
                linestyle="-",
                linewidth=Const.LW,
            )
            ax.plot(
                t,
                measurements[f"env-gen-{gen_id}"] / env.gen_pmax[gen_id],
                # measurements[f"env-gen-{gen_id}"],
                label=f"Gen-{gen_id} - ENV",
                c=color,
                linestyle="--",
                linewidth=Const.LW,
            )

        ax.set_xlabel("Time step t")
        # ax.set_ylabel("P [p.u.]")
        ax.set_ylabel(r"$\rho_g$")

        # fig.suptitle(title)
        if env.n_gen < 6:
            ax.legend()

        fig.tight_layout()
        if save_dir:
            file_name = "generators_p"
            if prefix:
                file_name = prefix + "-" + file_name

            fig.savefig(os.path.join(save_dir, file_name))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        for gen_id in range(env.n_gen):
            color = colors[gen_id % len(colors)]
            ax.plot(
                t,
                measurements[f"env-gen-{gen_id}-q"],
                label=f"Gen-{gen_id} - ENV",
                c=color,
                linestyle="--",
                linewidth=Const.LW,
            )

        ax.set_xlabel("Time step t")
        ax.set_ylabel("Q [p.u.]")
        # fig.suptitle(title)
        if env.n_gen < 3:
            ax.legend()
        fig.tight_layout()
        if save_dir:
            file_name = "generators_q"
            if prefix:
                file_name = prefix + "-" + file_name

            fig.savefig(os.path.join(save_dir, file_name))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        for gen_id in range(env.n_gen):
            color = colors[gen_id % len(colors)]
            ax.plot(
                t,
                measurements[f"env-q-p-{gen_id}"],
                label=f"Gen-{gen_id} - ENV",
                c=color,
                linestyle="--",
                linewidth=Const.LW,
            )

        ax.set_xlabel("Time step t")
        ax.set_ylabel("Q/P [p.u.]")
        # fig.suptitle(title)
        if env.n_gen < 3:
            ax.legend()
        fig.tight_layout()
        if save_dir:
            file_name = "generators_q-p"
            if prefix:
                file_name = prefix + "-" + file_name

            fig.savefig(os.path.join(save_dir, file_name))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        for line_id in range(env.n_line):
            color = colors[line_id % len(colors)]
            ax.plot(
                t,
                measurements[f"line-{line_id}"],
                label=f"Line-{line_id}",
                c=color,
                linestyle="-",
                linewidth=Const.LW,
            )
            ax.plot(
                t,
                measurements[f"env-line-{line_id}"],
                label=f"Line-{line_id} - ENV",
                c=color,
                linestyle="--",
                linewidth=Const.LW,
            )

        ax.set_xlabel("Time step t")
        ax.set_ylabel("P [p.u.]")
        # fig.suptitle(title)
        fig.tight_layout()
        if save_dir:
            file_name = "power-flows"
            if prefix:
                file_name = prefix + "-" + file_name

            fig.savefig(os.path.join(save_dir, file_name))
        plt.close(fig)
        fig.tight_layout()
        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        for line_id in range(env.n_line):
            color = colors[line_id % len(colors)]
            ax.plot(
                t,
                measurements[f"rho-{line_id}"],
                label=f"Line-{line_id}",
                c=color,
                linestyle="-",
                linewidth=Const.LW,
            )
            ax.plot(
                t,
                measurements[f"env-rho-{line_id}"],
                label=f"Line-{line_id} - ENV",
                c=color,
                linestyle="--",
                linewidth=Const.LW,
            )

        ax.set_xlabel("Time step t")
        ax.set_ylabel(r"$\rho$ [p.u.]")
        # fig.suptitle(title)
        fig.tight_layout()
        if save_dir:
            file_name = "rhos"
            if prefix:
                file_name = prefix + "-" + file_name

            fig.savefig(os.path.join(save_dir, file_name))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        ax.plot(
            t,
            measurements["env-gens-p"],
            label="Total production",
            linestyle="--",
            linewidth=Const.LW,
        )
        ax.plot(
            t,
            measurements["env-loads-p"],
            label="Total demand",
            linestyle="--",
            linewidth=Const.LW,
        )
        ax.set_xlabel("Time step t")
        ax.set_ylabel("P [p.u.]")
        ax.set_ylim(bottom=0.0)
        ax.legend()
        # fig.suptitle(title)
        fig.tight_layout()
        if save_dir:
            file_name = "production-demand"
            if prefix:
                file_name = prefix + "-" + file_name

            fig.savefig(os.path.join(save_dir, file_name))
        plt.close(fig)

    @staticmethod
    def _save_csv(data, save_dir, prefix=None):
        file_name = "measurements"
        if prefix:
            file_name = prefix + "-" + file_name

        data.to_csv(os.path.join(save_dir, file_name + ".csv"))
