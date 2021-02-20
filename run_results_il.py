import os
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from experience import load_agent_experience
from lib.action_space import is_do_nothing_action
from lib.agents import get_agent_color
from lib.constants import Constants as Const
from lib.data_utils import make_dir, env_pf, stacked_three_bar
from lib.dc_opf import load_case, CaseParameters

experience_dir = make_dir(os.path.join(Const.RESULTS_DIR, "performance-aug-il"))
results_dir = experience_dir

case_name = "l2rpn_2019_art"
env_dc = True
verbose = False

case_name_dc = f"{case_name}-{env_pf(env_dc)}"
case_results_dir = make_dir(os.path.join(results_dir, case_name_dc))

parameters = CaseParameters(case_name=case_name, env_dc=env_dc)
case = load_case(case_name, env_parameters=parameters)
env = case.env

agent_names = [
    "mixed-mip-agent-random",
    # "mixed-mip-agent-k-steps",
    "mixed-mip-agent-max-rho",
    "mixed-mip-agent-il",
    "do-nothing-agent",
]

labels = {
    "mixed-mip-agent-random": "rnd",
    "mixed-mip-agent-k-steps": "k",
    "mixed-mip-agent-max-rho": "rho",
    "mixed-mip-agent-il": "il",
    "do-nothing-agent": "dn",
}

"""
    Load data
"""

agents_collectors = dict()
for agent_name in agent_names:
    collector = load_agent_experience(
        env, agent_name, os.path.join(experience_dir, case_name_dc)
    )
    agents_collectors[agent_name] = collector

"""
    Chronics
"""

rewards = defaultdict(list)
semi_actions = defaultdict(list)
actions = defaultdict(list)
for agent_name in agent_names:
    collector = agents_collectors[agent_name]
    for chronic_idx in collector.chronic_ids:
        data = collector.data[chronic_idx]
        rewards[chronic_idx].append((agent_name, data["rewards"]))
        semi_actions[chronic_idx].append((agent_name, data["semi_actions"]))
        actions[chronic_idx].append((agent_name, data["actions"]))


"""
    Plot
"""

plt.style.use("seaborn-white")
plt.rcParams["font.family"] = "pcr"
plt.rcParams["axes.grid"] = False
plt.rcParams["font.size"] = 20
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["legend.title_fontsize"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["axes.labelsize"] = 20

chronic_ids = [i for i in sorted(semi_actions.keys()) if 110 <= i < 120]

# for chronic_idx in chronic_ids:
#     fig, ax = plt.subplots()
#     for agent_name, agent_rewards in rewards[chronic_idx]:
#         agent_label = labels[agent_name]
#         t = np.arange(agent_rewards.size)
#         ax.plot(t, agent_rewards, label=agent_label)
#     # ax.legend("lower right")
#     ax.set_xlabel(r"Reward $r$")
#     ax.set_xlabel(r"Time step $t$")
#     fig.tight_layout()
#     fig.show()


fig, ax = plt.subplots()
width = 0.6 / len(agent_names)
xs = []
x_ticks = []
for x, chronic_idx in enumerate(chronic_ids):
    xs.append(x)
    x_ticks.append(str(chronic_idx))
    for i, (agent_name, agent_semi_actions) in enumerate(semi_actions[chronic_idx]):
        actions_bool = is_do_nothing_action(
            actions[chronic_idx][i][-1], env, dtype=np.bool
        )

        color = get_agent_color(agent_name)
        stacked_three_bar(
            x + i * width - width / 2,
            agent_semi_actions,
            actions_bool,
            ax,
            height=width,
            color=color,
            alpha_min=0.3,
        )

legend_handles = []
for agent_name in agent_names:
    color = get_agent_color(agent_name)
    agent_label = labels[agent_name]
    legend_handles.append(mpatches.Patch(color=color, label=agent_label))
ax.legend(handles=legend_handles, loc="lower right")

ax.set_yticks(xs)
ax.set_yticklabels(x_ticks)
ax.invert_yaxis()
ax.set_ylabel(r"Scenario")
ax.set_xlabel(r"Time step $t$")
fig.tight_layout()
fig.show()
fig.savefig(os.path.join(case_results_dir, "semi_agents_scenarios"))

"""
    Agents
"""

durations = defaultdict(list)
actions = defaultdict(list)
semi_actions = defaultdict(list)

for agent_name in agent_names:
    agent_label = labels[agent_name]
    collector = agents_collectors[agent_name]
    for chronic_idx in collector.chronic_ids:
        data = collector.data[chronic_idx]

        durations[agent_label].append(int(data["duration"]))

        if agent_name != "do-nothing-agent":
            semi_actions_frac = float(np.mean(data["semi_actions"]))
            actions_frac = float(np.mean(is_do_nothing_action(data["actions"], env)))

            semi_actions[agent_label].append(semi_actions_frac)
            actions[agent_label].append(actions_frac / semi_actions_frac)


palette = {labels[name]: get_agent_color(name) for name in agent_names}

fig, ax = plt.subplots()
sns.histplot(
    data=durations,
    ax=ax,
    element="step",
    fill=True,
    # bins=20,
    legend=True,
    palette=palette,
)
ax.set_xlim(left=0, right=1000)
fig.tight_layout()
fig.show()
fig.savefig(os.path.join(case_results_dir, "semi_agents_durations"))

fig, ax = plt.subplots()
sns.histplot(
    data=semi_actions,
    ax=ax,
    element="step",
    fill=True,
    legend=True,
    palette=palette,
)
ax.set_xlim(left=0.0, right=1.0)
fig.tight_layout()
fig.show()
fig.savefig(os.path.join(case_results_dir, "semi_agents_semi_actions"))

fig, ax = plt.subplots()
sns.histplot(
    data=actions,
    ax=ax,
    element="step",
    fill=True,
    legend=True,
    palette=palette,
)
ax.set_xlim(left=0.0, right=1.0)
fig.tight_layout()
fig.show()
fig.savefig(os.path.join(case_results_dir, "semi_agents_actions"))


"""
    Table
"""

line_list = [
    "{:<20}".format("subagent"),
    "{:<10}".format("Scenarios"),
    "{:<15}".format("Durations"),
    "{:<15}".format("Fraction SA"),
    "{:<15}".format("Fraction A | SA"),
]
line_str = " & ".join(line_list)
print(line_str + r"\\")

for agent_name in agent_names:
    agent_label = labels[agent_name]

    durs = durations[agent_label]
    n_scenarios = len(durs)
    n_finished = np.sum(np.equal(durs, 1000))

    dur_mean, dur_std = np.mean(durs), np.std(durs)

    a_mean, a_std = np.mean(actions[agent_label]), np.std(actions[agent_label])
    sa_mean, sa_std = np.mean(semi_actions[agent_label]), np.std(
        semi_actions[agent_label]
    )

    line_list = [
        "{:<20}".format(agent_label),
        "{} / {:<5}".format(n_finished, n_scenarios),
        "{:.2f} \\pm {:<5.2f}".format(dur_mean, dur_std),
        "{:.2f} \\pm {:<5.2f}".format(sa_mean, sa_std),
        "{:.2f} \\pm {:<5.2f}".format(a_mean, a_std),
    ]
    line_str = " & ".join(line_list)

    print(line_str + r"\\")
