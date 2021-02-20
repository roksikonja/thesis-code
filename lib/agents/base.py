import numpy as np

from ..constants import Constants as Const
from ..dc_opf import (
    GridDCOPF,
)
from ..visualizer import pprint


class BaseAgent:
    def __init__(self, name, case):
        self.name = name

        self.case = case
        self.env = case.env

        self.grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        self.semi_agent = None

        # Grid precision
        if case.name == "Case RTE 5":
            self.grid.line["max_p_pu"] = [
                106.00150299072265625,
                38.867218017578125,
                28.26706695556640625,
                28.26706695556640625,
                106.0015106201171875,
                51.96152496337890625,
                51.96152496337890625,
                27.7128143310546875,
            ]
        elif case.name == "Case L2RPN 2019":
            self.grid.line["max_p_pu"] = [
                183.8731231689453125,
                73.7669219970703125,
                74.86887359619140625,
                65.43161773681640625,
                38.25582122802734375,
                82.47359466552734375,
                52.29061126708984375,
                54.646198272705078125,
                25.980762481689453125,
                44.721553802490234375,
                22.72505950927734375,
                17.978687286376953125,
                37.557476043701171875,
                67.636581,
                61.26263427734375,
                36.684833526611328125,
                30.32820892333984375,
                27.9899425506591796875,
                17.3205089569091796875,
                26.89875030517578125,
            ]

    def act(self, observation, reward, done=False):
        pass

    def reset(self, obs):
        pass

    def set_kwargs(self, **kwargs):
        pass

    def distance_to_ref_topology(self, topo_vect, line_status):
        """
        Count the number of unitary topological actions a topology is from the reference topology.
        The reference topology is the base case topology, fully meshed, with every line in service and a single
        electrical node, bus, per substation.
        """
        if np.equal(topo_vect, -1).all():
            return np.nan, np.nan, np.nan

        topo_vect = topo_vect.copy()
        line_status = line_status.copy()

        ref_topo_vect = np.ones_like(topo_vect)
        ref_line_status = np.ones_like(line_status)

        dist_status = 0
        for line_id, status in enumerate(line_status):
            if not status:
                line_or = self.grid.line_or_topo_pos[line_id]
                line_ex = self.grid.line_ex_topo_pos[line_id]

                # Reconnect power lines as in reference topology
                line_status[line_id] = ref_line_status[line_id]
                topo_vect[line_or] = ref_topo_vect[line_or]
                topo_vect[line_ex] = ref_topo_vect[line_ex]

                # Reconnection amounts to 1 unitary action
                dist_status = dist_status + 1

        assert not np.equal(topo_vect, -1).any()  # All element are connected

        dist_sub = 0
        for sub_id in range(self.grid.n_sub):
            sub_topology_mask = self.grid.substation_topology_mask[sub_id, :]
            sub_topo_vect = topo_vect[sub_topology_mask]
            ref_sub_topo_vect = ref_topo_vect[sub_topology_mask]

            sub_count = np.not_equal(
                sub_topo_vect, ref_sub_topo_vect
            ).sum()  # Count difference
            if sub_count > 0:
                # Reconfigure buses as in reference topology
                topo_vect[sub_topology_mask] = ref_sub_topo_vect

                # Substation bus reconfiguration amounts to 1 unitary action
                dist_sub = dist_sub + 1

        assert np.equal(
            topo_vect, ref_topo_vect
        ).all()  # Modified topology must be equal to reference topology

        dist = dist_status + dist_sub
        return dist, dist_status, dist_sub

    def print_agent(self, default=False):
        pprint("\nAgent:", self.name, shift=36)
        print("-" * 80)


def get_agent_color(agent_name):
    colors = Const.COLORS
    agent_names = [
        "do-nothing-agent",
        "mixed-mip-agent-il",
        "mixed-mip-agent-max-rho",
        "mixed-mip-agent-random",
        "agent-mip",
        "agent-multistep-mip",
        "mixed-mip-agent-k-steps",
        "agent-mip-l2rpn",
        "agent-mip-q",
    ]

    if agent_name in agent_names:
        color_id = agent_names.index(agent_name)
    else:
        color_id = len(colors) - 1

    color_id = color_id % len(colors)
    color = colors[color_id]
    return color
