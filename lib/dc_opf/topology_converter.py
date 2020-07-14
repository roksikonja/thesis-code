import numpy as np

from ..data_utils import hot_to_indices


class TopologyConverter:
    def __init__(self, env):
        self.env = env

        self.n_sub = self.env.n_sub
        self.n_bus = 2 * self.n_sub
        self.n_gen = self.env.n_gen
        self.n_load = self.env.n_load
        self.n_line = self.env.n_line
        self.n_topo = self.env.dim_topo
        assert self.n_topo == self.n_gen + self.n_load + 2 * self.n_line

        # Grid element positions in topology vector
        self.gen_topo_pos = self.env.gen_pos_topo_vect
        self.load_topo_pos = self.env.load_pos_topo_vect
        self.line_or_topo_pos = self.env.line_or_pos_topo_vect
        self.line_ex_topo_pos = self.env.line_ex_pos_topo_vect

        # Grid element positions in substation topology vector
        self.gen_sub_pos = self.env.gen_to_sub_pos
        self.load_sub_pos = self.env.load_to_sub_pos
        self.line_or_sub_pos = self.env.line_or_to_sub_pos
        self.line_ex_sub_pos = self.env.line_ex_to_sub_pos

        # Substation topology mask
        self.substation_topology_mask = self._get_substation_topology_mask()

    def _get_substation_element_ids(self, sub_id):
        gen_ids = hot_to_indices(self.env.gen_to_subid == sub_id)
        load_ids = hot_to_indices(self.env.load_to_subid == sub_id)
        line_or_ids = hot_to_indices(self.env.line_or_to_subid == sub_id)
        line_ex_ids = hot_to_indices(self.env.line_ex_to_subid == sub_id)
        return gen_ids, load_ids, line_or_ids, line_ex_ids

    def _get_substation_topology_mask(self):
        substation_topology_mask = np.zeros((self.n_sub, self.n_topo), dtype=np.bool)

        for sub_id, n_elements_sub in enumerate(self.env.sub_info):
            (
                gen_ids,
                load_ids,
                line_or_ids,
                line_ex_ids,
            ) = self._get_substation_element_ids(sub_id)

            sub_topo_pos = np.concatenate(
                (
                    self.line_ex_topo_pos[line_ex_ids],
                    self.line_or_topo_pos[line_or_ids],
                    self.gen_topo_pos[gen_ids],
                    self.load_topo_pos[load_ids],
                )
            )

            # Substation topology in topology vector
            sub_topo_vect_mask = np.zeros((self.n_topo,), dtype=np.bool)
            sub_topo_vect_mask[sub_topo_pos] = True

            substation_topology_mask[sub_id, :] = sub_topo_vect_mask

        assert np.equal(substation_topology_mask.sum(axis=1), self.env.sub_info).all()
        assert np.equal(substation_topology_mask.sum(axis=0), 1).all()

        return substation_topology_mask

    def _get_substation_topology_vector(self, topo_vect, sub_id):
        gen_ids, load_ids, line_or_ids, line_ex_ids = self._get_substation_element_ids(
            sub_id
        )

        # Buses within a substation
        gen_sub_bus = topo_vect[self.gen_topo_pos[gen_ids]]
        load_sub_bus = topo_vect[self.load_topo_pos[load_ids]]
        line_or_sub_bus = topo_vect[self.line_or_topo_pos[line_or_ids]]
        line_ex_sub_bus = topo_vect[self.line_ex_topo_pos[line_ex_ids]]

        # Positions within a substation
        gen_sub_pos = self.gen_sub_pos[gen_ids]
        load_sub_pos = self.load_sub_pos[load_ids]
        line_or_sub_pos = self.line_or_sub_pos[line_or_ids]
        line_ex_sub_pos = self.line_ex_sub_pos[line_ex_ids]

        sub_topo_vect = np.zeros((self.env.sub_info[sub_id],), dtype=np.int)
        sub_topo_vect[gen_sub_pos] = gen_sub_bus
        sub_topo_vect[load_sub_pos] = load_sub_bus
        sub_topo_vect[line_or_sub_pos] = line_or_sub_bus
        sub_topo_vect[line_ex_sub_pos] = line_ex_sub_bus

        assert np.equal(
            sub_topo_vect, topo_vect[self.substation_topology_mask[sub_id, :]]
        ).all()
        return sub_topo_vect

    def _construct_topology_vector(
        self, gen_sub_bus, load_sub_bus, line_or_sub_bus, line_ex_sub_bus, line_status
    ):
        line_or_sub_bus[~line_status] = -1
        line_ex_sub_bus[~line_status] = -1

        topo_vect = np.zeros((self.n_topo,), dtype=np.int)
        topo_vect[self.gen_topo_pos] = gen_sub_bus
        topo_vect[self.load_topo_pos] = load_sub_bus
        topo_vect[self.line_or_topo_pos] = line_or_sub_bus
        topo_vect[self.line_ex_topo_pos] = line_ex_sub_bus

        return topo_vect

    def _get_substation_buses(self, topo_vect):
        # Extract from topology vector sub bus of each grid element
        gen_sub_bus = topo_vect[self.gen_topo_pos]
        load_sub_bus = topo_vect[self.load_topo_pos]
        line_or_sub_bus = topo_vect[self.line_or_topo_pos]
        line_ex_sub_bus = topo_vect[self.line_ex_topo_pos]

        return gen_sub_bus, load_sub_bus, line_or_sub_bus, line_ex_sub_bus

    def _count_topology_changes(
        self, topo_vect, line_status, topo_vect_next, line_status_next, verbose=False
    ):
        n_max_sub_changed = 0
        for sub_id in range(self.n_sub):
            sub_topo_vect = self._get_substation_topology_vector(topo_vect, sub_id)
            sub_topo_vect_next = self._get_substation_topology_vector(
                topo_vect_next, sub_id
            )

            # If line change, do not count substation change
            _, _, line_or_ids, line_ex_ids = self._get_substation_element_ids(sub_id)

            for line_id in line_or_ids:
                if np.logical_xor(line_status, line_status_next)[line_id]:
                    sub_topo_vect[self.line_or_sub_pos[line_id]] = 0
                    sub_topo_vect_next[self.line_or_sub_pos[line_id]] = 0
            for line_id in line_ex_ids:
                if np.logical_xor(line_status, line_status_next)[line_id]:
                    sub_topo_vect[self.line_ex_sub_pos[line_id]] = 0
                    sub_topo_vect_next[self.line_ex_sub_pos[line_id]] = 0

            switch = np.greater(
                np.abs(sub_topo_vect - sub_topo_vect_next).sum(), 0
            ).astype(int)
            n_max_sub_changed += switch

        n_max_line_status_changed = np.logical_xor(line_status, line_status_next).sum()

        if verbose:
            print(
                "{:<35}{:<10}".format("LINE STATUS CHANGES", n_max_line_status_changed)
            )
            print(
                "{:<35}{:<10}".format("SUBSTATION TOPOLOGY CHANGES:", n_max_sub_changed)
            )

    def convert_mip_to_topology_vector(self, mip_solution):
        x_gen = mip_solution["res_x_gen"]
        x_load = mip_solution["res_x_load"]
        x_line_or_1 = mip_solution["res_x_line_or_1"]
        x_line_or_2 = mip_solution["res_x_line_or_2"]
        x_line_ex_1 = mip_solution["res_x_line_ex_1"]
        x_line_ex_2 = mip_solution["res_x_line_ex_2"]

        gen_sub_bus = -np.ones_like(x_gen, dtype=np.int)
        gen_sub_bus[~x_gen] = 1  # if x_gen[gen_id] = 0, then bus 1
        gen_sub_bus[x_gen] = 2  # if x_gen[gen_id] = 1, then bus 2
        load_sub_bus = -np.ones_like(x_load, dtype=np.int)
        load_sub_bus[~x_load] = 1  # if x_load[load_id] = 0, then bus 1
        load_sub_bus[x_load] = 2  # if x_load[load_id] = 1, then bus 2

        line_status = np.logical_and(
            np.logical_and(x_line_or_1, x_line_or_2),
            np.logical_and(x_line_ex_1, x_line_ex_2),
        )

        line_or_sub_bus = -np.ones_like(x_line_or_1, dtype=np.int)
        line_or_sub_bus[x_line_or_1] = 1
        line_or_sub_bus[x_line_or_2] = 2

        line_ex_sub_bus = -np.ones_like(x_line_ex_1, dtype=np.int)
        line_ex_sub_bus[x_line_ex_1] = 1
        line_ex_sub_bus[x_line_ex_2] = 2

        topo_vect = self._construct_topology_vector(
            gen_sub_bus, load_sub_bus, line_or_sub_bus, line_ex_sub_bus, line_status
        )

        return topo_vect, line_status