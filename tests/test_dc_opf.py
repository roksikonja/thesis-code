import itertools
import sys
import time
import unittest

import numpy as np
import pandapower as pp
import pandas as pd

from lib.data_utils import indices_to_hot, hot_to_indices
from lib.dc_opf import (
    StandardDCOPF,
    LineSwitchingDCOPF,
    TopologyOptimizationDCOPF,
    MultistepTopologyDCOPF,
    GridDCOPF,
    load_case,
    StandardParameters,
    LineSwitchingParameters,
    SinglestepTopologyParameters,
    MultistepTopologyParameters,
    ForecastsPlain,
)


class TestStandardDCOPF(unittest.TestCase):
    """
    Test standard DC-OPF implementation.
    """

    @classmethod
    def setUpClass(cls):
        print("\nStandard DC-OPF tests.\n")

    def runner_opf(self, model, n_tests=20, eps=1e-4, verbose=False):
        conditions = list()
        for i in range(n_tests):
            np.random.seed(i)
            model.gen["cost_pu"] = np.random.uniform(
                1.0, 5.0, (model.grid.gen.shape[0],)
            )

            model.build_model()
            result = model.solve_and_compare(verbose=verbose)

            conditions.append(
                {
                    "cost": np.less_equal(result["res_cost"]["diff"], eps).all(),
                    "bus": np.less_equal(result["res_bus"]["diff"], eps).all(),
                    "line": np.less_equal(result["res_line"]["diff"], eps).all(),
                    "gen": np.less_equal(result["res_gen"]["diff"], eps).all(),
                    "load": np.less_equal(result["res_load"]["diff"], eps).all(),
                    "ext_grid": np.less_equal(
                        result["res_ext_grid"]["diff"], eps
                    ).all(),
                    "trafo": np.less_equal(result["res_trafo"]["diff"], eps).all(),
                    "line_loading": np.less_equal(
                        result["res_line"]["diff_loading"], 100 * eps
                    ).all(),
                    "trafo_loading": np.less_equal(
                        result["res_trafo"]["diff_loading"], 100 * eps
                    ).all(),
                }
            )

        conditions = pd.DataFrame(conditions)
        conditions["passed"] = np.all(conditions.values, axis=-1)

        print(f"\n\n{model.name}\n")
        print(conditions.to_string())

        time.sleep(0.1)
        # Test DC Power Flow
        self.assertTrue(conditions["passed"].values.all())

    def test_case3(self):
        case = load_case("case3")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        params = StandardParameters(tol=1e-9)
        model = StandardDCOPF(
            f"{case.name} Standard DC OPF",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
            params=params,
        )

        self.runner_opf(model, verbose=False)

    def test_case4(self):
        case = load_case("case4")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        params = StandardParameters(tol=1e-9)
        model = StandardDCOPF(
            f"{case.name} Standard DC OPF",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
            params=params,
        )

        self.runner_opf(model, verbose=False)

    def test_case6(self):
        case = load_case("case6")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        params = StandardParameters(tol=1e-9)
        model = StandardDCOPF(
            f"{case.name} Standard DC OPF",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
            params=params,
        )

        self.runner_opf(model, verbose=False)

    def test_case3_by_value(self):
        """
        Test for power flow computation.
        """
        case = load_case("case3")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        params = StandardParameters(tol=1e-9)
        model = StandardDCOPF(
            f"{case.name} Standard DC OPF",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
            params=params,
        )

        model.build_model()
        result = model.solve(verbose=False)
        model.print_results()

        time.sleep(0.1)
        # Test DC Power Flow
        self.assertTrue(
            np.equal(
                result["res_bus"]["delta_pu"].values,
                np.array([0.0, -0.250, -0.375, 0.0, 0.0, 0.0]),
            ).all()
        )

    def test_rte_case5(self):
        case = load_case("rte_case5_example")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        params = StandardParameters(tol=1e-9)
        model = StandardDCOPF(
            f"{case.name} Standard DC OPF",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
            params=params,
        )

        self.runner_opf(model, verbose=False)

    def test_l2rpn2019(self):
        case = load_case("l2rpn_2019")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        params = StandardParameters(tol=1e-9)
        model = StandardDCOPF(
            f"{case.name} Standard DC OPF",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
            params=params,
        )

        self.runner_opf(model, verbose=False)

    # def test_l2rpn2020(self):
    #     if sys.platform != "win32":
    #         print("L2RPN 2020 not available.")
    #         self.assertTrue(True)
    #         return
    #
    #     case = load_case("l2rpn_wcci_2020")
    #     grid = GridDCOPF(
    #         case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
    #     )
    #
    #     params = StandardParameters(tol=1e-9)
    #     model = StandardDCOPF(
    #         f"{case.name} Standard DC OPF",
    #         grid=grid,
    #         grid_backend=case.grid_backend,
    #         base_unit_p=case.base_unit_p,
    #         base_unit_v=case.base_unit_v,
    #         params=params,
    #     )
    #
    #     self.runner_opf(model, eps=1e-3, verbose=False)


class TestLineSwitchingDCOPF(unittest.TestCase):
    """
    Test DC-OPF with line status switching implementation.
    """

    @classmethod
    def setUpClass(cls):
        print("\nDC-OPF with line switching tests.\n")

    def runner_opf_line_switching(
        self,
        model,
        grid,
        verbose=False,
    ):
        np.random.seed(0)
        model.gen["cost_pu"] = np.random.uniform(1.0, 5.0, (model.grid.gen.shape[0],))
        model.build_model()

        if verbose:
            model.print_model()

        """
            BACKEND BRUTE FORCE.
        """

        # Construct all possible configurations
        line_statuses = list()
        for i in range(
            model.params.n_max_line_status_changed + 1
        ):  # Number of line disconnection 0, 1, ..., n
            line_statuses.extend(
                [
                    ~indices_to_hot(
                        list(line_status),
                        length=grid.line.shape[0],
                        dtype=np.bool,
                    )
                    for line_status in itertools.combinations(grid.line.index, i)
                ]
            )

        results_backend = pd.DataFrame(
            columns=["status", "objective", "loads_p", "generators_p", "valid"]
        )

        for idx, status in enumerate(line_statuses):
            model.grid_backend.line["in_service"] = status[~grid.line.trafo]
            model.grid_backend.trafo["in_service"] = status[grid.line.trafo]
            result_backend = model.solve_backend()

            if (
                model.params.gen_cost
                and model.params.line_margin
                and model.params.solver_name != "glpk"
            ):
                objective = (
                    result_backend["res_cost"]
                    + np.square(
                        result_backend["res_line"]["p_pu"] / model.line["max_p_pu"]
                    ).sum()
                )
            elif model.params.line_margin and model.params.solver_name != "glpk":
                objective = np.square(
                    result_backend["res_line"]["p_pu"] / model.line["max_p_pu"]
                ).sum()
            else:
                objective = result_backend["res_cost"]

            loads_p = grid.load["p_pu"].sum()
            generators_p = result_backend["res_gen"]["p_pu"].sum()
            valid = result_backend["valid"]

            results_backend = results_backend.append(
                {
                    "status": tuple(status),
                    "objective": objective,
                    "loads_p": loads_p,
                    "generators_p": generators_p,
                    "valid": valid,
                },
                ignore_index=True,
            )

        # Solve for optimal line status configuration
        result = model.solve(verbose=verbose)
        result_status = result["res_x"]
        result_objective = result["res_cost"]
        result_gap = result["res_gap"]  # Gap for finding the optimal configuration

        if verbose:
            model.print_results()

        # Check with brute force solution
        objective_brute = results_backend["objective"][results_backend["valid"]].min()

        hot_brute = (
            results_backend["objective"].values
            < (1 + result_gap) * objective_brute + result_gap
        )
        hot_brute = np.logical_and(hot_brute, results_backend["valid"])
        indices_brute = hot_to_indices(hot_brute)
        status_brute = results_backend["status"][indices_brute]

        match_idx = [
            idx
            for idx, line_status in zip(indices_brute, status_brute)
            if np.equal(line_status, result_status).all()
        ]

        # Compare
        results_backend["candidates"] = hot_brute
        results_backend["result_objective"] = np.nan
        results_backend["result_objective"][match_idx] = result_objective

        solution_idx = [
            idx
            for idx, line_status in zip(
                results_backend.index, results_backend["status"]
            )
            if np.equal(line_status, result_status).all()
        ]
        results_backend["solution"] = 0
        results_backend["solution"][solution_idx] = 1

        results_backend["status"] = [
            " ".join(np.array(line_status).astype(int).astype(str))
            for line_status in results_backend["status"]
        ]

        results_backend = results_backend[
            results_backend["valid"] & results_backend["candidates"]
        ]
        print(f"\n{model.name}\n")
        print(f"Solver: {result_objective}")
        print(results_backend.to_string())

        time.sleep(0.1)
        self.assertTrue(bool(match_idx))

    def test_case3_line_switching(self):
        case = load_case("case3")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        params = LineSwitchingParameters(n_max_line_status_changed=2, tol=1e-9)
        model = LineSwitchingDCOPF(
            f"{case.name} DC OPF Line Switching",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
            params=params,
        )

        self.runner_opf_line_switching(model, grid, verbose=False)

    def test_case4_line_switching(self):
        case = load_case("case4")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        params = LineSwitchingParameters(n_max_line_status_changed=2, tol=1e-9)
        model = LineSwitchingDCOPF(
            f"{case.name} DC OPF Line Switching",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
            params=params,
        )

        self.runner_opf_line_switching(model, grid, verbose=False)

    def test_case6_line_switching(self):
        case = load_case("case6")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        params = LineSwitchingParameters(n_max_line_status_changed=1, tol=1e-9)
        model = LineSwitchingDCOPF(
            f"{case.name} DC OPF Line Switching",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
            params=params,
        )

        self.runner_opf_line_switching(model, grid, verbose=False)

    def test_rte_case5_line_switching(self):
        case = load_case("rte_case5_example")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        params = LineSwitchingParameters(n_max_line_status_changed=3, tol=1e-9)
        model = LineSwitchingDCOPF(
            f"{case.name} DC OPF Line Switching",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
            params=params,
        )

        self.runner_opf_line_switching(model, grid, verbose=False)

    def test_l2rpn2019_line_switching(self):
        case = load_case("l2rpn_2019")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        params = LineSwitchingParameters(n_max_line_status_changed=2, tol=1e-9)
        model = LineSwitchingDCOPF(
            f"{case.name} DC OPF Line Switching",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
            params=params,
        )

        self.runner_opf_line_switching(model, grid, verbose=False)

    # def test_l2rpn2020_line_switching(self):
    #     if sys.platform != "win32":
    #         print("L2RPN 2020 not available.")
    #         self.assertTrue(True)
    #         return
    #
    #     case = load_case("l2rpn_wcci_2020")
    #     grid = GridDCOPF(
    #         case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
    #     )
    #
    #     params = LineSwitchingParameters(n_max_line_status_changed=1, tol=1e-9)
    #     model = LineSwitchingDCOPF(
    #         f"{case.name} DC OPF Line Switching",
    #         grid=grid,
    #         grid_backend=case.grid_backend,
    #         base_unit_p=case.base_unit_p,
    #         base_unit_v=case.base_unit_v,
    #         params=params,
    #     )
    #
    #     self.runner_opf_line_switching(model, grid, verbose=False)


class TestTopologyOptimizationDCOPF(unittest.TestCase):
    """
    Test DC-OPF with line status switching implementation.
    """

    @classmethod
    def setUpClass(cls):
        print("\nDC-OPF with topology optimization tests.\n")

    @staticmethod
    def topology_to_parts(x_topology, n_gen, n_load, n_line):
        x_gen = x_topology[:n_gen]
        x_load = x_topology[n_gen : (n_gen + n_load)]
        x_line_or_1 = x_topology[(n_gen + n_load) : (n_gen + n_load + n_line)]
        x_line_or_2 = x_topology[
            (n_gen + n_load + n_line) : (n_gen + n_load + 2 * n_line)
        ]
        x_line_ex_1 = x_topology[
            (n_gen + n_load + 2 * n_line) : (n_gen + n_load + 3 * n_line)
        ]
        x_line_ex_2 = x_topology[(n_gen + n_load + 3 * n_line) :]

        return x_gen, x_load, x_line_or_1, x_line_or_2, x_line_ex_1, x_line_ex_2

    def is_valid_topology(self, x_topology, n_gen, n_load, n_line):
        (
            x_gen,
            x_load,
            x_line_or_1,
            x_line_or_2,
            x_line_ex_1,
            x_line_ex_2,
        ) = self.topology_to_parts(x_topology, n_gen, n_load, n_line)

        cond_line_or = np.less_equal(x_line_or_1 + x_line_or_2, 1).all()
        cond_line_ex = np.less_equal(x_line_ex_1 + x_line_ex_2, 1).all()
        cond_line_disconnected = np.equal(
            x_line_or_1 + x_line_or_2, x_line_ex_1 + x_line_ex_2
        ).all()

        return cond_line_or and cond_line_ex and cond_line_disconnected

    def runner_opf_topology_optimization(
        self,
        model,
        verbose=False,
    ):
        np.random.seed(0)
        model.gen["cost_pu"] = np.random.uniform(1.0, 5.0, (model.grid.gen.shape[0],))
        model.build_model()

        if verbose:
            model.print_model()

        if model.params.solver_name == "glpk":
            print("Solver does not support bilinear or quadratic terms.")
            self.assertTrue(True)
            return

        result = model.solve(verbose=verbose)
        result_x = result["res_x"]
        result_objective = result["res_cost"]

        n_gen = model.grid.gen.shape[0]
        n_load = model.grid.load.shape[0]
        n_line = model.grid.line.shape[0]

        """
            BACKEND BRUTE FORCE.
        """

        results_backend = []
        for idx, x_topology in enumerate(
            itertools.product([0, 1], repeat=n_gen + n_load + 4 * n_line)
        ):
            # x_topology = [x_gen, x_load, x_line_or_1, x_line_or_2, x_line_ex_1, x_line_ex_2]
            x_topology = np.array(x_topology, dtype=np.int)

            # Initialization of variables
            (
                x_gen,
                x_load,
                x_line_or_1,
                x_line_or_2,
                x_line_ex_1,
                x_line_ex_2,
            ) = self.topology_to_parts(x_topology, n_gen, n_load, n_line)

            # Check valid topology
            if self.is_valid_topology(x_topology, n_gen, n_load, n_line):
                # Generator bus
                gen_sub_bus = np.ones_like(x_gen, dtype=np.int)
                gen_sub_bus[x_gen.astype(np.bool)] = 2
                gen_bus = [
                    model.grid.sub["bus"][sub_id][sub_bus - 1]
                    for sub_bus, sub_id in zip(gen_sub_bus, model.grid.gen["sub"])
                ]

                # Load bus
                load_sub_bus = np.ones_like(x_load, dtype=np.int)
                load_sub_bus[x_load.astype(np.bool)] = 2
                load_bus = [
                    model.grid.sub["bus"][sub_id][sub_bus - 1]
                    for sub_bus, sub_id in zip(load_sub_bus, model.grid.load["sub"])
                ]

                # Power line status
                line_status = np.logical_and(
                    np.logical_or(x_line_or_1, x_line_or_2),
                    np.logical_or(x_line_ex_1, x_line_ex_2),
                )

                # Power line - Origin bus
                line_or_sub_bus = -np.ones_like(x_line_or_1, dtype=np.int)
                line_or_sub_bus[x_line_or_1.astype(np.bool)] = 1
                line_or_sub_bus[x_line_or_2.astype(np.bool)] = 2
                line_or_bus = np.array(
                    [
                        model.grid.sub["bus"][sub_id][sub_bus - 1]
                        if sub_bus != -1
                        else model.grid.sub["bus"][sub_id][0]
                        for sub_bus, sub_id in zip(
                            line_or_sub_bus, model.grid.line["sub_or"]
                        )
                    ]
                )

                # Power line - Extremity bus
                line_ex_sub_bus = -np.ones_like(x_line_ex_1, dtype=np.int)
                line_ex_sub_bus[x_line_ex_1.astype(np.bool)] = 1
                line_ex_sub_bus[x_line_ex_2.astype(np.bool)] = 2
                line_ex_bus = np.array(
                    [
                        model.grid.sub["bus"][sub_id][sub_bus - 1]
                        if sub_bus != -1
                        else model.grid.sub["bus"][sub_id][0]
                        for sub_bus, sub_id in zip(
                            line_ex_sub_bus, model.grid.line["sub_ex"]
                        )
                    ]
                )

                # Construct grid for backend
                grid_tmp = model.grid_backend.deepcopy()
                grid_tmp.gen["bus"] = gen_bus
                grid_tmp.load["bus"] = load_bus

                grid_tmp.line["in_service"] = line_status[~model.grid.line.trafo]
                grid_tmp.line["from_bus"] = line_or_bus[~model.grid.line.trafo]
                grid_tmp.line["to_bus"] = line_ex_bus[~model.grid.line.trafo]

                grid_tmp.trafo["in_service"] = line_status[model.grid.line.trafo]
                grid_tmp.trafo["hv_bus"] = line_or_bus[model.grid.line.trafo]
                grid_tmp.trafo["lv_bus"] = line_ex_bus[model.grid.line.trafo]

                for gen_id in grid_tmp.gen.index.values:
                    pp.create_poly_cost(
                        grid_tmp,
                        gen_id,
                        "gen",
                        cp1_eur_per_mw=model.convert_per_unit_to_mw(
                            model.grid.gen["cost_pu"][gen_id]
                        ),
                    )

                print(f"{len(results_backend) + 1}/{idx}: Running DC-OPF ...")
                try:
                    pp.rundcopp(grid_tmp)
                    valid = True
                except (pp.optimal_powerflow.OPFNotConverged, IndexError) as e:
                    grid_tmp.res_cost = 0.0
                    print(e)
                    continue

                load_p = model.convert_mw_to_per_unit(grid_tmp.load["p_mw"].sum())
                gen_p = model.convert_mw_to_per_unit(grid_tmp.res_gen["p_mw"].sum())
                valid = valid and np.abs(gen_p - load_p) < 1e-6

                if (
                    model.params.obj_gen_cost
                    and model.params.obj_reward_quad
                    and model.params.solver_name != "glpk"
                ):
                    objective = (
                        grid_tmp.res_cost
                        + np.square(
                            model.convert_mw_to_per_unit(grid_tmp.res_line["p_from_mw"])
                            / model.grid.line["max_p_pu"]
                        ).sum()
                    )
                elif (
                    model.params.obj_reward_quad and model.params.solver_name != "glpk"
                ):
                    objective = +np.square(
                        model.convert_mw_to_per_unit(grid_tmp.res_line["p_from_mw"])
                        / model.grid.line["max_p_pu"]
                    ).sum()
                else:
                    objective = grid_tmp.res_cost

                results_backend.append(
                    {
                        "x": np.concatenate(
                            (
                                x_gen,
                                x_load,
                                x_line_or_1,
                                x_line_or_2,
                                x_line_ex_1,
                                x_line_ex_2,
                            )
                        ),
                        "gen_bus": gen_bus,
                        "load_bus": load_bus,
                        "line_or_bus": line_or_bus,
                        "line_ex_bus": line_ex_bus,
                        "line_status": line_status.astype(int),
                        "valid": valid,
                        "objective": np.round(objective, 3),
                        "load_p": load_p,
                        "gen_p": np.round(gen_p, 3),
                    }
                )

        results_backend = pd.DataFrame(results_backend)

        # Check with brute force solution
        objective_brute = results_backend["objective"][results_backend["valid"]].min()
        hot_brute = np.abs(results_backend["objective"].values - objective_brute) < 0.05
        indices_brute = hot_to_indices(hot_brute)
        status_brute = results_backend["x"][indices_brute]

        match_idx = [
            idx
            for idx, line_status in zip(indices_brute, status_brute)
            if np.equal(line_status, result_x).all()
        ]

        # Compare
        results_backend["candidates"] = hot_brute
        results_backend["result_objective"] = np.nan
        results_backend["result_objective"][match_idx] = np.round(result_objective, 3)

        print(f"\n{model.name}\n")
        print(f"Solver: {result_objective}")
        print(
            results_backend[
                [
                    "gen_bus",
                    "load_bus",
                    "line_or_bus",
                    "line_ex_bus",
                    "line_status",
                    "load_p",
                    "gen_p",
                    "valid",
                    "candidates",
                    "objective",
                    "result_objective",
                ]
            ][results_backend["candidates"] & results_backend["valid"]].to_string()
        )

        time.sleep(0.1)
        self.assertTrue(bool(match_idx))

    def test_case3_topology(self):
        case = load_case("case3")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        params = SinglestepTopologyParameters(
            obj_gen_cost=True,
            obj_reward_lin=False,
            obj_reward_quad=True,
            obj_reward_max=False,
            obj_lin_gen_penalty=False,
            obj_quad_gen_penalty=False,
        )
        model = TopologyOptimizationDCOPF(
            f"{case.name} DC OPF Topology Optimization",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
            params=params,
        )

        self.runner_opf_topology_optimization(model, verbose=False)

    def test_case3_topology_multistep(self):
        case = load_case("case3")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        params = MultistepTopologyParameters(
            horizon=1,
            obj_gen_cost=True,
            obj_reward_lin=False,
            obj_reward_quad=True,
            obj_reward_max=False,
            obj_lin_gen_penalty=False,
            obj_quad_gen_penalty=False,
        )
        forecasts = ForecastsPlain(env=case.env, horizon=params.horizon)
        model = MultistepTopologyDCOPF(
            f"{case.name} DC OPF Topology Optimization",
            grid=grid,
            forecasts=forecasts,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
            params=params,
        )

        self.runner_opf_topology_optimization(model, verbose=False)
