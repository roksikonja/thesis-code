import os

import numpy as np

from experiments import ExperimentPerformance
from lib.agents import make_test_agent
from lib.constants import Constants as Const
from lib.data_utils import make_dir, env_pf
from lib.dc_opf import load_case, CaseParameters
from lib.run_utils import create_logger
from lib.visualizer import Visualizer

visualizer = Visualizer()

save_dir = make_dir(os.path.join(Const.RESULTS_DIR, "performance-aug-f"))

env_dc = True
verbose = False

kwargs = dict(horizon=2, forecasts=False)
# kwargs = dict(horizon=2, obj_lambda_action=0.0)
# kwargs = dict(horizon=2, con_unitary_action=True)

for case_name in [
    # "rte_case5_example",
    # "rte_case5_example_art",
    # "l2rpn_2019",
    "l2rpn_2019_art",
    # "l2rpn_wcci_2020",
]:
    if "l2rpn_2019_art" not in case_name:
        continue

    case_save_dir = make_dir(os.path.join(save_dir, f"{case_name}-{env_pf(env_dc)}"))
    create_logger(logger_name=f"logger", save_dir=case_save_dir)

    """
        Initialize environment.
    """
    parameters = CaseParameters(case_name=case_name, env_dc=env_dc)
    case = load_case(case_name, env_parameters=parameters, verbose=verbose)

    experiment_performance = ExperimentPerformance(save_dir=case_save_dir)

    for agent_name in [
        # "do-nothing-agent",
        "agent-mip",
        # "agent-mip-l2rpn",
        # "agent-mip-q",
        # "agent-multistep-mip",
    ]:
        np.random.seed(0)
        if "rte_case5" in case_name:
            do_chronics = np.arange(20)
        elif "l2rpn_2019" in case_name:
            if "_art" not in case_name:
                # do_chronics = [0, 10, 100, 196, 200, 201, 206, 226, 259, 375, 384, 491]
                # do_chronics.extend(np.random.randint(0, 1000, 500).tolist())
                # do_chronics = np.random.randint(0, 1000, 15).tolist()

                do_chronics = np.random.randint(0, 1000, 50).tolist()
            else:
                do_chronics = np.arange(0, 151).tolist()
                # do_chronics = [18, 19, 20, 21, 22]
                # do_chronics = np.arange(0, 35).tolist()
        else:
            do_chronics = [*np.arange(0, 2880, 240), *(np.arange(0, 2880, 240) + 1)]

        """
            Initialize agent.
        """
        agent = make_test_agent(agent_name, case, **kwargs)

        """
            Experiments.
        """
        experiment_performance.analyse(
            case=case,
            agent=agent,
            do_chronics=do_chronics,
            n_chronics=-1,
            n_steps=-1,
            verbose=verbose,
        )

    experiment_performance.compare_agents(case, save_dir=case_save_dir)
