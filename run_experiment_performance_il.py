import os

import numpy as np

from experiments import ExperimentPerformance
from lib.agents import (
    make_test_agent,
    make_mixed_agent,
    SemiAgentMaxRho,
    SemiAgentRandom,
    SemiAgentILRho,
    SemiAgentIL,
)
from lib.constants import Constants as Const
from lib.data_utils import make_dir, env_pf
from lib.dc_opf import load_case, CaseParameters
from lib.run_utils import create_logger

save_dir = make_dir(os.path.join(Const.RESULTS_DIR, "performance-aug-il-rho"))
# save_dir = make_dir(os.path.join(Const.RESULTS_DIR, "performance-aug-il"))

env_dc = True
verbose = False

kwargs = dict()

for case_name in [
    "l2rpn_2019_art",
]:
    case_save_dir = make_dir(os.path.join(save_dir, f"{case_name}-{env_pf(env_dc)}"))
    create_logger(logger_name=f"logger", save_dir=case_save_dir)

    """
        Initialize environment.
    """
    parameters = CaseParameters(case_name=case_name, env_dc=env_dc)
    case = load_case(case_name, env_parameters=parameters)

    experiment_performance = ExperimentPerformance(save_dir=case_save_dir)

    np.random.seed(0)
    if "rte_case5" in case_name:
        do_chronics = np.arange(20)
    elif "l2rpn_2019" in case_name:
        do_chronics = np.arange(104, 120).tolist()
        do_chronics.extend(np.arange(160, 231).tolist())
        do_chronics.extend([11, 18, 20, 24, 28, 43, 47, 64, 79, 84, 93, 102, 157])
    else:
        do_chronics = [*np.arange(0, 2880, 240), *(np.arange(0, 2880, 240) + 1)]

    """
        Initialize agent.
    """
    for semi_agent_name in [
        # "random",
        "max-rho",
        # "il",
        # "do-nothing-agent",
        # "il-rho"
        # "agent-mip",
    ]:
        if semi_agent_name == "max-rho":
            semi_agent = SemiAgentMaxRho(max_rho=0.83)
        elif semi_agent_name == "random":
            semi_agent = SemiAgentRandom(probability=0.2)
        elif semi_agent_name == "il":
            # standard
            model_dir = "./results/paper/l2rpn_2019_art-dc/2021-02-19_00-42-00_res"
            # small
            # model_dir = "./results/paper/l2rpn_2019_art-dc/2021-02-21_12-21-52_res"
            semi_agent = SemiAgentIL(case, model_dir)
        elif semi_agent_name == "il-rho":
            model_dir = "./results/paper/l2rpn_2019_art-dc/2021-02-28_01-21-44_res"
            semi_agent = SemiAgentILRho(case, model_dir, max_rho=0.80)
        else:
            semi_agent = None

        if semi_agent is not None:
            agent = make_mixed_agent(case, **kwargs)
            agent.set_semi_agent(semi_agent)
        else:
            agent = make_test_agent(semi_agent_name, case, **kwargs)

        """
            Experiments
        """
        experiment_performance.analyse(
            case=case,
            agent=agent,
            do_chronics=do_chronics,
            n_chronics=-1,
            n_steps=1000,
            verbose=verbose,
        )

    experiment_performance.compare_agents(case, save_dir=case_save_dir)
