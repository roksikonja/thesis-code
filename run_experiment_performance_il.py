import os

import numpy as np

from experiments import ExperimentPerformance
from lib.agents import (
    make_test_agent,
    make_mixed_agent,
    SemiAgentMaxRho,
    SemiAgentRandom,
    SemiAgentKSteps,
    SemiAgentIL,
)
from lib.constants import Constants as Const
from lib.data_utils import make_dir, env_pf
from lib.dc_opf import load_case, CaseParameters
from lib.run_utils import create_logger

save_dir = make_dir(os.path.join(Const.RESULTS_DIR, "performance-aug-il-test"))

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
        do_chronics = np.arange(0, 10).tolist()
    else:
        do_chronics = [*np.arange(0, 2880, 240), *(np.arange(0, 2880, 240) + 1)]

    """
        Initialize agent.
    """
    # for semi_agent_name in ["k-steps", "max-rho", "random", "il", "do-nothing-agent"]:
    for semi_agent_name in ["il"]:
        if semi_agent_name == "max-rho":
            semi_agent = SemiAgentMaxRho(max_rho=0.85)
        elif semi_agent_name == "k-steps":
            semi_agent = SemiAgentKSteps(k=10)
        elif semi_agent_name == "random":
            semi_agent = SemiAgentRandom(probability=0.1)
        elif semi_agent_name == "il":
            model_dir = "./results/paper/l2rpn_2019_art-dc/2021-02-16_20-17-13_res"
            semi_agent = SemiAgentIL(case, model_dir)
        else:
            semi_agent = None

        if semi_agent is not None:
            agent = make_mixed_agent(case, **kwargs)
            agent.set_semi_agent(semi_agent)
        else:
            agent = make_test_agent("do-nothing-agent", case, **kwargs)

        """
            Experiments.
        """
        experiment_performance.analyse(
            case=case,
            agent=agent,
            do_chronics=do_chronics,
            n_chronics=-1,
            verbose=verbose,
        )

    experiment_performance.compare_agents(case, save_dir=case_save_dir)
