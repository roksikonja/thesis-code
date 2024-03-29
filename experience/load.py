import os

from lib.data_utils import make_dir, env_pf
from lib.dc_opf import load_case, CaseParameters
from lib.visualizer import pprint
from .collector import ExperienceCollector


def load_experience(case_name, agent_name, experience_dir, env_dc=True):
    case_experience_dir = make_dir(
        os.path.join(experience_dir, f"{case_name}-{env_pf(env_dc)}")
    )

    parameters = CaseParameters(case_name=case_name, env_dc=env_dc)
    case = load_case(case_name, env_parameters=parameters)
    env = case.env

    collector = load_agent_experience(env, agent_name, case_experience_dir)

    return case, collector


def load_agent_experience(env, agent_name, case_experience_dir):
    collector = ExperienceCollector(save_dir=case_experience_dir)
    collector.load_data(agent_name=agent_name, env=env)

    pprint("    - Number of loaded chronics:", len(collector.chronic_ids))

    return collector
