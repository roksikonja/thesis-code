import os

from lib.agents import (
    make_mixed_agent,
    SemiAgentIL,
)
from lib.constants import Constants as Const
from lib.data_utils import make_dir
from lib.dc_opf import load_case, CaseParameters
from lib.visualizer import pprint


def start_chronic(env, chronic_id):
    env.chronics_handler.tell_id(chronic_id - 1)  # Set chronic id
    obs = env.reset()
    pprint(
        "    chronic:", env.chronics_handler.get_name(), env.chronics_handler.get_id()
    )

    return obs


save_dir = make_dir(os.path.join(Const.RESULTS_DIR, "performance-aug-il-test"))

env_dc = True
verbose = False

kwargs = dict()

case_name = "l2rpn_2019_art"
parameters = CaseParameters(case_name=case_name, env_dc=env_dc)
case = load_case(case_name, env_parameters=parameters)
env = case.env

model_dir = "./results/paper/l2rpn_2019_art-dc/2021-02-16_20-17-13_res"
semi_agent = SemiAgentIL(case, model_dir)
agent = make_mixed_agent(case, **kwargs)
agent.set_semi_agent(semi_agent)
action = agent.actions[0]

obs = start_chronic(env, chronic_id=0)
agent.reset(obs)
x = semi_agent.create_x(obs, action)
y = semi_agent.model(x)
pprint("    - 0", y)

obs = start_chronic(env, chronic_id=2)
agent.reset(obs)
x = semi_agent.create_x(obs, action)
y = semi_agent.model(x)
pprint("    - 2", y)

obs = start_chronic(env, chronic_id=18)
agent.reset(obs)
x = semi_agent.create_x(obs, action)
y = semi_agent.model(x)
pprint("    - 18", y)
