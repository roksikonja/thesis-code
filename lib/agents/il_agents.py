from .base import BaseAgent
from .test_agents import AgentDoNothingTest, AgentMIPTest
from ..rewards import RewardL2RPN2019
from ..visualizer import pprint


def make_mixed_agent(
    case,
    save_dir=None,
    verbose=False,
    **kwargs,
):
    action_set = case.generate_unitary_action_set(
        case, case_save_dir=save_dir, verbose=verbose
    )

    case_name = case.env.name

    if "obj_lambda_action" not in kwargs:
        if "l2rpn_2019" in case_name:
            kwargs["obj_lambda_action"] = 0.07
        elif "rte_case5_example" in case_name:
            kwargs["obj_lambda_action"] = 0.006
        else:
            kwargs["obj_lambda_action"] = 0.05

    agent = AgentMIPMixed(case=case, action_set=action_set, **kwargs)

    return agent


class AgentMIPMixed(BaseAgent):
    def __init__(self, case, action_set, reward_class=RewardL2RPN2019, **kwargs):
        BaseAgent.__init__(self, name="Mixed MIP agent", case=case)

        self.agent_mip = AgentMIPTest(
            case=case, action_set=action_set, reward_class=reward_class, **kwargs
        )
        self.model_kwargs = self.agent_mip.model_kwargs
        self.agent_dn = AgentDoNothingTest(case=case, action_set=action_set)

        self.semi_agent = None
        self.agent = None

        self.action = self.case.env.action_space({})

        self.actions, self.actions_info = action_set

    def set_kwargs(self, **kwargs):
        self.agent_mip.set_kwargs(**kwargs)

    def set_semi_agent(self, semi_agent):
        self.semi_agent = semi_agent
        self.name = self.name + self.semi_agent.name

    def act(self, observation, reward, done=False):
        take_switching_action = self.semi_agent.take_switching_action(
            observation, self.action
        )

        if take_switching_action:
            self.agent = self.agent_mip
        else:  # Take a do-nothing action
            self.agent = self.agent_dn

        self.action = self.agent.act(observation, reward, done)

        return self.action

    def reset(self, obs):
        self.agent_mip.reset(obs)
        self.agent_dn.reset(obs)
        self.semi_agent.reset()

    def get_reward(self):
        return self.agent.get_reward()

    def compare_with_observation(self, obs, verbose=False):
        return self.agent.compare_with_observation(obs, verbose)

    def print_agent(self, default=False):
        pprint("\nAgent:", self.name, shift=36)
        self.agent_mip.print_agent(default)
        self.agent_dn.print_agent(default)
