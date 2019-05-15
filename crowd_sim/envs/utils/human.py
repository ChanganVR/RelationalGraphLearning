from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Human(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.id = None

    def act(self, ob, t_in_real = None):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        if getattr(self.policy, 'name') == 'realsim_GrandCentral':
            action = self.policy.predict(state, t_in_real)
        else:
            action = self.policy.predict(state)
        return action
