import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY

class realsim_GrandCentral(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'realsim_GrandCentral'
        self.trainable = False
        self.kinematics = 'holonomic'
        self.multiagent_training = True
        self.trajectory = None

    def configure(self, config):
        return

    def load_trajectory(self, trajectory):
        self.trajectory = trajectory

    def predict(self, state, t_in_real):
        last_t = max(list(self.trajectory.keys()))
        start_t = min(list(self.trajectory.keys()))
        if t_in_real == last_t + 1 or t_in_real == start_t:
            # for those newly appeared human, the action is to just appear and stay in the appear position
            action = ActionXY(0, 0)
        else:
            vx = (self.trajectory[t_in_real][0] - self.trajectory[t_in_real - 1][0])/self.time_step
            vy = (self.trajectory[t_in_real][1] - self.trajectory[t_in_real - 1][1])/self.time_step
            action = ActionXY(vx, vy)

        return action