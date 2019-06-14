import torch
import numpy as np
import itertools
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import ObservableState, FullState
from crowd_nav.policy.state_predictor import StatePredictor
from crowd_nav.policy.graph_model import RGL
from crowd_nav.policy.value_estimator import ValueEstimator


class ModelPredictiveRL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'ModelPredictiveRL'
        self.trainable = True
        self.multiagent_training = True
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.action_space = None
        self.rotation_constraint = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.robot_state_dim = 9
        self.human_state_dim = 5
        self.v_pref = 1
        self.value_estimator = None
        self.state_predictor = None
        self.planning_depth = None

    def configure(self, config):
        self.set_common_parameters(config)
        self.planning_depth = config.model_predictive_rl.planning_depth
        graph_model = RGL(config, self.robot_state_dim, self.human_state_dim)
        self.value_estimator = ValueEstimator(config, graph_model)
        self.state_predictor = StatePredictor(config, graph_model, self.time_step)
        self.model = [graph_model, self.value_estimator.value_network, self.state_predictor.human_motion_predictor]

    def set_common_parameters(self, config):
        self.gamma = config.rl.gamma
        self.kinematics = config.action_space.kinematics
        self.sampling = config.action_space.sampling
        self.speed_samples = config.action_space.speed_samples
        self.rotation_samples = config.action_space.rotation_samples
        self.rotation_constraint = config.action_space.rotation_constraint

    def set_device(self, device):
        self.device = device
        for model in self.model:
            model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_time_step(self, time_step):
        self.time_step = time_step
        self.state_predictor.time_step = time_step

    def get_normalized_gamma(self):
        return pow(self.gamma, self.time_step * self.v_pref)

    def get_model(self):
        return self.value_estimator

    def build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        holonomic = True if self.kinematics == 'holonomic' else False
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        else:
            rotations = np.linspace(-self.rotation_constraint, self.rotation_constraint, self.rotation_samples)

        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            if holonomic:
                action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else:
                action_space.append(ActionRot(speed, rotation))

        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.robot_state.v_pref)

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            max_action = None
            max_value = float('-inf')
            for action in self.action_space:
                # preprocess the state
                # TODO: separate features instead of concatenating
                # state_tensor = torch.cat([torch.Tensor([state.robot_state + human_state]).to(self.device)
                #                          for human_state in state.human_states], dim=0)
                robot_state_tensor = torch.Tensor([state.robot_state.to_tuple()]).to(self.device).unsqueeze(0)
                human_states_tensor = torch.Tensor([human_state.to_tuple() for human_state in state.human_states]).\
                    to(self.device).unsqueeze(0)
                next_state = self.state_predictor((robot_state_tensor, human_states_tensor), action)
                value = self.estimate_reward(state) + self.get_normalized_gamma() \
                        * self.V_planning(next_state, self.planning_depth)
                if value > max_value:
                    max_value = value
                    max_action = action
            if max_action is None:
                raise ValueError('Value network is not well trained.')

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action

    def V_planning(self, state, depth):
        if depth == 1:
            return self.value_estimator(state)

        action_space = self.action_space
        sums = []
        for action in action_space:
            next_state_est = self.state_predictor(state, action)
            reward_est = self.estimate_reward(state)
            sums.append(reward_est + self.get_normalized_gamma() * self.V_planning(next_state_est, depth - 1))

        return self.value_estimator(state) / depth + (depth - 1) / depth * max(sums)

    def estimate_reward(self, state):
        """ If the time step is small enough, it's okay to model agent as linear movement during this period

        """
        # TODO: to create a unified version
        # collision detection
        if isinstance(state, list):
            robot_state, human_states = state
            robot_state = robot_state.squeeze().data.numpy()
            human_states = human_states.squeeze().data.numpy()
            robot_state = FullState(robot_state[0], robot_state[1], robot_state[2], robot_state[3], robot_state[4],
                                    robot_state[5], robot_state[6], robot_state[7], robot_state[8])
            human_states = [ObservableState(human_state[0], human_state[1], human_state[2], human_state[3],
                                            human_state[4]) for human_state in human_states]
        else:
            robot_state, human_states = state.robot_state, state.human_states

        dmin = float('inf')
        collision = False
        for i, human in enumerate(human_states):
            dist = np.linalg.norm((robot_state.px - human.px, robot_state.py - human.py)) - robot_state.radius - human.radius
            if dist < 0:
                collision = True
                break
            if dist < dmin:
                dmin = dist

        # check if reaching the goal
        reaching_goal = np.linalg.norm((robot_state.px - robot_state.gx, robot_state.py - robot_state.gy)) < robot_state.radius
        if collision:
            reward = -0.25
        elif reaching_goal:
            reward = 1
        elif dmin < 0.2:
            reward = (dmin - 0.2) * 0.5 * self.time_step
        else:
            reward = 0

        return reward

    def transform(self, state):
        """
        Take the JointState to tensors

        :param state:
        :return: tensor of shape (# of agent, len(state))
        """
        robot_state_tensor = torch.Tensor([state.robot_state.to_tuple()]).to(self.device)
        human_states_tensor = torch.Tensor([human_state.to_tuple() for human_state in state.human_states]). \
            to(self.device)

        return robot_state_tensor, human_states_tensor
