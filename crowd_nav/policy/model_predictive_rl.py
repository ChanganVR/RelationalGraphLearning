import logging
import torch
import numpy as np
from numpy.linalg import norm
import itertools
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import ObservableState, FullState, tensor_to_joint_state
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_sim.envs.utils.utils import VTree, print_vtree
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
        self.share_graph_model = None
        self.value_estimator = None
        self.state_predictor = None
        self.planning_depth = None
        self.planning_width = None
        self.do_action_clip = None
        self.traj = None

        #for visualization
        self.clipped_action_values = None
        self.dilated_action_values = None
        self.Q_d = dict()
        self.vtree = None
        self.action_taken = None

    def configure(self, config):
        self.set_common_parameters(config)
        self.planning_depth = config.model_predictive_rl.planning_depth
        self.do_action_clip = config.model_predictive_rl.do_action_clip
        self.planning_width = config.model_predictive_rl.planning_width
        self.share_graph_model = config.model_predictive_rl.share_graph_model
        if self.share_graph_model:
            graph_model = RGL(config, self.robot_state_dim, self.human_state_dim)
            self.value_estimator = ValueEstimator(config, graph_model)
            self.state_predictor = StatePredictor(config, graph_model, self.time_step)
            self.model = [graph_model, self.value_estimator.value_network, self.state_predictor.human_motion_predictor]
        else:
            graph_model1 = RGL(config, self.robot_state_dim, self.human_state_dim)
            self.value_estimator = ValueEstimator(config, graph_model1)
            graph_model2 = RGL(config, self.robot_state_dim, self.human_state_dim)
            self.state_predictor = StatePredictor(config, graph_model2, self.time_step)
            self.model = [graph_model1, graph_model2, self.value_estimator.value_network,
                          self.state_predictor.human_motion_predictor]

        logging.info('Planning depth: {}'.format(self.planning_depth))
        logging.info('Planning width: {}'.format(self.planning_width))

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

    def get_state_dict(self):
        if self.share_graph_model:
            return {
                'graph_model': self.value_estimator.graph_model.state_dict(),
                'value_network': self.value_estimator.value_network.state_dict(),
                'motion_predictor': self.state_predictor.human_motion_predictor.state_dict()
            }
        else:
            return {
                'graph_model1': self.value_estimator.graph_model.state_dict(),
                'graph_model2': self.state_predictor.graph_model.state_dict(),
                'value_network': self.value_estimator.value_network.state_dict(),
                'motion_predictor': self.state_predictor.human_motion_predictor.state_dict()
            }

    def get_traj(self):
        return self.traj

    def get_vtree(self):
        return self.vtree

    def get_action_taken(self):
        return self.action_taken

    def load_state_dict(self, state_dict):
        if self.share_graph_model:
            self.value_estimator.graph_model.load_state_dict(state_dict['graph_model'])
        else:
            self.value_estimator.graph_model.load_state_dict(state_dict['graph_model1'])
            self.state_predictor.graph_model.load_state_dict(state_dict['graph_model2'])

        self.value_estimator.value_network.load_state_dict(state_dict['value_network'])
        self.state_predictor.human_motion_predictor.load_state_dict(state_dict['motion_predictor'])

    def save_model(self, file):
        torch.save(self.get_state_dict(), file)

    def load_model(self, file):
        checkpoint = torch.load(file)
        self.load_state_dict(checkpoint)

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
        self.action_clip_counter = 0
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
            self.vtree = VTree(state, 0, self.planning_width)
            if self.do_action_clip:
                #self.action_values = [-float('Inf') for _ in range(len(self.action_space))]
                for i in range(self.planning_depth):
                    self.Q_d[i+1] = list()
                self.action_values = [0 for _ in range(len(self.action_space))]

                self.clipped_action_values = list()
            else:
                self.action_values = list()
            max_action = None
            max_value = float('-inf')
            max_traj = None

            if self.do_action_clip:
                state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
                action_space_clipped = self.action_clip(state_tensor, self.action_space, self.planning_width, vtree=self.vtree)
            else:
                action_space_clipped = self.action_space

            for action in action_space_clipped:
                state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
                next_state = self.state_predictor(state_tensor, action)
                self.vtree.add_child(action, next_state, self.planning_width)
                max_next_return, max_next_traj = self.V_planning(next_state, self.planning_depth, self.planning_width, self.vtree.children[action])
                reward_est = self.estimate_reward(state, action)
                value = reward_est + self.get_normalized_gamma() * max_next_return
                self.vtree.q_value_d_n[action] = value.tolist()[0][0]

                if self.do_action_clip:
                    action_index = self.action_space.index(action)
                    self.clipped_action_values.append(value.tolist()[0][0])
                    self.action_values[action_index] = value.tolist()[0][0]
                else:
                    self.action_values.append(value.tolist()[0][0])
                if value > max_value:
                    max_value = value
                    max_action = action
                    max_traj = [(state_tensor, action, reward_est)] + max_next_traj

            '''
            only for tree visualization
            '''
            for action in self.action_space:
                if action not in self.vtree.q_value_d_n:
                    self.vtree.q_value_d_n[action] = self.vtree.one_step_trajs_est[action][0] + self.get_normalized_gamma() * self.vtree.one_step_trajs_est[action][2]

            if max_action is None:
                raise ValueError('Value network is not well trained.')

        if self.phase == 'train':
            self.last_state = self.transform(state)
        else:
            self.traj = max_traj
            self.action_taken = max_action

        return max_action

    def action_clip(self, state, action_space, width, vtree):
        depth = 1
        self.action_clip_counter += 1
        values = []

        for action in action_space:
            next_state_est = self.state_predictor(state, action)
            reward_est = self.estimate_reward(state, action)

            next_return, _ = self.V_planning(next_state_est, depth, width)
            vtree.add_one_step_trajs(action, reward_est, next_state_est, next_return.data.tolist()[0][0])

            value = reward_est + self.get_normalized_gamma() * next_return
            values.append(value)
            if self.action_clip_counter == 1:
                self.Q_d[1].append(value.data.tolist()[0][0])
        #max_indexes = np.argpartition(np.array(values), -width)[-width:]
        max_indexes = np.array(values).argsort()[::-1][:width]
        clipped_action_space = [action_space[i] for i in max_indexes]
        return clipped_action_space

    def V_planning(self, state, depth, width, vtree=None):
        """ Plans n steps into future. Computes the value for the current state as well as the trajectories
        defined as a list of (state, action, reward) triples

        :param state:
        :param depth:
        :return:
        """
        current_state_value = self.value_estimator(state)

        if vtree != None:
            vtree.state_value_d_1 = current_state_value.data.tolist()[0][0]

        if depth == 1:
            return current_state_value, [(state, None, None)]

        if self.do_action_clip:
            action_space_clipped = self.action_clip(state, self.action_space, width, vtree)
        else:
            action_space_clipped = self.action_space

        returns = []
        trajs = []

        for action in action_space_clipped:
            next_state_est = self.state_predictor(state, action)
            reward_est = self.estimate_reward(state, action)
            vtree.add_child(action, next_state_est, width)
            vtree.children[action].state_value_d_1 = next_state_est
            next_value, next_traj = self.V_planning(next_state_est, depth - 1, self.planning_width, vtree.children[action])
            # TODO: verify this equation
            return_value = current_state_value / depth + (depth - 1) / depth * (self.get_normalized_gamma() * next_value + reward_est)

            returns.append(return_value)
            trajs.append([(state, action, reward_est)] + next_traj)

        max_index = np.argmax(returns)
        max_return = returns[max_index]
        max_traj = trajs[max_index]

        return max_return, max_traj

    def estimate_reward(self, state, action):
        """ If the time step is small enough, it's okay to model agent as linear movement during this period

        """
        # collision detection
        if isinstance(state, list) or isinstance(state, tuple):
            state = tensor_to_joint_state(state)
        human_states = state.human_states
        robot_state = state.robot_state

        dmin = float('inf')
        collision = False
        for i, human in enumerate(human_states):
            px = human.px - robot_state.px
            py = human.py - robot_state.py
            if self.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + robot_state.theta)
                vy = human.vy - action.v * np.sin(action.r + robot_state.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - robot_state.radius
            if closest_dist < 0:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # check if reaching the goal
        if self.kinematics == 'holonomic':
            px = robot_state.px + action.vx * self.time_step
            py = robot_state.py + action.vy * self.time_step
        else:
            theta = robot_state.theta + action.r
            px = robot_state.px + np.cos(theta) * action.v * self.time_step
            py = robot_state.py + np.sin(theta) * action.v * self.time_step

        end_position = np.array((px, py))
        reaching_goal = norm(end_position - np.array([robot_state.gx, robot_state.gy])) < robot_state.radius

        if collision:
            reward = -0.25
        elif reaching_goal:
            reward = 1
        elif dmin < 0.2:
            # adjust the reward based on FPS
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

class Discomfort(object):
    def __init__(self, min_dist):
        self.min_dist = min_dist













