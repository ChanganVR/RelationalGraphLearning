import torch
import torch.nn as nn
import numpy as np
from crowd_nav.policy.helpers import mlp


class StatePredictor(nn.Module):
    def __init__(self, config, graph_model, time_step):
        """
        This function predicts the next state given the current state as input.
        It uses a graph model to encode the state into a latent space and predict each human's next state.
        """
        super().__init__()
        self.trainable = True
        self.kinematics = config.action_space.kinematics
        self.graph_model = graph_model
        self.human_motion_predictor = mlp(config.gcn.X_dim, config.model_predictive_rl.motion_predictor_dims)
        self.time_step = time_step

    def forward(self, state, action, detach=False):
        """ Predict the next state tensor given current state as input.

        :return: tensor of shape (batch_size, # of agents, feature_size)
        """
        assert len(state[0].shape) == 3
        assert len(state[1].shape) == 3

        state_embedding = self.graph_model(state)
        if detach:
            state_embedding = state_embedding.detach()
        if action is None:
            # for training purpose
            next_robot_state = None
        else:
            next_robot_state = self.compute_next_state(state[0], action)
        next_human_states = self.human_motion_predictor(state_embedding)[:, 1:, :]

        next_observation = [next_robot_state, next_human_states]
        return next_observation

    def compute_next_state(self, robot_state, action):
        # currently it can not perform parallel computation
        if robot_state.shape[0] != 1:
            raise NotImplementedError

        # px, py, vx, vy, radius, gx, gy, v_pref, theta
        next_state = robot_state.clone().squeeze()
        if self.kinematics == 'holonomic':
            next_state[0] = next_state[0] + action.vx * self.time_step
            next_state[1] = next_state[1] + action.vy * self.time_step
            next_state[2] = action.vx
            next_state[3] = action.vy
        else:
            next_state[7] = next_state[7] + action.r
            next_state[0] = next_state[0] + np.cos(next_state[7]) * action.v * self.time_step
            next_state[1] = next_state[1] + np.sin(next_state[7]) * action.v * self.time_step
            next_state[2] = np.cos(next_state[7]) * action.v
            next_state[3] = np.sin(next_state[7]) * action.v

        return next_state.unsqueeze(0).unsqueeze(0)


class LinearStatePredictor(object):
    def __init__(self, config, time_step):
        """
        This function predicts the next state given the current state as input.
        It uses a graph model to encode the state into a latent space and predict each human's next state.
        """
        super().__init__()
        self.trainable = False
        self.kinematics = config.action_space.kinematics
        self.time_step = time_step

    def __call__(self, state, action):
        """ Predict the next state tensor given current state as input.

        :return: tensor of shape (batch_size, # of agents, feature_size)
        """
        assert len(state[0].shape) == 3
        assert len(state[1].shape) == 3

        next_robot_state = self.compute_next_state(state[0], action)
        next_human_states = self.linear_motion_approximator(state[1])

        next_observation = [next_robot_state, next_human_states]
        return next_observation

    def compute_next_state(self, robot_state, action):
        # currently it can not perform parallel computation
        if robot_state.shape[0] != 1:
            raise NotImplementedError

        # px, py, vx, vy, radius, gx, gy, v_pref, theta
        next_state = robot_state.clone().squeeze()
        if self.kinematics == 'holonomic':
            next_state[0] = next_state[0] + action.vx * self.time_step
            next_state[1] = next_state[1] + action.vy * self.time_step
            next_state[2] = action.vx
            next_state[3] = action.vy
        else:
            next_state[7] = next_state[7] + action.r
            next_state[0] = next_state[0] + np.cos(next_state[7]) * action.v * self.time_step
            next_state[1] = next_state[1] + np.sin(next_state[7]) * action.v * self.time_step
            next_state[2] = np.cos(next_state[7]) * action.v
            next_state[3] = np.sin(next_state[7]) * action.v

        return next_state.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def linear_motion_approximator(human_states):
        """ approximate human states with linear motion, input shape : (batch_size, human_num, human_state_size)
        """
        # px, py, vx, vy, radius
        next_state = human_states.clone().squeeze()
        next_state[:, 0] = next_state[:, 0] + next_state[:, 2]
        next_state[:, 1] = next_state[:, 1] + next_state[:, 3]

        return next_state.unsqueeze(0)

