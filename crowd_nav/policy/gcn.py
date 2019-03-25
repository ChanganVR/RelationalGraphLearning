import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, num_layer, X_dim, wr_dims, wh_dims, final_state_dim,
                 gcn2_w1_dim, planning_dims):
        super().__init__()
        # architecture design parameters
        self.diagonal_A = False
        self.equal_attention = False
        self.with_w_a = False
        logging.info('self.equal_attention:{}'.format(self.equal_attention))
        logging.info('self.with_w_a:{}'.format(self.with_w_a))

        human_state_dim = input_dim - self_state_dim
        self.self_state_dim = self_state_dim
        self.human_state_dim = human_state_dim
        self.num_layer = num_layer
        self.X_dim = X_dim

        self.w_r = mlp(self_state_dim, wr_dims, last_relu=True)
        self.w_h = mlp(human_state_dim, wh_dims, last_relu=True)

        if self.with_w_a:
            self.w_a = torch.nn.Parameter(torch.randn(self.X_dim, self.X_dim))
        else:
            self.w_a = torch.eye(self.X_dim)

        if num_layer == 1:
            self.w1 = torch.nn.Parameter(torch.randn(self.X_dim, final_state_dim))
        elif num_layer == 2:
            self.w1 = torch.nn.Parameter(torch.randn(self.X_dim, gcn2_w1_dim))
            self.w2 = torch.nn.Parameter(torch.randn(gcn2_w1_dim, final_state_dim))
            
        else:
            raise NotImplementedError

        self.value_net = mlp(final_state_dim, planning_dims)

        # for visualization
        self.A = None

    def forward(self, state_input):
        if isinstance(state_input, tuple):
            state, lengths = state_input
        else:
            state = state_input
            # lengths = torch.IntTensor([state.size()[1]])
       
        self_state = state[:, 0, :self.self_state_dim]
        human_states = state[:, :, self.self_state_dim:]

        # compute feature matrix X
        self_state_embedings = self.w_r(self_state)
        human_state_embedings = self.w_h(human_states)
        X = torch.cat([self_state_embedings.unsqueeze(1), human_state_embedings], dim=1)

        # compute matrix A
        if self.diagonal_A:
            normalized_A = torch.eye(X.size(1), X.size(1))
            self.A = normalized_A
        elif self.equal_attention:
            normalized_A = torch.ones(X.size(1), X.size(1)) / X.size(1)
            self.A = normalized_A
        else:
            A = torch.matmul(torch.matmul(X, self.w_a), X.permute(0, 2, 1))
            normalized_A = torch.nn.functional.softmax(A, dim=2)
            self.A = normalized_A[0, :, :].data.cpu().numpy()

        # graph convolution
        if self.num_layer == 0:
            feat = X[:, 0, :]
        elif self.num_layer == 1:
            h1 = relu(torch.matmul(torch.matmul(A, X), self.w1))
            feat = h1[:, 0, :]
        else:
            # compute h1 and h2
            h1 = relu(torch.matmul(torch.matmul(A, X), self.w1))
            h2 = relu(torch.matmul(torch.matmul(normalized_A, h1), self.w2))
            feat = h2[:, 0, :]

        # do planning using only the final layer feature of the agent
        value = self.value_net(feat)
        return value


class GCN(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'GCN'

    def configure(self, config):
        self.multiagent_training = config.gcn.multiagent_training
        num_layer = config.gcn.num_layer
        X_dim = config.gcn.X_dim = 32
        wr_dims = config.gcn.wr_dims = [64, X_dim]
        wh_dims = config.gcn.wh_dims = [64, X_dim]
        final_state_dim = config.gcn.final_state_dim = 64
        gcn2_w1_dim = config.gcn.gcn2_w1_dim = 64
        planning_dims = config.gcn.planning_dims = [150, 100, 100, 1]
        self.set_common_parameters(config)
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, num_layer, X_dim, wr_dims, wh_dims,
                                  final_state_dim, gcn2_w1_dim, planning_dims)
        logging.info('self.model:{}'.format(self.model))
        logging.info('GCN layers: {}'.format(num_layer))
        logging.info('Policy: {}'.format(self.name))

    def get_matrix_A(self):
        return self.model.A
