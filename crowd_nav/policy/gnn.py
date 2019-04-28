import logging
import itertools
import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu
from torch.nn import Parameter
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, node_dim, wr_dims, wh_dims, edge_dim, planning_dims):

        # design choice
        human_state_dim = input_dim - self_state_dim
        self.self_state_dim = self_state_dim
        self.human_state_dim = human_state_dim

        self.w_r = mlp(self_state_dim, wr_dims, last_relu=True)
        self.w_h = mlp(human_state_dim, wh_dims, last_relu=True)

        self.f_e = mlp(node_dim *2, edge_dim, last_relu=True)
        self.f_n = mlp(edge_dim, node_dim, last_relu=True)

        self.value_net = mlp(final_state_dim, planning_dims)

        # for visualization
        self.A = None

    def node2edge(self, nodes):
        edges = self.f_e(nodes)
        return edges

    def edge2node(self, edge):
        nodes = self.f_n(edge)
        return nodes

    def forward(self, state_input):
        if isinstance(state_input, tuple):
            state, lengths = state_input
        else:
            state = state_input
            # lengths = torch.IntTensor([state.size()[1]])

        self_state = state[:, 0, :self.self_state_dim]
        human_states = state[:, :, self.self_state_dim:]

        self_state_embeddings = self.w_r(self_state)
        human_state_embedings = self.w_h(human_states)

        # compute edge features
        edges = self.f_e()

        self_state_embedings = self.w_r(self_state)
        human_state_embedings = self.w_h(human_states)
        X = torch.cat([self_state_embedings.unsqueeze(1), human_state_embedings], dim=1)


        # graph convolution
        if self.num_layer == 0:
            feat = X[:, 0, :]
        elif self.num_layer == 1:
            h1 = relu(torch.matmul(torch.matmul(normalized_A, X), self.w1))
            feat = h1[:, 0, :]
        else:
            # compute h1 and h2
            if not self.skip_connection:
                h1 = relu(torch.matmul(torch.matmul(normalized_A, X), self.w1))
            else:
                h1 = relu(torch.matmul(torch.matmul(normalized_A, X), self.w1)) + X
            if self.layerwise_graph:
                normalized_A2 = self.compute_similarity_matrix(h1)
            else:
                normalized_A2 = normalized_A
            if not self.skip_connection:
                h2 = relu(torch.matmul(torch.matmul(normalized_A2, h1), self.w2))
            else:
                h2 = relu(torch.matmul(torch.matmul(normalized_A2, h1), self.w2)) + h1
            feat = h2[:, 0, :]

        # do planning using only the final layer feature of the agent
        value = self.value_net(feat)
        return value


class GNN(MultiHumanRL):
    # general graph neural network, assume fully connected graph and assign equal weights to all pairwise relations
    def __init__(self):
        super().__init__()
        self.name = 'GNN'

    def configure(self, config):
        self.multiagent_training = config.gcn.multiagent_training
        node_dim = config.gnn.node_dim
        wr_dims = config.gnn.wr_dims
        wh_dims = config.gnn.wh_dims
        edge_dim = config.gnn.edge_dim
        planning_dims = config.gnn.planning_dims
        self.set_common_parameters(config)
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, node_dim, wr_dims, wh_dims, edge_dim, planning_dims)

    def get_matrix_A(self):
        return self.model.A
