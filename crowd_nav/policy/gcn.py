import logging
import itertools
import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, num_layer, X_dim, wr_dims, wh_dims, final_state_dim,
                 gcn2_w1_dim, planning_dims, similarity_function, update_edge):
        super().__init__()
        # design choice

        # 'gaussian', 'embedded_gaussian', 'cosine', 'cosine_softmax', 'concatenation'
        self.similarity_function = similarity_function
        logging.info('self.similarity_func: {}'.format(self.similarity_function))
        human_state_dim = input_dim - self_state_dim
        self.self_state_dim = self_state_dim
        self.human_state_dim = human_state_dim
        self.num_layer = num_layer
        self.X_dim = X_dim
        self.update_edge = update_edge

        self.w_r = mlp(self_state_dim, wr_dims, last_relu=True)
        self.w_h = mlp(human_state_dim, wh_dims, last_relu=True)

        if self.similarity_function == 'embedded_gaussian':
            self.w_a = torch.nn.Parameter(torch.randn(self.X_dim, self.X_dim))
            self.w_a2 = torch.nn.Parameter(torch.randn(gcn2_w1_dim, gcn2_w1_dim))
        elif self.similarity_function == 'concatenation':
            self.w_a = mlp(2 * X_dim, [2 * X_dim, 1], last_relu=True)
            self.w_a2 = mlp(2 * gcn2_w1_dim, [2 * gcn2_w1_dim, 1], last_relu=True)

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

    def compute_similarity_matrix(self, X):
        if self.similarity_function == 'embedded_gaussian':
            if self.w_a.size(0) == X.size(2):
                A = torch.matmul(torch.matmul(X, self.w_a), X.permute(0, 2, 1))
            else:
                A = torch.matmul(torch.matmul(X, self.w_a2), X.permute(0, 2, 1))
            normalized_A = softmax(A, dim=2)
        elif self.similarity_function == 'gaussian':
            A = torch.matmul(X, X.permute(0, 2, 1))
            normalized_A = softmax(A, dim=2)
        elif self.similarity_function == 'cosine':
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            normalized_A = torch.div(A, norm_matrix)
        elif self.similarity_function == 'cosine_softmax':
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            normalized_A = softmax(torch.div(A, norm_matrix), dim=2)
        elif self.similarity_function == 'concatenation':
            indices = [pair for pair in itertools.product(list(range(X.size(1))), repeat=2)]
            selected_features = torch.index_select(X, dim=1, index=torch.LongTensor(indices).reshape(-1))
            pairwise_features = selected_features.reshape((-1, X.size(1) * X.size(1), X.size(2) * 2))
            if self.w_a[0].in_features == X.size(2) * 2:
                A = self.w_a(pairwise_features).reshape(-1, X.size(1), X.size(1))
            else:
                A = self.w_a2(pairwise_features).reshape(-1, X.size(1), X.size(1))
            normalized_A = A
        elif self.similarity_function == 'equal_attention':
            normalized_A = (torch.ones(X.size(1), X.size(1)) / X.size(1)).expand(X.size(0), X.size(1), X.size(1))
        elif self.similarity_function == 'diagonal':
            normalized_A = (torch.eye(X.size(1), X.size(1))).expand(X.size(0), X.size(1), X.size(1))
        else:
            raise NotImplementedError

        return normalized_A

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
        normalized_A = self.compute_similarity_matrix(X)
        self.A = normalized_A[0, :, :].data.cpu().numpy()

        # graph convolution
        if self.num_layer == 0:
            feat = X[:, 0, :]
        elif self.num_layer == 1:
            h1 = relu(torch.matmul(torch.matmul(normalized_A, X), self.w1))
            feat = h1[:, 0, :]
        else:
            # compute h1 and h2
            h1 = relu(torch.matmul(torch.matmul(normalized_A, X), self.w1))
            if self.update_edge:
                normalized_A2 = self.compute_similarity_matrix(h1)
            else:
                normalized_A2 = normalized_A
            h2 = relu(torch.matmul(torch.matmul(normalized_A2, h1), self.w2))
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

        X_dim = config.gcn.X_dim
        wr_dims = config.gcn.wr_dims
        wh_dims = config.gcn.wh_dims
        final_state_dim = config.gcn.final_state_dim
        gcn2_w1_dim = config.gcn.gcn2_w1_dim
        planning_dims = config.gcn.planning_dims
        similarity_function = config.gcn.similarity_function
        update_edge = config.gcn.update_edge

        self.set_common_parameters(config)
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, num_layer, X_dim, wr_dims, wh_dims,
                                  final_state_dim, gcn2_w1_dim, planning_dims, similarity_function, update_edge)
        logging.info('self.model:{}'.format(self.model))
        logging.info('GCN layers: {}'.format(num_layer))
        logging.info('Policy: {}'.format(self.name))

    def get_matrix_A(self):
        return self.model.A
