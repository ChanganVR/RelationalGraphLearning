import torch.nn as nn
from crowd_nav.policy.helpers import mlp


class ValueEstimator(nn.Module):
    def __init__(self, config, graph_model):
        super().__init__()
        self.graph_model = graph_model
        self.value_network = mlp(config.gcn.X_dim, config.model_predictive_rl.value_network_dims)

    def forward(self, state):
        state_embedding = self.graph_model(state)[:, 0, :]
        value = self.value_network(state_embedding)
        return value
