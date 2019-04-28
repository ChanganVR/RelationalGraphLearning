from crowd_nav.configs.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)
        self.sim.nonstop_human = True


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        # sarl
        self.sarl.with_om = True

        # gcn
        self.gcn.num_layer = 1
        self.gcn.X_dim = 16
        self.gcn.wr_dims = [self.gcn.X_dim * 3, self.gcn.X_dim]
        self.gcn.wh_dims = [self.gcn.X_dim * 3, self.gcn.X_dim]
        self.gcn.final_state_dim = self.gcn.X_dim
        self.gcn.gcn2_w1_dim = self.gcn.X_dim

        self.gcn.similarity_function = 'gaussian'
        self.gcn.layerwise_graph = False
        self.gcn.skip_connection = False

class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)
