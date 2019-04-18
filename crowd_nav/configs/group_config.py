from crowd_nav.configs.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)
        self.sim.train_val_scenario = 'circle_crossing'
        self.sim.test_scenario = 'circle_crossing'
        self.sim.square_width = 10
        self.sim.circle_radius = 4
        self.sim.human_num = 5
        self.sim.group_num = 2
        self.sim.group_size = 1

        self.env.train_size = 2500


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)

        # gcn
        self.gcn.num_layer = 2
        self.gcn.X_dim = 16
        self.gcn.wr_dims = [gcn.X_dim * 3, gcn.X_dim]
        self.gcn.wh_dims = [gcn.X_dim * 3, gcn.X_dim]
        self.gcn.final_state_dim = gcn.X_dim
        self.gcn.gcn2_w1_dim = gcn.X_dim
        self.gcn.planning_dims = [32, 1]
        self.gcn.similarity_function = 'gaussian'
        self.gcn.layerwise_graph = False
        self.gcn.skip_connection = False


class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)
        self.train.rl_train_epochs = 10

