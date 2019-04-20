from crowd_nav.configs.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)
        self.sim.train_val_scenario = 'group_circle_crossing'
        self.sim.test_scenario = 'group_circle_crossing'
        self.sim.square_width = 10
        self.sim.circle_radius = 4
        self.sim.group_num = 1
        self.sim.group_size = 1

        self.env.train_size = 2500
        self.env.val_size = 500
        self.env.test_size = 500
        if debug:
            self.env.train_size = 5
            self.env.val_size = 2
            self.env.test_size = 2


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)

        # gcn
        self.gcn.num_layer = 1
        self.gcn.X_dim = 16
        self.gcn.wr_dims = [self.gcn.X_dim * 3, self.gcn.X_dim]
        self.gcn.wh_dims = [self.gcn.X_dim * 3, self.gcn.X_dim]
        self.gcn.final_state_dim = self.gcn.X_dim
        self.gcn.gcn2_w1_dim = self.gcn.X_dim
        self.gcn.planning_dims = [32, 1]
        self.gcn.similarity_function = 'gaussian'
        self.gcn.layerwise_graph = False
        self.gcn.skip_connection = False


class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)
        self.train.rl_train_epochs = 10
        self.train.train_episodes = 2500
        self.imitation_learning.il_episodes = 500
        self.train.checkpoint_interval = 2500
        self.train.evaluation_interval = 2500
        if debug:
            self.imitation_learning.il_episodes = 2
            self.train.train_episodes = 5
            self.train.checkpoint_interval = 5
            self.train.evaluation_interval = 5


