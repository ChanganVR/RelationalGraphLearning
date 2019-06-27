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

        self.env.train_size = 10000
        self.env.val_size = 500
        self.env.test_size = 500
        if debug:
            self.env.train_size = 50
            self.env.val_size = 20
            self.env.test_size = 20


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)

        #sarl
        self.sarl.with_om = False

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
        self.train.rl_train_epochs = 1
        self.train.train_episodes = 10000
        self.imitation_learning.il_episodes = 2000
        self.train.checkpoint_interval = 1000
        self.train.evaluation_interval = 1000
        if debug:
            self.imitation_learning.il_episodes = 20
            self.train.train_episodes = 50
            self.train.checkpoint_interval = 50
            self.train.evaluation_interval = 50


