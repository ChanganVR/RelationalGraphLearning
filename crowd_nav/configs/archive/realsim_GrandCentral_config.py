from crowd_nav.configs.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)
        self.sim.train_val_scenario = 'realsim_GrandCentral'
        self.sim.test_scenario = 'realsim_GrandCentral'
        self.sim.square_width = 10
        self.sim.circle_radius = 4
        self.sim.group_num = 1
        self.sim.group_size = 1
        self.sim.centralized_planning = False

        self.humans.policy = 'realsim_GrandCentral'
        self.humans.radius = 0

        self.robot.v_pref = 1
        self.env.train_size = 1000
        self.env.val_size = 100
        self.env.test_size = 400
        if debug:
            self.env.train_size = 20
            self.env.val_size = 2
            self.env.test_size = 2


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.action_space.query_env = False
        #sarl
        self.sarl.with_om = True

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

        # gnn

        #


class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)
        self.train.rl_train_epochs = 1
        self.train.train_episodes = 1000
        self.imitation_learning.il_episodes = 200
        self.train.checkpoint_interval = 100
        self.train.evaluation_interval = 100
        self.train.train_with_pretend_batch = True
        if debug:
            self.imitation_learning.il_episodes = 5
            self.train.train_episodes = 20
            self.train.checkpoint_interval = 20
            self.train.evaluation_interval = 20



