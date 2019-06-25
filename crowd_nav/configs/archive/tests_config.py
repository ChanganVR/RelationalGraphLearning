from crowd_nav.configs.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=True):
        super(EnvConfig, self).__init__(debug)


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=True):
        super(PolicyConfig, self).__init__(debug)


class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=True):
        super(TrainConfig, self).__init__(debug)

