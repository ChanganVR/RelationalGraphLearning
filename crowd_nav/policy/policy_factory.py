from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.gcn import GCN
from crowd_nav.policy.gnn import GNN
from crowd_nav.policy.rrn import RRN
from crowd_nav.policy.cgcn import CGCN

policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['gcn'] = GCN
policy_factory['gnn'] = GNN
policy_factory['rrn'] = RRN
policy_factory['cgcn'] = CGCN