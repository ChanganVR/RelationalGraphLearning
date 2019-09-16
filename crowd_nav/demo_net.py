import logging
import argparse
import importlib.util
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.state import ObservableState, FullState
from crowd_sim.envs.utils.state import JointState



def demo_net(humans, goal, robot_vel, robot_heading):

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--policy', type=str, default='model_predictive_rl')
    parser.add_argument('-m', '--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('-v', '--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-c', '--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--human_num', type=int, default=None)
    parser.add_argument('--group_size', type=int, default=None)
    parser.add_argument('--group_num', type=int, default=None)
    parser.add_argument('--safety_space', type=float, default=0.2)
    #parser.add_argument('--test_scenario', type=str, default='realsim_GrandCentral')
    parser.add_argument('--test_scenario', type=str, default=None)
    parser.add_argument('--plot_test_scenarios_hist', default=True, action='store_true')
    parser.add_argument('-d', '--planning_depth', type=int, default=None)
    parser.add_argument('-w', '--planning_width', type=int, default=None)
    parser.add_argument('--sparse_search', default=False, action='store_true')

    args = parser.parse_args()

    # configure logging and device

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)
    model_dir = '/local-scratch/changan/icra_benchmark_15000/mp_separate_l2_d2_w4'

    if args.model_dir is not None:
        config_file = os.path.join(args.model_dir, 'config.py')
        model_weights = os.path.join(args.model_dir, 'best_val.pth')
        logging.info('Loaded RL weights with best VAL')

    # configure policy
    policy_config = config.PolicyConfig(args.debug)
    policy = policy_factory[policy_config.name]()
    if args.planning_depth is not None:
        policy_config.model_predictive_rl.do_action_clip = True
        policy_config.model_predictive_rl.planning_depth = args.planning_depth
    if args.planning_width is not None:
        policy_config.model_predictive_rl.do_action_clip = True
        policy_config.model_predictive_rl.planning_width = args.planning_width
    if args.sparse_search:
        policy_config.model_predictive_rl.sparse_search = True

    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.load_model(model_weights)

    # configure environment
    env_config = config.EnvConfig(args.debug)

    if args.human_num is not None:
        env_config.sim.human_num = args.human_num
    if args.group_num is not None:
        env_config.sim.group_num = args.group_num
    if args.group_size is not None:
        env_config.sim.group_size = args.group_size

    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    robot.time_step = env.time_step
    robot.set_policy(policy)

    train_config = config.TrainConfig(args.debug)
    epsilon_end = train_config.train.epsilon_end
    if not isinstance(robot.policy, ORCA):
        robot.policy.set_epsilon(epsilon_end)

    policy.set_phase('test')
    policy.set_device(device)
    policy.set_env(env)
    robot.print_info()

    '''
    do some processing for the raw observation from the robot
    '''
    ob = []
    for human in humans:
        # human observable state: self.px, self.py, self.vx, self.vy, self.radius
        # robot full state: (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)
        ob.append(ObservableState(human[0], human[1], human[2], human[3], 0.3))

    """
    set the robot state from sensor data
    """
    robot.px = 0
    robot.py = 0
    robot.vx = robot_vel[0]
    robot.vy = robot_vel[1]
    robot.theta = 1.5707963267948966
    action = robot.act(ob)

    return action.vx, action.vy