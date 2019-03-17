import sys
import logging
import argparse
import os
import shutil
import importlib.util
import torch
import gym
import git
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from tensorboardX import SummaryWriter
import time

def main(args):
    # configure paths
    make_new_dir = True
    if os.path.exists(args.output_dir):
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        else:
            key = input('Output directory already exists! Overwrite the folder? (y/n)')
            if key == 'y' and not args.resume:
                shutil.rmtree(args.output_dir)
            else:
                make_new_dir = False
                args.config = os.path.join(args.output_dir, args.config)
    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.config, args.output_dir)
    
    log_file = os.path.join(args.output_dir, 'output.log')
    
    #writer_p = os.path.join(args.output_dir + time.strftime("%m-%d-%H-%M-%S", time.localtime()))
    #writer = SummaryWriter(log_dir = writer_p)
    writer = SummaryWriter(log_dir = args.output_dir)
    il_weight_file = os.path.join(args.output_dir, 'il_model.pth')
    rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')
    
    spec = importlib.util.spec_from_file_location('config', args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure logging
    mode = 'a' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    repo = git.Repo(search_parent_directories=True)
    logging.info('Current git head hash code: {}'.format(repo.head.object.hexsha))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    if not policy.trainable:
        parser.error('Policy has to be trainable')
    policy_config = config.PolicyConfig()
    policy.configure(policy_config)
    policy.set_device(device)

    # configure environment
    env_config = config.EnvConfig(args.debug)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    # read training parameters
    train_config = config.TrainConfig(args.debug)
    rl_learning_rate = train_config.train.rl_learning_rate
    train_batches = train_config.train.train_batches
    train_episodes = train_config.train.train_episodes
    sample_episodes = train_config.train.sample_episodes
    target_update_interval = train_config.train.target_update_interval
    evaluation_interval = train_config.train.evaluation_interval
    capacity = train_config.train.capacity
    epsilon_start = train_config.train.epsilon_start
    epsilon_end = train_config.train.epsilon_end
    epsilon_decay = train_config.train.epsilon_decay
    checkpoint_interval = train_config.train.checkpoint_interval

    # configure trainer and explorer
    memory = ReplayMemory(capacity)
    model = policy.get_model()
    batch_size = train_config.trainer.batch_size
    trainer = Trainer(model, memory, device, batch_size)
    explorer = Explorer(env, robot, device, memory, policy.gamma, target_policy=policy)

    # imitation learning
    if args.resume:
        if not os.path.exists(rl_weight_file):
            logging.error('RL weights does not exist')
        model.load_state_dict(torch.load(rl_weight_file))
        rl_weight_file = os.path.join(args.output_dir, 'resumed_rl_model.pth')
        logging.info('Load reinforcement learning trained weights. Resume training')
    elif os.path.exists(il_weight_file):
        model.load_state_dict(torch.load(il_weight_file))
        logging.info('Load imitation learning trained weights.')
    else:
        args.learning_phase = 'il'
        il_episodes = train_config.imitation_learning.il_episodes
        il_policy = train_config.imitation_learning.il_policy
        il_epochs = train_config.imitation_learning.il_epochs
        il_learning_rate = train_config.imitation_learning.il_learning_rate
        
        trainer.set_learning_rate(il_learning_rate, policy.name, args.learning_phase)
        if robot.visible:
            safety_space = 0
        else:
            safety_space = train_config.imitation_learning.safety_space
        il_policy = policy_factory[il_policy]()
        il_policy.multiagent_training = policy.multiagent_training
        il_policy.safety_space = safety_space
        robot.set_policy(il_policy)
        explorer.run_k_episodes(il_episodes, 'train', update_memory=True, imitation_learning=True)
        trainer.optimize_epoch(il_epochs, writer)
        torch.save(model.state_dict(), il_weight_file)
        logging.info('Finish imitation learning. Weights saved.')
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
    explorer.update_target_model(model)

    # reinforcement learning
    policy.set_env(env)
    robot.set_policy(policy)
    robot.print_info()
    args.learning_phase = 'rl'
    trainer.set_learning_rate(rl_learning_rate, policy.name, args.learning_phase)
    # fill the memory pool with some RL experience
    if args.resume:
        robot.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(100, 'train', update_memory=True, episode=0)
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
    episode = 0
    args.slidewindow = 200
    
    success_rates_train = []
    collision_rates_train = []
    ave_nav_times_train = []
    rewards_train = []
    best_val_reward = -1
    val_iter = 0

    # evaluate the model after imitation learning
    if episode % evaluation_interval == 0:
        logging.info('evaluate the model instantly after imitation learning on the validaton cases')           
        success_rate_val, collision_rate_val, avg_nav_time_val, reward_val = explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)
        writer.add_scalar('val_data/success_rate', success_rate_val, val_iter)
        writer.add_scalar('val_data/collision_rate', collision_rate_val, val_iter)
        writer.add_scalar('val_data/nav_time', avg_nav_time_val, val_iter)
        writer.add_scalar('val_data/rewards', reward_val, val_iter)
       

    
    while episode < train_episodes:
        if args.resume:
            epsilon = epsilon_end
        else:
            if episode < epsilon_decay:
                epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
            else:
                epsilon = epsilon_end
        robot.policy.set_epsilon(epsilon)

        # sample k episodes into memory and optimize over the generated memory
        success_rate_train, collision_rate_train, ave_nav_time_train, reward_train = explorer.run_k_episodes(sample_episodes, 'train', update_memory=True, episode=episode)        
        success_rates_train.append(success_rate_train)
        collision_rates_train.append(collision_rate_train)
        ave_nav_times_train.append(ave_nav_time_train)
        rewards_train.append(reward_train)
        writer.add_scalar('train_data/success_rate', sum(success_rates_train[-args.slidewindow:])/len(success_rates_train[-args.slidewindow:]), episode)
        writer.add_scalar('train_data/collision_rate', sum(collision_rates_train[-args.slidewindow:])/len(collision_rates_train[-args.slidewindow:]), episode)
        writer.add_scalar('train_data/nav_time', sum(ave_nav_times_train[-args.slidewindow:])/len(collision_rates_train[-args.slidewindow:]), episode)
        writer.add_scalar('train_data/rewards', sum(rewards_train[-args.slidewindow:])/len(rewards_train[-args.slidewindow:]), episode)
        
        trainer.optimize_batch(train_batches)
        episode += 1

        if episode % target_update_interval == 0:
            explorer.update_target_model(model)
        # evaluate the model
        if episode % evaluation_interval == 0:
            success_rate_val, collision_rate_val, avg_nav_time_val, reward_val = explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)
            writer.add_scalar('val_data/success_rate', success_rate_val, val_iter)
            writer.add_scalar('val_data/collision_rate', collision_rate_val, val_iter)
            writer.add_scalar('val_data/nav_time', avg_nav_time_val, val_iter)
            writer.add_scalar('val_data/rewards', reward_val, val_iter)
            val_iter += 1
            
        if episode != 0 and episode % checkpoint_interval == 0:
            if reward_val > best_val_reward:
                best_val_reward = reward_val
                logging.info('save the best model in episode:{}'.format(episode))
                torch.save(model.state_dict(), rl_weight_file.split('.')[0] + 'best_val.pth' )
            
            torch.save(model.state_dict(), rl_weight_file.split('.')[0] + '_'  + str(int(episode /checkpoint_interval)) + '.pth' )

    # final test
    explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='cadrl')
    parser.add_argument('--config', type=str, default='config.py')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    sys_args = parser.parse_args()

    main(sys_args)
