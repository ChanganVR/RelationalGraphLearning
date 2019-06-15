import sys
import logging
import argparse
import os
import shutil
import importlib.util
import torch
import gym
import copy
import git
import re
from tensorboardX import SummaryWriter
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory


def set_random_seeds(seed):
    """
    Sets the random seeds for pytorch cpu and gpu
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def main(args):
    import time
    # set_random_seeds(args.randomseed)
    set_random_seeds(time.time())
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
    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.config, os.path.join(args.output_dir, 'config.py'))

        # insert the arguments from command line to the config file
        with open(os.path.join(args.output_dir, 'config.py'), 'r') as fo:
            config_text = fo.read()
        search_pairs = {r"gcn.X_dim = \d*": "gcn.X_dim = {}".format(args.X_dim),
                        r"gcn.num_layer = \d": "gcn.num_layer = {}".format(args.layers),
                        r"gcn.similarity_function = '\w*'": "gcn.similarity_function = '{}'".format(args.sim_func),
                        r"gcn.layerwise_graph = \w*": "gcn.layerwise_graph = {}".format(args.layerwise_graph),
                        r"gcn.skip_connection = \w*": "gcn.skip_connection = {}".format(args.skip_connection)}

        for find, replace in search_pairs.items():
            config_text = re.sub(find, replace, config_text)

        with open(os.path.join(args.output_dir, 'config.py'), 'w') as fo:
            fo.write(config_text)

    if args.save_scene:
        save_scene_dir = os.path.join(args.output_dir, 'save_scene')
        os.makedirs(save_scene_dir)
    else:
        save_scene_dir = None

    args.config = os.path.join(args.output_dir, 'config.py')
    log_file = os.path.join(args.output_dir, 'output.log')
    il_weight_file = os.path.join(args.output_dir, 'il_model.pth')
    rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')

    spec = importlib.util.spec_from_file_location('config', args.config)
    if spec is None:
        parser.error('Config file not found.')
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
    logging.info('Current config content is :{}'.format(config))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)
    writer = SummaryWriter(log_dir=args.output_dir)

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
    robot.time_step = env.time_step
    env.set_robot(robot)
    env.save_scene_dir = save_scene_dir

    # read training parameters
    train_config = config.TrainConfig(args.debug)
    rl_train_epochs = train_config.train.rl_train_epochs
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
    train_with_pretend_batch = train_config.train.train_with_pretend_batch

    # configure trainer and explorer
    memory = ReplayMemory(capacity)
    model = policy.get_model()
    batch_size = train_config.trainer.batch_size
    optimizer = train_config.trainer.optimizer
    trainer = Trainer(model, policy.state_predictor, memory, device, batch_size, optimizer)
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
        il_episodes = train_config.imitation_learning.il_episodes
        il_policy = train_config.imitation_learning.il_policy
        il_epochs = train_config.imitation_learning.il_epochs
        il_learning_rate = train_config.imitation_learning.il_learning_rate
        trainer.set_learning_rate(il_learning_rate)
        if robot.visible:
            safety_space = 0
        else:
            safety_space = train_config.imitation_learning.safety_space
        il_policy = policy_factory[il_policy]()
        il_policy.multiagent_training = policy.multiagent_training
        il_policy.safety_space = safety_space
        robot.set_policy(il_policy)
        explorer.run_k_episodes(il_episodes, 'train', update_memory=True, imitation_learning=True)
        if train_with_pretend_batch:
            trainer.optimize_epoch_pretend_batch(il_epochs, writer)
        else:
            trainer.optimize_epoch(il_epochs, writer)
        policy.save_model(il_weight_file)
        logging.info('Finish imitation learning. Weights saved.')
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)

    explorer.update_target_model(model)

    # reinforcement learning
    policy.set_env(env)
    robot.set_policy(policy)
    robot.print_info()
    trainer.set_learning_rate(rl_learning_rate)
    # fill the memory pool with some RL experience
    if args.resume:
        robot.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(100, 'train', update_memory=True, episode=0)
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
    episode = 0
    best_val_reward = -1
    best_val_model = None
    # evaluate the model after imitation learning

    if episode % evaluation_interval == 0:
        logging.info('Evaluate the model instantly after imitation learning on the validation cases')
        sr, cr, time, reward, avg_return = explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)
        writer.add_scalar('val/success_rate', sr, episode // evaluation_interval)
        writer.add_scalar('val/collision_rate', cr, episode // evaluation_interval)
        writer.add_scalar('val/time', time, episode // evaluation_interval)
        writer.add_scalar('val/reward', reward, episode // evaluation_interval)
        writer.add_scalar('val/avg_return', avg_return, episode // evaluation_interval)

        if args.test_after_every_eval:
            sr, cr, time, reward = explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode, print_failure=True)
            writer.add_scalar('test/success_rate', sr, episode // evaluation_interval)
            writer.add_scalar('test/collision_rate', cr, episode // evaluation_interval)
            writer.add_scalar('test/time', time, episode // evaluation_interval)
            writer.add_scalar('test/reward', reward, episode // evaluation_interval)

    if args.save_stable_models:
        stable_srs = []
        stable_crs = []
        stable_rewards = []
        stable_times = []

    for e_id in range(rl_train_epochs):
        episode = 0
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
            sr, cr, time, reward, avg_return = explorer.run_k_episodes(sample_episodes, 'train', update_memory=True,
                                                           episode=episode, epoch=e_id)
            writer.add_scalar('train/success_rate', sr, episode + train_episodes * e_id)
            writer.add_scalar('train/collision_rate', cr, episode + train_episodes * e_id)
            writer.add_scalar('train/time', time, episode + train_episodes * e_id)
            writer.add_scalar('train/reward', reward, episode + train_episodes * e_id)
            writer.add_scalar('train/avg_return', avg_return, episode + train_episodes * e_id)

            if train_with_pretend_batch:
                trainer.optimize_pretend_batch(train_batches)
            else:
                trainer.optimize_batch(train_batches)
            episode += 1

            if (episode + train_episodes * e_id) % target_update_interval == 0:
                explorer.update_target_model(model)
            # evaluate the model
            if (episode + train_episodes * e_id) % evaluation_interval == 0:
                sr, cr, time, reward, avg_return = explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode, epoch=e_id)
                writer.add_scalar('val/success_rate', sr, (episode + train_episodes * e_id) // evaluation_interval)
                writer.add_scalar('val/collision_rate', cr, (episode + train_episodes * e_id) // evaluation_interval)
                writer.add_scalar('val/time', time, (episode + train_episodes * e_id) // evaluation_interval)
                writer.add_scalar('val/reward', reward, (episode + train_episodes * e_id) // evaluation_interval)
                writer.add_scalar('val/avg_return', avg_return, (episode + train_episodes * e_id) // evaluation_interval)

                if (episode + train_episodes * e_id) % checkpoint_interval == 0 and reward > best_val_reward:
                    best_val_reward = reward
                    best_val_model = copy.deepcopy(policy.get_state_dict())
            # test after every evaluation to check how the generalization performance evolves
                if args.test_after_every_eval:
                    sr, cr, time, reward, avg_return = explorer.run_k_episodes(
                        env.case_size['test'], 'test', episode=episode, epoch=e_id, print_failure=True)
                    writer.add_scalar('test/success_rate', sr, (episode + train_episodes * e_id) // evaluation_interval)
                    writer.add_scalar('test/collision_rate', cr, (episode + train_episodes * e_id) // evaluation_interval)
                    writer.add_scalar('test/time', time, (episode + train_episodes * e_id) // evaluation_interval)
                    writer.add_scalar('test/reward', reward, (episode + train_episodes * e_id) // evaluation_interval)
                    writer.add_scalar('test/avg_return', avg_return, (episode + train_episodes * e_id) // evaluation_interval)

            if episode != 0 and (episode + train_episodes * e_id) % checkpoint_interval == 0:
                current_checkpoint = (episode + train_episodes * e_id) // checkpoint_interval - 1
                save_every_checkpoint_rl_weight_file = rl_weight_file.split('.')[0] + '_' + str(current_checkpoint) + '.pth'
                policy.save_model(save_every_checkpoint_rl_weight_file)

            if args.save_stable_models:
                stable_checkpoint_interval = 20
                save_after = 0.9
                total_stable_models = train_episodes * (1 - save_after) // stable_checkpoint_interval
                test_size = int(env_config.env.test_size // total_stable_models)
                logging.info('check the test_size: {}'.format(test_size))
                logging.info('save_after: {}'.format(save_after))
                logging.info('stable_checkpoint_interval: {}'.format(stable_checkpoint_interval))
                if (episode + train_episodes * e_id) >= train_episodes * save_after:
                    if episode != 0 and (episode + train_episodes * e_id) % stable_checkpoint_interval == 0:
                        current_stable_checkpoint = (episode + train_episodes * e_id) // stable_checkpoint_interval - 1
                        save_every_stable_rl_weight_file = rl_weight_file.split('.')[0] + '_' + str(episode) + '.pth'
                        policy.save(save_every_stable_rl_weight_file)
                        logging.info('check the env.case_encounter: {}'.format(env.case_counter['test']))
                        sr, cr, time, reward = explorer.run_k_episodes(test_size, 'test', episode=episode, epoch=e_id, print_failure=True)
                        stable_srs.append(sr)
                        stable_crs.append(cr)
                        stable_times.append(time)
                        stable_rewards.append(reward)
                        writer.add_scalar('stable_test/success_rate', sr, (episode + train_episodes * e_id) // stable_checkpoint_interval)
                        writer.add_scalar('stable_test/collision_rate', cr, (episode + train_episodes * e_id) // stable_checkpoint_interval)
                        writer.add_scalar('stable_test/time', time, (episode + train_episodes * e_id) // stable_checkpoint_interval)
                        writer.add_scalar('stable_test/reward', reward, (episode + train_episodes * e_id) // stable_checkpoint_interval)

    if args.save_stable_models:
        logging.info('the {} stable models average reward on the test scenarios are :{}'.format(len(stable_rewards), sum(stable_rewards)/len(stable_rewards)))
        logging.info('the {} stable models average sr on the test scenarios are :{}'.format(len(stable_srs), sum(stable_srs) /len(stable_srs)))
        logging.info('the {} stable models average cr on the test scenarios are :{}'.format(len(stable_crs), sum(stable_crs) /len(stable_crs)))
        logging.info('the {} stable models average time on the test scenarios are :{}'.format(len(stable_times), sum(stable_times) /len(stable_times)))

    # # test with the best val model
    if best_val_model is not None:
        # TODO: replace that with
        policy.load_state_dict(best_val_model)
        torch.save(best_val_model, os.path.join(args.output_dir, 'best_val.pth'))
        logging.info('Save the best val model with the reward: {}'.format(best_val_reward))
    explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode, epoch=e_id, print_failure=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='model_predictive_rl')
    parser.add_argument('--config', type=str, default='configs/model_predictive_1.py')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--save_scene', default=False, action='store_true')
    parser.add_argument('--test_after_every_eval', default=False, action='store_true')
    # parser.add_argument('--randomseed', default=None)
    parser.add_argument('--save_stable_models', default=False, action='store_true')

    # arguments for GCN
    parser.add_argument('--X_dim', type=int, default=32)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--sim_func', type=str, default='embedded_gaussian')
    parser.add_argument('--layerwise_graph', default=False, action='store_true')
    parser.add_argument('--skip_connection', default=False, action='store_true')

    # arguments for training with scenarios with variable number of pedestrians in one episode
    parser.add_argument('--pretend_batch', default=False, action='store_true')

    sys_args = parser.parse_args()

    main(sys_args)
