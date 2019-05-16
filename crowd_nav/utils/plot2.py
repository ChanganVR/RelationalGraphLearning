import re
import argparse
import os
import pprint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='darkgrid')


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_files', type=str, nargs='+')
    parser.add_argument('--plot_sr', default=False, action='store_true')
    parser.add_argument('--plot_cr', default=False, action='store_true')
    parser.add_argument('--plot_time', default=False, action='store_true')
    parser.add_argument('--plot_reward', default=True, action='store_true')
    parser.add_argument('--plot_train', default=True, action='store_true')
    parser.add_argument('--plot_val', default=False, action='store_true')
    parser.add_argument('--plot_all', default=False, action='store_true')
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--models', type=str, default=None)
    args = parser.parse_args()
    args.models = args.models.split(',')

    max_episodes = None
    _, ax4 = plt.subplots()

    if args.plot_all:
        log_dir = args.log_files[0]
        if not os.path.isdir(log_dir):
            parser.error('Input argument should be the directory containing all experiment folders')
        args.log_files = [os.path.join(log_dir, exp_dir, 'output.log') for exp_dir in os.listdir(log_dir)]

    args.log_files = sorted(args.log_files)

    if args.models:
        models = args.models
    else:
        models = [os.path.basename(log_file[:-11]) for log_file in args.log_files]
    models_dict = {}
    for model in models:
        models_dict[model] = list()

    df = pd.DataFrame(columns=['Step', 'Reward', 'Model'])
    for i, log_file in enumerate(args.log_files):
        # skip models not present
        model = None
        for m in models:
            if m in log_file:
                model = m
                models_dict[model].append(log_file)
                break
        if model is None:
            continue

        with open(log_file, 'r') as file:
            log = file.read()

        # val_pattern = r"VAL   in episode (?P<episode>\d+) has success rate: (?P<sr>[0-1].\d+), " \
        #               r"collision rate: (?P<cr>[0-1].\d+), nav time: (?P<time>\d+.\d+), " \
        #               r"total reward: (?P<reward>[-+]?\d+.\d+)"
        # val_episode = []
        # val_sr = []
        # val_cr = []
        # val_time = []
        # val_reward = []
        # for r in re.findall(val_pattern, log):
        #     val_episode.append(int(r[0]))
        #     val_sr.append(float(r[1]))
        #     val_cr.append(float(r[2]))
        #     val_time.append(float(r[3]))
        #     val_reward.append(float(r[4]))

        train_pattern = r"TRAIN in episode (?P<episode>\d+)  in epoch (?P<epoch>\d+) has success rate: (?P<sr>[0-1].\d+), " \
                        r"collision rate: (?P<cr>[0-1].\d+), nav time: (?P<time>\d+.\d+), " \
                        r"total reward: (?P<reward>[-+]?\d+.\d+), average return: (?P<return>[-+]?\d+.\d+)"
        train_episode = []
        train_sr = []
        train_cr = []
        train_time = []
        train_reward = []
        train_avg_return = []
        for r in re.findall(train_pattern, log):
            train_episode.append(int(r[0]))
            train_sr.append(float(r[2]))
            train_cr.append(float(r[3]))
            train_time.append(float(r[4]))
            train_reward.append(float(r[5]))
            train_avg_return.append(float(r[6]))
        if max_episodes is not None:
            train_episode = train_episode[:max_episodes]
            train_sr = train_sr[:max_episodes]
            train_cr = train_cr[:max_episodes]
            train_time = train_time[:max_episodes]
            train_reward = train_reward[:max_episodes]
            train_avg_return = train_avg_return[:max_episodes]

        # # smooth training plot
        # train_sr_smooth = running_mean(train_sr, args.window_size)
        # train_cr_smooth = running_mean(train_cr, args.window_size)
        # train_time_smooth = running_mean(train_time, args.window_size)
        train_reward_smooth = running_mean(train_reward, args.window_size)
        train_avg_return_smooth = running_mean(train_avg_return, args.window_size)

        smoothed_length = len(train_reward_smooth)
        df = df.append(pd.DataFrame({'Step': range(smoothed_length), 'Reward': train_reward_smooth,
                                     'Model': [model] * smoothed_length}), ignore_index=True)

        # smoothed_length = len(train_reward_smooth)
        # df = df.append(pd.DataFrame({'Step': range(smoothed_length), 'Reward': train_avg_return_smooth,
        #                              'Model': [model] * smoothed_length}), ignore_index=True)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(models_dict)

    sns.lineplot(x="Step", y='Reward', data=df, hue='Model', ax=ax4)

    plt.tick_params(axis='both', which='major')
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.125)

    plt.show()


if __name__ == '__main__':
    main()
