import argparse
from crowd_nav.train import main


def get_default_args():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='cadrl')
    parser.add_argument('--config', type=str, default='config.py')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args(args=[])
    return args


def test_sarl():
    args = get_default_args()
    args.policy = 'sarl'
    args.config = 'configs/tests_config.py'
    args.overwrite = True
    main(args)


def test_cadrl():
    args = get_default_args()
    args.policy = 'cadlr'
    args.config = 'configs/tests_config.py'
    args.overwrite = True
    main(args)


def test_lstm_rl():
    args = get_default_args()
    args.policy = 'lstm_rl'
    args.config = 'configs/tests_config.py'
    args.overwrite = True
    main(args)


def test_gcn():
    args = get_default_args()
    args.policy = 'gcn'
    args.config = 'configs/tests_config.py'
    args.overwrite = True
    main(args)
