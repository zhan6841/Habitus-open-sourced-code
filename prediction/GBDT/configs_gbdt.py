import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Model Augments')

    # dirs and names
    parser.add_argument('--trace_data_dir', type=str, default='../../Data/zed/trace/')
    parser.add_argument('--pose_data_dir', type=str, default='../../Data/zed/pose_lr2/')
    parser.add_argument('--model_save_dir', type=str, default='models/')

    # features
    parser.add_argument('--prediction_window', type=int, default=60)
    parser.add_argument('--series_history_window', type=int, default=1)
    parser.add_argument('--use_pose', action='store_true')

    # unused features
    parser.add_argument('--use_speed', action='store_true')
    parser.add_argument('--feature_mode', type=int, default=2)
    parser.add_argument('--label_mode', type=int, default=1)
    parser.add_argument('--n_pw', type=int, default=8)
    parser.add_argument('--n_interval', type=int, default=24)
    parser.add_argument('--tput_step', type=float, default=50.0)

    # models
    parser.add_argument('--model_id', type=str, default='model_default')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=3)

    # training
    parser.add_argument('--valid_size', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--cross_scenario_validation', action='store_true')
    parser.add_argument('--no_test', action='store_true') # use all data as training data, no test data

    return parser