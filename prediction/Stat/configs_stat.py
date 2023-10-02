import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Model Augments')

    # dirs and names
    parser.add_argument('--trace_data_dir', type=str, default='../../Data/zed/trace/')
    parser.add_argument('--model_save_dir', type=str, default='models/')

    # features
    parser.add_argument('--prediction_window', type=int, default=60)

    # models
    parser.add_argument('--model_id', type=str, default='model_default')

    # Moving Average
    parser.add_argument('--ma', action='store_true', help='Moving Average')
    parser.add_argument('--ma_n_samples', type=int, default=5)

    # Exponentially Weighted Moving Average
    parser.add_argument('--ewma', action='store_true', help='Exponentially Weighted Moving Average')
    parser.add_argument('--ewma_alpha', type=float, default=0.9)

    # Holt-Winters
    parser.add_argument('--hw', action='store_true', help='Holt-Winters')
    parser.add_argument('--hw_alpha', type=float, default=0.9)
    parser.add_argument('--hw_beta', type=float, default=0.1)

    # Harmonic Mean
    parser.add_argument('--harmonic', action='store_true', help='Harmonic Mean')
    parser.add_argument('--harmonic_n_samples', type=int, default=20)

    return parser