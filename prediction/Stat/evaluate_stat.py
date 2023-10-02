import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils.utils import load_trace_data, compute_metrics

from configs_stat import get_parser

import numpy as np
import pandas as pd

import random
import errno

def get_data_local(FLAGS, scenarios_included=[1,2,3,4,5,6,7,8,9,10]):
    ad_tput_list = []
    for i in scenarios_included:
        for j in range(1, 4):
            trace_filepath = os.path.join(FLAGS.trace_data_dir, "trace%02d%02d.txt" % (i, j))
            print(trace_filepath)
            dofs, frame_ids, ad_tputs, ad_signals, timestamps = load_trace_data(trace_filepath)
            ad_tput_list.append(ad_tputs)
    return ad_tput_list

# Moving Average
def moving_average(y_test, prediction_window=60, n_samples=5):
    y_pred = []
    index = []
    for i in range(0, n_samples):
        index.append(prediction_window*i)
    while(index[-1] < len(y_test) - prediction_window):
        th_est = 0
        for idx in index:
            th_est += y_test[idx]
        th_est /= n_samples
        y_pred.append(th_est)
        for i in range(0, len(index)):
            index[i] += 1
    return y_test[len(y_test)-len(y_pred):], y_pred

# Exponentially Weighted Moving Average
def exponentially_weighted_ma(y_test, prediction_window=60, alpha=0.9):
    y_pred = []
    v_list = [y_test[i] for i in range(0, prediction_window)]
    start_idx = 0
    for i in range(start_idx, len(y_test) - prediction_window):
        idx = i % prediction_window
        v_list[idx] = v_list[idx]*(1-alpha) + alpha*y_test[i]
        y_pred.append(v_list[idx])
    return y_test[len(y_test)-len(y_pred):], y_pred

# Holt-Winters
def holt_winters(y_test, prediction_window=60, alpha=0.9, beta=0.1):
    y_pred = []
    y_s = [y_test[i] for i in range(0, prediction_window)]
    y_s_last = [y_test[i] for i in range(0, prediction_window)]
    y_t = [(y_test[i+prediction_window] - y_test[i]) for i in range(0, prediction_window)]
    y_t_last = [(y_test[i+prediction_window] - y_test[i]) for i in range(0, prediction_window)]
    start_idx = prediction_window*2
    for i in range(start_idx, len(y_test) - prediction_window):
        idx = i % prediction_window
        y_f = y_s[idx]+y_t[idx]
        # update y_t
        tmp = y_t[idx]
        y_t[idx] = beta*(y_s[idx]-y_s_last[idx]) + (1-beta)*y_t_last[idx]
        y_t_last[idx] = tmp
        # update y_s
        y_s_last[idx] = y_s[idx]
        y_s[idx] = alpha*y_test[i] + (1 - alpha)*y_f

        y_f = y_s[idx]+y_t[idx]
        y_pred.append(y_f)
    return y_test[len(y_test)-len(y_pred):], y_pred

# Harmonic Mean
def harmonic_mean(y_test, prediction_window=60, n_samples=20):
    y_pred = []
    index = []
    for i in range(0, n_samples):
        index.append(prediction_window*i)
    while(index[-1] < len(y_test) - prediction_window):
        th_est = 0
        for idx in index:
            th_est += (1.0 / y_test[idx])
        th_est = n_samples / th_est
        y_pred.append(th_est)
        for i in range(0, len(index)):
            index[i] += 1
    return y_test[len(y_test)-len(y_pred):], y_pred

def eval_stat(FLAGS, LOG_DIR, ad_tput_list, scenarios_included=[1,2,3,4,10]):
    EVAL_DIR = os.path.join(LOG_DIR, 'eval')
    if(not os.path.exists(EVAL_DIR)):
        # create LOG_DIR
        try:
            os.makedirs(EVAL_DIR)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # This was not a "directory exist" error..

    EVAL_PRED_DIR = os.path.join(EVAL_DIR, 'pred')
    if(not os.path.exists(EVAL_PRED_DIR)):
        # create LOG_DIR
        try:
            os.makedirs(EVAL_PRED_DIR)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # This was not a "directory exist" error..

    results_map = {}
    for i in range(0, len(scenarios_included)):
        y_test_all = []
        y_pred_all = []
        for j in range(0, 3):
            idx = i*3+j
            if(FLAGS.ma):
                y_test, y_pred = moving_average(ad_tput_list[idx], FLAGS.prediction_window, FLAGS.ma_n_samples)
            elif(FLAGS.ewma):
                y_test, y_pred = exponentially_weighted_ma(ad_tput_list[idx], FLAGS.prediction_window, FLAGS.ewma_alpha)
            elif(FLAGS.hw):
                y_test, y_pred = holt_winters(ad_tput_list[idx], FLAGS.prediction_window, FLAGS.hw_alpha, FLAGS.hw_beta)
            elif(FLAGS.harmonic):
                y_test, y_pred = harmonic_mean(ad_tput_list[idx], FLAGS.prediction_window, FLAGS.harmonic_n_samples)
            else:
                print('Unknown algorithm!!!')
                exit(0)
            y_test_all += y_test
            y_pred_all += y_pred

        y_test_all = np.asarray(y_test_all)
        y_pred_all = np.asarray(y_pred_all)
        idx_all = y_test_all.argsort()
        y_test_all = y_test_all[idx_all]
        y_pred_all = y_pred_all[idx_all]
        pred_save_path = os.path.join(EVAL_PRED_DIR, '%02d_pred.csv' % (scenarios_included[i]))
        df = pd.DataFrame.from_dict({'Y_test':y_test_all, 'Y_pred':y_pred_all})
        df[['Y_test', 'Y_pred']].to_csv(pred_save_path)
        
        mae, rmse, mar, are95, pare10 = compute_metrics(y_test_all, y_pred_all)
        print('MAE: %.6f, RMSE: %.6f, MAR: %.6f, ARE95: %.6f, PARE10: %.6f' % (mae, rmse, mar, are95, pare10))
        results_map[scenarios_included[i]] = {'MAE':mae, 'RMSE':rmse, 'MAR':mar, 'ARE95':are95, 'PARE10':pare10}
    result_save_path = os.path.join(EVAL_DIR, 'model.csv')
    print('Save result to ' + result_save_path)
    df = pd.DataFrame.from_dict(results_map).T
    df.to_csv(result_save_path)

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # parse args first
    parser = get_parser()
    FLAGS, unknown = parser.parse_known_args()
    print(FLAGS)

    scenarios_included = [1,2,3,4,5,6,7,8,9,10]
    ad_tput_list = get_data_local(FLAGS, scenarios_included)

    LOG_DIR = os.path.join(FLAGS.model_save_dir, FLAGS.model_id)
    if(not os.path.exists(LOG_DIR)):
        # create LOG_DIR
        try:
            os.makedirs(LOG_DIR)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # This was not a "directory exist" error..
        # save args
        with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as log:
            for arg in sorted(vars(FLAGS)):
                log.write(arg + ': ' + str(getattr(FLAGS, arg)) + '\n')  # log of arguments

    eval_stat(FLAGS, LOG_DIR, ad_tput_list, scenarios_included)