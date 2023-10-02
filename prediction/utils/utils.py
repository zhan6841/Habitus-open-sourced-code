import csv
import numpy as np
import logging
import os
from datetime import datetime
import argparse
import glob
import math
import errno
import pandas as pd
import math
import matplotlib.pyplot as plt

def load_trace_data_loc03(file_path, port):
    TRACE_FILE_NAMES = ['timestamp','h_x','h_y','h_z','h_r_x','h_r_y','h_r_z',
                        's_r_x','s_r_y','s_r_z','s_r_w','s_x','s_y','s_z',
                        'signal_ad','signal_ac','throughput_ad','throughput_ac',
                        '5051','5052','5053']
    # print(file_path)
    dofs = [[], [], [], [], [], []]
    frame_ids = []
    ad_tputs = []
    ad_signals = []
    timestamps = []
    trace_df = pd.read_csv(file_path, names=TRACE_FILE_NAMES)
    for i in range(0, len(trace_df['timestamp'])):
        if(len(timestamps) == 0 and trace_df['throughput_ad'][i]/1000000.0 < 500.0):
            continue
        if(len(timestamps) == 0):
            base_ts = trace_df['timestamp'][i]
        dofs[0].append(trace_df['h_x'][i])
        dofs[1].append(trace_df['h_y'][i])
        dofs[2].append(trace_df['h_z'][i])
        dofs[3].append(trace_df['h_r_x'][i] * math.pi / 180.0)
        dofs[4].append(trace_df['h_r_y'][i] * math.pi / 180.0)
        dofs[5].append(trace_df['h_r_z'][i] * math.pi / 180.0)
        frame_ids.append(int(trace_df[str(port)][i]))
        if(trace_df['throughput_ad'][i] == 0.0):
            ad_tputs.append(ad_tputs[-1])
        else:
            ad_tputs.append(trace_df['throughput_ad'][i]/1000000.0) # convert to Mbps
        ad_signals.append(trace_df['signal_ad'][i]) # signal strength of 802.11ad
        ts = trace_df['throughput_ad'][i] - base_ts
        if(len(timestamps) > 0 and ts < timestamps[-1]):
            timestamps.append(timestamps[-1]+0.016)
        else:
            timestamps.append(ts)

    for i in range(1, len(ad_signals)):
        if(ad_signals[i] > -10.0):
            ad_signals[i] = ad_signals[i-1]

    return dofs, frame_ids, ad_tputs, ad_signals, timestamps

def load_trace_data(file_path):
    # print(file_path)
    dofs = [[], [], [], [], [], []]
    frame_ids = []
    ad_tputs = []
    ad_signals = []
    timestamps = []
    f = open(file_path, 'r')
    csv_reader = csv.reader(f, delimiter=',')
    base_ts = 0.0
    for row in csv_reader:
        if(len(timestamps) == 0 and float(row[-3])/1000000.0 < 500.0):
            continue
        if(len(timestamps) == 0):
            base_ts = float(row[-7])
            # print("trace index = %d" % int(row[0]))
        dofs[0].append(float(row[1]))
        dofs[1].append(float(row[2]))
        dofs[2].append(float(row[3]))
        dofs[3].append(float(row[7]))
        dofs[4].append(float(row[8]))
        dofs[5].append(float(row[9]))
        frame_ids.append(int(row[11]))
        if(float(row[-3]) == 0.0):
            ad_tputs.append(ad_tputs[-1])
        else:
            ad_tputs.append(float(row[-3])/1000000.0) # convert to Mbps
        ad_signals.append(float(row[-5])) # signal strength of 802.11ad
        ts = float(row[-7])-base_ts
        if(len(timestamps) > 0 and ts < timestamps[-1]):
            timestamps.append(timestamps[-1]+0.016)
        else:
            timestamps.append(ts)
    f.close()

    for i in range(1, len(ad_signals)):
        if(ad_signals[i] > -10.0):
            ad_signals[i] = ad_signals[i-1]

    return dofs, frame_ids, ad_tputs, ad_signals, timestamps

def load_pose_data(file_path):
    # print(file_path)
    # key: frame_id, value: {body_part: [x, y, z, w]}
    posetrace = {}
    f = open(file_path, 'r')
    csv_reader = csv.reader(f, delimiter=',')
    for row in csv_reader:
        frame_id = int(row[0])
        pose = {}
        for i in range(0, int((len(row)-1)/4)):
            x = float(row[1+i*4+0])
            y = float(row[1+i*4+1])
            z = float(row[1+i*4+2])
            w = float(row[1+i*4+3])
            pose[i] = [x, y, z, w]
        posetrace[frame_id] = pose
    f.close()

    return posetrace

# y_test and y_pred must be numpy array
def compute_metrics(y_test, y_pred):
    y_diff = abs(y_pred - y_test)
    mae = y_diff.mean()
    mse = (y_diff ** 2).mean()
    rmse = np.sqrt(mse)
    mar_array = y_diff / y_test
    mar = np.average(mar_array)
    are95 = np.percentile(mar_array, 95, interpolation='higher')
    pare10 = np.sum(mar_array<0.1)/mar_array.shape[0]

    return mae, rmse, mar, are95, pare10

# frame_id: []
# posetrace: {key: frame_id, value: {body_part: [x, y, z, w]}}
def process_posetrace(frame_ids, posetrace, with_confidence=False):
    processed_posetrace = []
    for frame_id in frame_ids:
        pose = []
        for key in sorted(posetrace[frame_id].keys()):
            if(with_confidence == True):
                pose += posetrace[frame_id][key]
            else:
                pose += posetrace[frame_id][key][:-1]
        processed_posetrace.append(pose)
    return processed_posetrace

def prepare_data_pairs_gbdt(data_list={"6DoF":[],"Throughput":[],"Signal":[],"Pose":[],"Speed":[]}, 
                        features_included={"6DoF":True,"Throughput":True,"Signal":True,"Pose":True,"Speed":False},
                        prediction_window=60, 
                        series_history_window=36):
    dofs = data_list["6DoF"]
    ad_tputs = data_list["Throughput"]
    ad_signals = data_list["Signal"]
    poses = data_list["Pose"]
    speeds = data_list["Speed"]

    features = []
    labels = []

    num_samples = len(ad_tputs)
    sample_start_idx = series_history_window - 1
    # print("sample_start_idx = %d" % sample_start_idx)
    for i in range(sample_start_idx, num_samples-prediction_window):
        # construct features
        tmp = []
        start_idx = i - series_history_window + 1
        feature_end = series_history_window
        for j in range(0, feature_end):
            # decide idx
            dofs_idx = start_idx+j
            signals_idx = start_idx+j
            tputs_idx = start_idx+j
            poses_idx = start_idx+j
            speeds_idx = start_idx+j
            
            features_1d = []
            if(features_included["6DoF"] == True):
                for dof in dofs:
                    features_1d.append(dof[dofs_idx])
            if(features_included["Pose"] == True):
                features_1d += poses[poses_idx]
            if(features_included["Throughput"] == True):
                features_1d.append(ad_tputs[tputs_idx])
            if(features_included["Signal"] == True):
                features_1d.append(ad_signals[signals_idx])
            if(features_included["Speed"] == True):
                for speed in speeds:
                    features_1d.append(speed[speeds_idx])
            tmp += features_1d
        features.append(tmp)

        # construct labels
        labels.append([ad_tputs[i+prediction_window]])

    return np.asarray(features), np.asarray(labels)

def prepare_data_pairs_bp(data_list={"6DoF":[],"Throughput":[],"Signal":[],"Pose":[],"Speed":[]}, 
                        features_included={"6DoF":True,"Throughput":True,"Signal":True,"Pose":True,"Speed":False},
                        prediction_window=60, 
                        n_pw=8):
    dofs = data_list["6DoF"]
    ad_tputs = data_list["Throughput"]
    ad_signals = data_list["Signal"]
    poses = data_list["Pose"]
    speeds = data_list["Speed"]

    features = []
    labels = []

    num_samples = len(ad_tputs)
    sample_start_idx = prediction_window * (n_pw - 1)
    # print("sample_start_idx = %d" % sample_start_idx)
    for i in range(sample_start_idx, num_samples-prediction_window):
        # construct features
        tmp = []
        start_idx = i - (n_pw - 1)*prediction_window
        feature_end = n_pw

        # decide idx
        dofs_idx = i
        signals_idx = i
        poses_idx = i
        speeds_idx = i
        for j in range(0, feature_end):
            # decide idx
            tputs_idx = start_idx+j*prediction_window

            if(features_included["Throughput"] == True):
                tmp.append(ad_tputs[tputs_idx])

        if(features_included["6DoF"] == True):
            for dof in dofs:
                tmp.append(dof[dofs_idx])
        if(features_included["Pose"] == True):
            tmp += poses[poses_idx]
        if(features_included["Signal"] == True):
            tmp.append(ad_signals[signals_idx])
        if(features_included["Speed"] == True):
            for speed in speeds:
                tmp.append(speed[speeds_idx])

        features.append(tmp)

        # construct labels
        labels.append([ad_tputs[i+prediction_window]])

    return np.asarray(features), np.asarray(labels)

def prepare_data_pairs_rnn(data_list={"6DoF":[],"Throughput":[],"Signal":[],"Pose":[],"Speed":[]}, 
                        features_included={"6DoF":True,"Throughput":True,"Signal":True,"Pose":True,"Speed":False},
                        prediction_window=60, 
                        n_pw=8):
    dofs = data_list["6DoF"]
    ad_tputs = data_list["Throughput"]
    ad_signals = data_list["Signal"]
    poses = data_list["Pose"]
    speeds = data_list["Speed"]

    features = []
    labels = []

    num_samples = len(ad_tputs)
    sample_start_idx = prediction_window * (n_pw - 1)
    # print("sample_start_idx = %d" % sample_start_idx)
    for i in range(sample_start_idx, num_samples-prediction_window):
        # construct features
        tmp = []
        start_idx = i - (n_pw - 1)*prediction_window
        feature_end = n_pw
        for j in range(0, feature_end):
            # decide idx
            dofs_idx = start_idx+j*prediction_window
            signals_idx = start_idx+j*prediction_window
            tputs_idx = start_idx+j*prediction_window
            poses_idx = start_idx+j*prediction_window
            speeds_idx = start_idx+j*prediction_window
            
            features_1d = []
            if(features_included["6DoF"] == True):
                for dof in dofs:
                    features_1d.append(dof[dofs_idx])
            if(features_included["Pose"] == True):
                features_1d += poses[poses_idx]
            if(features_included["Throughput"] == True):
                features_1d.append(ad_tputs[tputs_idx])
            if(features_included["Signal"] == True):
                features_1d.append(ad_signals[signals_idx])
            if(features_included["Speed"] == True):
                for speed in speeds:
                    features_1d.append(speed[speeds_idx])
            tmp.append(features_1d)

        features.append(tmp)

        # construct labels
        labels.append([ad_tputs[i+prediction_window]])

    return np.asarray(features), np.asarray(labels)

def prepare_data_pairs_seq2seq(data_list={"6DoF":[],"Throughput":[],"Signal":[],"Pose":[],"Speed":[]}, 
                        features_included={"6DoF":True,"Throughput":True,"Signal":True,"Pose":True,"Speed":False},
                        prediction_window=60, 
                        series_history_window=36):
    dofs = data_list["6DoF"]
    ad_tputs = data_list["Throughput"]
    ad_signals = data_list["Signal"]
    poses = data_list["Pose"]
    speeds = data_list["Speed"]

    features = []
    labels = []

    num_samples = len(ad_tputs)
    sample_start_idx = series_history_window - 1
    # print("sample_start_idx = %d" % sample_start_idx)

    for i in range(sample_start_idx, num_samples-prediction_window):
        # construct features
        tmp = []
        start_idx = i - series_history_window + 1
        feature_end = series_history_window
        for j in range(0, feature_end):
            # decide idx
            dofs_idx = start_idx+j
            signals_idx = start_idx+j
            tputs_idx = start_idx+j
            poses_idx = start_idx+j
            speeds_idx = start_idx+j
            
            features_1d = []
            if(features_included["6DoF"] == True):
                for dof in dofs:
                    features_1d.append(dof[dofs_idx])
            if(features_included["Pose"] == True):
                features_1d += poses[poses_idx]
            if(features_included["Throughput"] == True):
                features_1d.append(ad_tputs[tputs_idx])
            if(features_included["Signal"] == True):
                features_1d.append(ad_signals[signals_idx])
            if(features_included["Speed"] == True):
                for speed in speeds:
                    features_1d.append(speed[speeds_idx])
            tmp.append(features_1d)

        features.append(tmp)

        # construct labels
        labels.append([ad_tputs[(i+1):(i+prediction_window+1)]])
        # labels.append([[ad_tputs[k]] for k in range((i+1), (i+prediction_window+1))])

    return np.asarray(features), np.asarray(labels)

def setup_logger(name, log_file, level=logging.DEBUG):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

def create_log_dir(log_dir):
    log_dir_path = os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.makedirs(log_dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
    return log_dir_path

def create_model_dir(model_save_dir, model_id):
    log_dir_path = os.path.join(model_save_dir, model_id)
    if(os.path.exists(log_dir_path)):
        return log_dir_path
    try:
        os.makedirs(log_dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
    return log_dir_path

def parse_logs(log_path):
    # MAE, RMSE, MAR, ARE95, PARE10, Train Loss, Eval Loss
    value_list = []
    for i in range(0, 9):
        value_list.append([])

    f = open(log_path, 'r')
    lines = f.readlines()
    f.close()
    for i in range(0, len(lines)):
        val = float(lines[i].split(' ')[-1][0:-1])
        tmp = i % 10
        if(tmp > 0):
            value_list[tmp-1].append(val)
    
    return value_list

def load_args_from_file(file_path, args_default):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()

    args_map = {}
    for line in lines:
        tmp = line.split(': ')
        args_map[tmp[0]] = tmp[1][:-1]
    # print(args_map)

    args_default_vars = vars(args_default)
    keys_in_file = args_map.keys()
    for key in args_default_vars.keys():
        if(key in keys_in_file):
            if(args_map[key] == 'True' or args_map[key] == 'False'):
                setattr(args_default, key, eval(args_map[key]))
            else:
                setattr(args_default, key, type(args_default_vars[key])(args_map[key]))
    # print(args_default)

    return args_default

if __name__ == '__main__':
    print('utils')