import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils.utils import prepare_data_pairs_gbdt, compute_metrics, load_pose_data, load_trace_data, process_posetrace, load_trace_data_loc03

from configs_gbdt import get_parser

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

import random
import time
import pickle
import errno

class GBDTModel(object):
    def __init__(self, args):
        self.lr = args.learning_rate
        self.n_estimators = args.n_estimators
        self.max_depth = args.max_depth

        self.eval_mae_list = []
        self.eval_rmse_list = []
        self.eval_mar_list = []
        self.eval_are95_list = []
        self.eval_pare10_list = []

    def build_model(self):
        self.model = GradientBoostingRegressor(learning_rate=self.lr, n_estimators=self.n_estimators, max_depth=self.max_depth)
    
    def train_model(self, X_train, y_train):
        t0 = time.time()
        self.model.fit(X_train, y_train.ravel())
        t1 = time.time()
        print("Training Time: %.3f" % (t1-t0))
    
    def eval_model(self, X_test, y_test, save_path='test.csv', save_to_csv=False):
        y_pred = self.model.predict(X_test)
        y_pred = y_pred.reshape(-1, 1)
        out = np.concatenate([y_test, y_pred], axis=1)

        col_names = ['true_bw', 'pred_bw']
        df = pd.DataFrame(out, columns=col_names)

        if(save_to_csv):
            df.to_csv(save_path)

        Y_eval = np.array(df['true_bw'])
        Y_eval_pred = np.array(df['pred_bw'])
        # print(Y_eval.shape)
        # print(Y_eval_pred.shape)

        Y_test = Y_eval
        Y_test_pred = Y_eval_pred

        Y_test = Y_test.reshape(-1)
        Y_test_pred = Y_test_pred.reshape(-1)
        idx = Y_test.argsort()
        Y_test = Y_test[idx]
        Y_test_pred = Y_test_pred[idx]

        mae, rmse, mar, are95, pare10 = compute_metrics(Y_test, Y_test_pred)
        # print('MAE: %.6f, RMSE: %.6f, MAR: %.6f, ARE95: %.6f, PARE10: %.6f' % (mae, rmse, mar, are95, pare10))
        print('%.6f,%.6f,%.6f,%.6f,%.6f' % (mae, rmse, mar, are95, pare10))

        self.eval_mae_list.append(mae)
        self.eval_rmse_list.append(rmse)
        self.eval_mar_list.append(mar)
        self.eval_are95_list.append(are95)
        self.eval_pare10_list.append(pare10)
        
        return Y_test, Y_test_pred

    def load_model(self, path):
        self.model = pickle.load(open(path, 'rb'))

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

def get_data_local(FLAGS, scenarios_included=[1,2,3,4,5,6,7,8,9,10]):
    features_list = []
    labels_list = []

    features_included={"6DoF":True,"Throughput":True,"Signal":True,"Pose":FLAGS.use_pose,"Speed":False}

    data_list = {}

    for i in scenarios_included:
        for j in range(1, 4):
            trace_filepath = os.path.join(FLAGS.trace_data_dir, "trace%02d%02d.txt" % (i, j))
            print(trace_filepath)
            if ('loc03' in FLAGS.trace_data_dir):
                if ('5051' in FLAGS.pose_data_dir):
                    dofs, frame_ids, ad_tputs, ad_signals, timestamps = load_trace_data_loc03(trace_filepath, 5051)
                elif ('5052' in FLAGS.pose_data_dir):
                    dofs, frame_ids, ad_tputs, ad_signals, timestamps = load_trace_data_loc03(trace_filepath, 5052)
                else:
                    dofs, frame_ids, ad_tputs, ad_signals, timestamps = load_trace_data_loc03(trace_filepath, 5053)
            else:
                dofs, frame_ids, ad_tputs, ad_signals, timestamps = load_trace_data(trace_filepath)
            processed_posetrace = []
            if(FLAGS.use_pose):
                # pose_filepath = os.path.join(FLAGS.pose_data_dir, "pose_lerp%02d%02d.txt" % (i, j))
                pose_filepath = os.path.join(FLAGS.pose_data_dir, "pose%02d%02d.txt" % (i, j))
                print(pose_filepath)
                posetrace = load_pose_data(pose_filepath)
                processed_posetrace = process_posetrace(frame_ids,posetrace,with_confidence=True)
            data_list = {"6DoF":dofs, 
                        "Throughput":ad_tputs, 
                        "Signal":ad_signals, 
                        "Pose":processed_posetrace, 
                        "Speed":[]
                        }
            features, labels = prepare_data_pairs_gbdt(data_list=data_list, 
                                                features_included=features_included, 
                                                prediction_window=FLAGS.prediction_window, 
                                                series_history_window=FLAGS.series_history_window)
            if(features.shape[1] == 1):
                features = features.reshape(features.shape[0], features.shape[-1])
            # print(features.shape)
            # print(labels.shape)
            features_list.append(features)
            labels_list.append(labels)
    return features_list, labels_list

def train_gbdt_10fold(FLAGS, LOG_DIR, features_list, labels_list):
    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=50)
    kf.get_n_splits(X)

    count = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)

        myModel = GBDTModel(FLAGS)
        myModel.build_model()

        print('Training Model%02d...' % (count))
        model_save_path = os.path.join(LOG_DIR, 'model%02d.pkl' % (count))
        if(os.path.exists(model_save_path)):
            continue

        myModel.train_model(X_train, y_train)
        myModel.save_model(model_save_path)

        del myModel
        count += 1

def train_gbdt_cross_scenario(FLAGS, LOG_DIR, features_list, labels_list, scenarios_included=[1,2,3,4,5,6,7,8,9,10]):
    for i in range(0, int(len(features_list)/3)):
        X_train = np.concatenate(features_list[:(i*3)]+features_list[((i+1)*3):], axis=0)
        y_train = np.concatenate(labels_list[:(i*3)]+labels_list[((i+1)*3):], axis=0)

        X_test = np.concatenate(features_list[(i*3):((i+1)*3)], axis=0)
        y_test = np.concatenate(labels_list[(i*3):((i+1)*3)], axis=0)

        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)

        myModel = GBDTModel(FLAGS)
        myModel.build_model()

        print('Training Model%02d...' % (scenarios_included[i]))
        model_save_path = os.path.join(LOG_DIR, 'model%02d.pkl' % (scenarios_included[i]))
        if(os.path.exists(model_save_path)):
            continue

        myModel.train_model(X_train, y_train)
        myModel.save_model(model_save_path)

        del myModel

def train_gbdt_no_test(FLAGS, LOG_DIR, features_list, labels_list):
    X_train = np.concatenate(features_list, axis=0)
    y_train = np.concatenate(labels_list, axis=0)

    myModel = GBDTModel(FLAGS)
    myModel.build_model()

    print('Training Model...')
    model_save_path = os.path.join(LOG_DIR, 'model.pkl')
    if(not os.path.exists(model_save_path)):
        myModel.train_model(X_train, y_train)
        myModel.save_model(model_save_path)
    del myModel

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # parse args first
    parser = get_parser()
    FLAGS, unknown = parser.parse_known_args()
    print(FLAGS)

    # load data and prepare data pairs
    scenarios_included=[1,2,3,4,5,6,7,8,9,10]
    features_list, labels_list = get_data_local(FLAGS, scenarios_included)

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

    # train
    if(FLAGS.no_test):
        train_gbdt_no_test(FLAGS, LOG_DIR, features_list, labels_list)
    else:
        if(FLAGS.cross_scenario_validation):
            train_gbdt_cross_scenario(FLAGS, LOG_DIR, features_list, labels_list, scenarios_included)
        else:
            train_gbdt_10fold(FLAGS, LOG_DIR, features_list, labels_list)