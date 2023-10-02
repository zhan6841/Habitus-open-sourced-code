import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils.utils import load_args_from_file

from configs_gbdt import get_parser
from train_gbdt import GBDTModel, get_data_local

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import random
import glob
import errno

def eval_gbdt_10fold(FLAGS, LOG_DIR, features_list, labels_list):
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
    
    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    kf = KFold(n_splits=10, shuffle=True, random_state=50)
    kf.get_n_splits(X)
    
    count = 0
    for train_index, test_index in kf.split(X):
        idx = count

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # print(X_train.shape)
        # print(y_train.shape)
        # print(X_test.shape)
        # print(y_test.shape)

        existed_model_list = sorted(glob.glob(os.path.join(LOG_DIR, 'model%02d.pkl' % (idx))))
        if(len(existed_model_list) == 0):
            continue

        results_map = {}

        for model_save_path in existed_model_list:
            # print('Evaluate ' + model_save_path)
            model_name = model_save_path[-11:-4]

            myModel = GBDTModel(FLAGS)
            myModel.build_model()
            myModel.load_model(model_save_path)

            Y_test, Y_test_pred = myModel.eval_model(X_test, y_test)
            pred_save_path = os.path.join(EVAL_PRED_DIR, '%s_pred.csv' % (model_name))
            df = pd.DataFrame.from_dict({'Y_test':Y_test, 'Y_pred':Y_test_pred})
            df[['Y_test', 'Y_pred']].to_csv(pred_save_path)

            results_map[model_name] = {'MAE':myModel.eval_mae_list[0], 
                                        'RMSE':myModel.eval_rmse_list[0], 
                                        'MAR':myModel.eval_mar_list[0], 
                                        'ARE95':myModel.eval_are95_list[0], 
                                        'PARE10':myModel.eval_pare10_list[0]}
                
            del myModel

        result_save_path = os.path.join(EVAL_DIR, 'model%02d.csv' % (idx))
        print('Save result to ' + result_save_path)
        df = pd.DataFrame.from_dict(results_map).T
        df.to_csv(result_save_path)

        count += 1

def eval_gbdt_cross_scenario(FLAGS, LOG_DIR, features_list, labels_list, scenarios_included=[1,2,3,4,5,6,7,8,9,10]):
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

    for i in range(0, len(scenarios_included)):
        idx = scenarios_included[i]

        existed_model_list = sorted(glob.glob(os.path.join(LOG_DIR, 'model%02d.pkl' % (idx))))
        if(len(existed_model_list) == 0):
            continue
        
        X_test = np.concatenate(features_list[(i*3):((i+1)*3)], axis=0)
        y_test = np.concatenate(labels_list[(i*3):((i+1)*3)], axis=0)

        results_map = {}
            
        for model_save_path in existed_model_list:
            # print('Evaluate ' + model_save_path)
            model_name = model_save_path[-11:-4]
            
            myModel = GBDTModel(FLAGS)
            myModel.build_model()
            myModel.load_model(model_save_path)

            Y_test, Y_test_pred = myModel.eval_model(X_test, y_test)
            pred_save_path = os.path.join(EVAL_PRED_DIR, '%s_pred.csv' % (model_name))
            df = pd.DataFrame.from_dict({'Y_test':Y_test, 'Y_pred':Y_test_pred})
            df[['Y_test', 'Y_pred']].to_csv(pred_save_path)

            results_map[model_name] = {'MAE':myModel.eval_mae_list[0], 
                                        'RMSE':myModel.eval_rmse_list[0], 
                                        'MAR':myModel.eval_mar_list[0], 
                                        'ARE95':myModel.eval_are95_list[0], 
                                        'PARE10':myModel.eval_pare10_list[0]}
                
            del myModel

        result_save_path = os.path.join(EVAL_DIR, 'model%02d.csv' % (idx))
        # print('Save result to ' + result_save_path)
        df = pd.DataFrame.from_dict(results_map).T
        df.to_csv(result_save_path)

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    if(len(sys.argv) != 2):
        print("Usage: %s [model_dir]" % (sys.argv[0]))
        exit(0)

    MODEL_DIR = sys.argv[1]

    scenarios_included = [1,2,3,4,5,6,7,8,9,10]
    
    # get default parser and args
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    args_vars = vars(args)

    # load FLAGS from file
    FLAGS_path = os.path.join(MODEL_DIR, 'args.txt')
    FLAGS = load_args_from_file(FLAGS_path, args)
    print(FLAGS)

    # get features and labels
    features_list, labels_list = get_data_local(FLAGS, scenarios_included)

    # evaluate
    if(FLAGS.cross_scenario_validation):
        eval_gbdt_cross_scenario(FLAGS, MODEL_DIR, features_list, labels_list, scenarios_included)
    else:
        eval_gbdt_10fold(FLAGS, MODEL_DIR, features_list, labels_list)