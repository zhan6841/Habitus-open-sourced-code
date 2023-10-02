import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils.utils import load_args_from_file

from dataset_bp import Dataset
from configs_bp import get_parser
from train_bp import BPModel, get_data_local

import torch
from torch.utils import data

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import random
import glob
import errno

def eval_bp_10fold(FLAGS, LOG_DIR, features_list, labels_list, MAX_EPOCH, EPOCH_STEP):
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
    logger = None
    count = 0
    for train_index, test_index in kf.split(X):
        idx = count

        existed_model_list = sorted(glob.glob(os.path.join(LOG_DIR, 'model%02d_*.pth' % (idx))))
        print(existed_model_list)
        to_eval_models = []
        eval_epoch = EPOCH_STEP
        while(eval_epoch <= MAX_EPOCH):
            to_eval_models.append(os.path.join(LOG_DIR, 'model%02d_%04d.pth' % (idx, eval_epoch)))
            eval_epoch += EPOCH_STEP
        # print(to_eval_models)

        test_dataset = Dataset(X[test_index], y[test_index])
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=FLAGS.batch_size)
        num_input_features = test_dataset.num_input_features
        num_output_features = FLAGS.n_interval + 1

        results_map = {}

        for model_save_path in to_eval_models:
            if model_save_path not in existed_model_list:
                continue
            print('Evaluate ' + model_save_path)
            model_name = model_save_path[-16:-4]

            myModel = BPModel(FLAGS, num_input_features, num_output_features, logger)
            myModel.build_model()
            myModel.load_model(model_save_path)

            pred_save_path = os.path.join(EVAL_PRED_DIR, '%s_pred.csv' % (model_name))
            _, Y_test, Y_test_pred = myModel.eval_model(test_loader, pred_save_path, save_to_csv=False)
            # _, Y_test, Y_test_pred = myModel.eval_model(test_loader)
            # pred_save_path = os.path.join(EVAL_PRED_DIR, '%s_pred.csv' % (model_name))
            # df = pd.DataFrame.from_dict({'Y_test':Y_test, 'Y_pred':Y_test_pred})
            # df[['Y_test', 'Y_pred']].to_csv(pred_save_path)

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

def eval_bp_cross_scenario(FLAGS, LOG_DIR, features_list, labels_list, MAX_EPOCH, EPOCH_STEP, scenarios_included=[1,2,3,4,5,6,7,8,9,10]):
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

    logger = None
    for i in range(0, len(scenarios_included)):
        idx = scenarios_included[i]

        existed_model_list = sorted(glob.glob(os.path.join(LOG_DIR, 'model%02d_*.pth' % (idx))))
        print(existed_model_list)
        to_eval_models = []
        eval_epoch = EPOCH_STEP
        while(eval_epoch <= MAX_EPOCH):
            to_eval_models.append(os.path.join(LOG_DIR, 'model%02d_%04d.pth' % (idx, eval_epoch)))
            eval_epoch += EPOCH_STEP
        # print(to_eval_models)
        
        X_test = np.concatenate(features_list[(i*3):((i+1)*3)], axis=0)
        y_test = np.concatenate(labels_list[(i*3):((i+1)*3)], axis=0)
        test_dataset = Dataset(X_test, y_test)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=FLAGS.batch_size)
        num_input_features = test_dataset.num_input_features
        num_output_features = FLAGS.n_interval + 1

        results_map = {}
            
        for model_save_path in to_eval_models:
            if model_save_path not in existed_model_list:
                continue
            print('Evaluate ' + model_save_path)
            model_name = model_save_path[-16:-4]
            
            myModel = BPModel(FLAGS, num_input_features, num_output_features, logger)
            myModel.build_model()
            myModel.load_model(model_save_path)

            pred_save_path = os.path.join(EVAL_PRED_DIR, '%s_pred.csv' % (model_name))
            _, Y_test, Y_test_pred = myModel.eval_model(test_loader, pred_save_path, save_to_csv=True)

            results_map[model_name] = {'MAE':myModel.eval_mae_list[0], 
                                        'RMSE':myModel.eval_rmse_list[0], 
                                        'MAR':myModel.eval_mar_list[0], 
                                        'ARE95':myModel.eval_are95_list[0], 
                                        'PARE10':myModel.eval_pare10_list[0]}

            # result_save_path = os.path.join(EVAL_DIR, '%s.csv' % (model_name))
            # print('Save result to ' + result_save_path)
            # f = open(result_save_path, 'w')
            # for j in range(0, len(myModel.eval_mae_list)):
            #     f.write('%.6f,%.6f,%.6f,%.6f,%.6f\n' % (myModel.eval_mae_list[j], myModel.eval_rmse_list[j], myModel.eval_mar_list[j], myModel.eval_are95_list[j], myModel.eval_pare10_list[j]))
            # f.close()
                
            del myModel

        result_save_path = os.path.join(EVAL_DIR, 'model%02d.csv' % (idx))
        print('Save result to ' + result_save_path)
        df = pd.DataFrame.from_dict(results_map).T
        df.to_csv(result_save_path)

def eval_bp_no_test(FLAGS, LOG_DIR, features_list, labels_list, MAX_EPOCH, EPOCH_STEP, scenarios_included=[1,2,3,4,5,6,7,8,9,10]):
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

    logger = None
    
    existed_model_list = sorted(glob.glob(os.path.join(LOG_DIR, 'model_*.pth')))
    print(existed_model_list)
    to_eval_models = []
    eval_epoch = EPOCH_STEP
    while(eval_epoch <= MAX_EPOCH):
        to_eval_models.append(os.path.join(LOG_DIR, 'model_%04d.pth' % (eval_epoch)))
        eval_epoch += EPOCH_STEP
    # print(to_eval_models)

    for model_save_path in to_eval_models:
        if model_save_path not in existed_model_list:
            continue
        print('Evaluate ' + model_save_path)
        model_name = model_save_path[-14:-4]

        test_dataset = Dataset(features_list[0], labels_list[0])
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=FLAGS.batch_size)
        num_input_features = test_dataset.num_input_features
        num_output_features = num_output_features = FLAGS.n_interval + 1

        myModel = BPModel(FLAGS, num_input_features, num_output_features, logger)
        myModel.build_model()
        myModel.load_model(model_save_path)

        results_map = {}

        for i in range(0, len(scenarios_included)):
            for j in range(0, 3):
                idx = i*3 + j
                X_test = features_list[idx]
                y_test = labels_list[idx]
                
                test_dataset = Dataset(X_test, y_test)
                test_loader = data.DataLoader(dataset=test_dataset, batch_size=FLAGS.batch_size)
                
                pred_save_path = os.path.join(EVAL_PRED_DIR, '%s_pred%02d%02d.csv' % (model_name, scenarios_included[i], j+1))
                _, Y_test, Y_test_pred = myModel.eval_model(test_loader, pred_save_path, save_to_csv=True)

                key_name = '%02d%02d' % (scenarios_included[i], j+1)
                results_map[key_name] = {'MAE':myModel.eval_mae_list[-1], 
                                        'RMSE':myModel.eval_rmse_list[-1], 
                                        'MAR':myModel.eval_mar_list[-1], 
                                        'ARE95':myModel.eval_are95_list[-1], 
                                        'PARE10':myModel.eval_pare10_list[-1]}
        
        result_save_path = os.path.join(EVAL_DIR, '%s.csv' % (model_name))
        print('Save result to ' + result_save_path)
        df = pd.DataFrame.from_dict(results_map).T
        df.to_csv(result_save_path)
        
        del myModel

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if(len(sys.argv) != 4):
        print("Usage: %s [model_dir] [max_epoch] [epoch_step]" % (sys.argv[0]))
        exit(0)

    MODEL_DIR = sys.argv[1]
    MAX_EPOCH = int(sys.argv[2])
    EPOCH_STEP = int(sys.argv[3])

    scenarios_included = [1,2,3,4,5,6,7,8,9,10]
    
    # get default parser and args
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    args_vars = vars(args)

    # load FLAGS from file
    FLAGS_path = os.path.join(MODEL_DIR, 'args.txt')
    FLAGS = load_args_from_file(FLAGS_path, args)
    print(FLAGS)

    if(MAX_EPOCH == -1):
        MAX_EPOCH = FLAGS.num_epochs
    if(EPOCH_STEP == -1):
        EPOCH_STEP = FLAGS.save_steps

    # get features and labels
    features_list, labels_list = get_data_local(FLAGS, scenarios_included)

    # evaluate
    # evaluate
    if(FLAGS.no_test):
        eval_bp_no_test(FLAGS, MODEL_DIR, features_list, labels_list, MAX_EPOCH, EPOCH_STEP, scenarios_included)
    else:
        if(FLAGS.cross_scenario_validation):
            eval_bp_cross_scenario(FLAGS, MODEL_DIR, features_list, labels_list, MAX_EPOCH, EPOCH_STEP, scenarios_included)
        else:
            eval_bp_10fold(FLAGS, MODEL_DIR, features_list, labels_list, MAX_EPOCH, EPOCH_STEP)