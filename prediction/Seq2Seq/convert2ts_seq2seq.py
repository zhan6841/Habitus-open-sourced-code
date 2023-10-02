import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils.utils import load_args_from_file

from dataset_seq2seq import Dataset
from configs_seq2seq import get_parser
from train_seq2seq import Seq2SeqModel, get_data_local

import torch
from torch.utils import data

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import random
import glob
import errno

def convert2ts_seq2seq_10fold(FLAGS, LOG_DIR, features_list, labels_list, MAX_EPOCH, EPOCH_STEP):
    EVAL_DIR = os.path.join(LOG_DIR, 'ts_models')
    if(not os.path.exists(EVAL_DIR)):
        # create LOG_DIR
        try:
            os.makedirs(EVAL_DIR)
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

        EVAL_PRED_DIR = os.path.join(EVAL_DIR, 'model%02d' % (idx))
        if(not os.path.exists(EVAL_PRED_DIR)):
            # create LOG_DIR
            try:
                os.makedirs(EVAL_PRED_DIR)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise  # This was not a "directory exist" error..

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
        num_output_features = test_dataset.num_output_features

        for model_save_path in to_eval_models:
            if model_save_path not in existed_model_list:
                continue
            print('Evaluate ' + model_save_path)
            model_name = model_save_path[-16:-4]

            myModel = Seq2SeqModel(FLAGS, num_input_features, num_output_features, logger)
            myModel.build_model()
            myModel.load_model(model_save_path)

            ts_model_save_path = os.path.join(EVAL_PRED_DIR, '%s.pt' % (model_name))
            myModel.save_model_in_torch_script(test_loader, ts_model_save_path)
            print('Save torch script model to %s' % (ts_model_save_path))
                
            del myModel

        count += 1

def convert2ts_seq2seq_cross_scenario(FLAGS, LOG_DIR, features_list, labels_list, MAX_EPOCH, EPOCH_STEP, scenarios_included=[1,2,3,4,5,6,7,8,9,10]):
    EVAL_DIR = os.path.join(LOG_DIR, 'ts_models')
    if(not os.path.exists(EVAL_DIR)):
        # create LOG_DIR
        try:
            os.makedirs(EVAL_DIR)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # This was not a "directory exist" error..

    logger = None
    for i in range(0, len(scenarios_included)):
        idx = scenarios_included[i]

        EVAL_PRED_DIR = os.path.join(EVAL_DIR, 'model%02d' % (idx))
        if(not os.path.exists(EVAL_PRED_DIR)):
            # create LOG_DIR
            try:
                os.makedirs(EVAL_PRED_DIR)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise  # This was not a "directory exist" error..

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
        num_output_features = test_dataset.num_output_features
            
        for model_save_path in to_eval_models:
            if model_save_path not in existed_model_list:
                continue
            print('Evaluate ' + model_save_path)
            model_name = model_save_path[-16:-4]
            
            myModel = Seq2SeqModel(FLAGS, num_input_features, num_output_features, logger)
            myModel.build_model()
            myModel.load_model(model_save_path)

            ts_model_save_path = os.path.join(EVAL_PRED_DIR, '%s.pt' % (model_name))
            myModel.save_model_in_torch_script(test_loader, ts_model_save_path)
            print('Save torch script model to %s' % (ts_model_save_path))
                
            del myModel

def convert2ts_seq2seq_no_test(FLAGS, LOG_DIR, features_list, labels_list, MAX_EPOCH, EPOCH_STEP, scenarios_included=[1,2,3,4,5,6,7,8,9,10]):
    EVAL_DIR = os.path.join(LOG_DIR, 'ts_models')
    if(not os.path.exists(EVAL_DIR)):
        # create LOG_DIR
        try:
            os.makedirs(EVAL_DIR)
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
        num_output_features = test_dataset.num_output_features

        myModel = Seq2SeqModel(FLAGS, num_input_features, num_output_features, logger)
        myModel.build_model()
        myModel.load_model(model_save_path)

        for i in range(0, len(scenarios_included)):
            for j in range(0, 3):
                idx = i*3 + j
                X_test = features_list[idx]
                y_test = labels_list[idx]
                
                test_dataset = Dataset(X_test, y_test)
                test_loader = data.DataLoader(dataset=test_dataset, batch_size=FLAGS.batch_size)
                
                ts_model_save_path = os.path.join(EVAL_DIR, '%s.pt' % (model_name))
                myModel.save_model_in_torch_script(test_loader, ts_model_save_path)
                print('Save torch script model to %s' % (ts_model_save_path))
                break
            break
        
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
    if(FLAGS.no_test or FLAGS.specific_scenario):
        convert2ts_seq2seq_no_test(FLAGS, MODEL_DIR, features_list, labels_list, MAX_EPOCH, EPOCH_STEP, scenarios_included)
    else:
        if(FLAGS.cross_scenario_validation):
            convert2ts_seq2seq_cross_scenario(FLAGS, MODEL_DIR, features_list, labels_list, MAX_EPOCH, EPOCH_STEP, scenarios_included)
        else:
            convert2ts_seq2seq_10fold(FLAGS, MODEL_DIR, features_list, labels_list, MAX_EPOCH, EPOCH_STEP)