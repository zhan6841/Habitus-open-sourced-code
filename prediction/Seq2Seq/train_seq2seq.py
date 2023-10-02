import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils.utils import prepare_data_pairs_seq2seq, compute_metrics, setup_logger, load_pose_data, load_trace_data, process_posetrace, load_args_from_file, load_trace_data_loc03

from dataset_seq2seq import Dataset
from model_seq2seq import Seq2Seq, EncoderRNN, DecoderRNN
from configs_seq2seq import get_parser

import torch
from torch.utils import data
import torch.optim as optim
from torch import nn
from torch.utils.mobile_optimizer import optimize_for_mobile

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold

import random
import time
import glob
import errno

import matplotlib.pyplot as plt

class Seq2SeqModel(object):
    def __init__(self, args, num_input_features, num_output_features, logger):
        self.hidden_dim = args.hidden_dim
        self.num_rnn_layers = args.num_rnn_layers
        self.drop_prob = args.drop_prob
        self.lambda_regression_reg = args.lambda_regression_reg
        self.device = torch.device('cuda', args.gpu_device)
        # self.device = torch.device('cpu')
        self.lr = args.learning_rate
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        # self.train_loader = train_loader
        # self.eval_loader = eval_loader
        self.train_loss_list = []
        self.eval_loss_list = []
        self.eval_mae_list = []
        self.eval_rmse_list = []
        self.eval_mar_list = []
        self.eval_are95_list = []
        self.eval_pare10_list = []

        self.logger = logger

    def build_model(self):
        self.encoder = EncoderRNN(self.num_input_features, self.hidden_dim, self.num_rnn_layers, self.drop_prob).to(self.device)
        self.decoder = DecoderRNN(self.num_output_features, self.hidden_dim, self.num_rnn_layers, self.drop_prob).to(self.device)
        self.model = Seq2Seq(self.encoder, self.decoder, self.device).to(self.device)

        self.loss_regression_func = nn.MSELoss()
        # self.loss_regression_func = nn.L1Loss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def compute_loss(self, Y_seq, pred_regression_seq):
        loss_regression = self.lambda_regression_reg * self.loss_regression_func(Y_seq, pred_regression_seq)
        return loss_regression

    def compute_loss_eval(self, Y_seq, pred_regression_seq):
        loss_regression = self.lambda_regression_reg * self.loss_regression_func(Y_seq, pred_regression_seq)
        rmse = torch.sqrt(loss_regression)
        return loss_regression, rmse

    def train_per_epoch(self, train_loader, valid_loader):
        start_time = time.time()
        self.model.train()
        total_samples = 0
        total_loss = 0
        num_batches = 0

        for idx_batch, batch in enumerate(train_loader):
            X_seq = batch[0].float().to(self.device)
            Y_seq = batch[1].float().to(self.device)

            self.optimizer.zero_grad()
            logits_regression_seq = self.model(X_seq, Y_seq)
            loss = self.compute_loss(Y_seq, logits_regression_seq)
            loss.backward()
            self.optimizer.step()

            total_samples += logits_regression_seq.shape[0]
            total_loss += loss.item()
            num_batches = num_batches + 1
        
        loss_train_per_sample = total_loss / num_batches
        end_time = time.time()

        # test/evaluate the model
        loss_eval_per_sample, Y_test, Y_test_pred = self.eval_model(valid_loader)

        print('Train Loss Per Sample: %.6f\nEval Loss Per Sample: %.6f' % (loss_train_per_sample, loss_eval_per_sample))
        if(self.logger != None):
            self.logger.info('Train Loss Per Sample: %.6f' % (loss_train_per_sample))
            self.logger.info('Eval Loss Per Sample: %.6f' % (loss_eval_per_sample))

        self.train_loss_list.append(loss_train_per_sample)
        print('Time: %.6f' % (end_time-start_time))
        self.logger.info('Time: %.6f' % (end_time-start_time))
    
    def eval_model(self, eval_loader, save_path='test.csv', save_to_csv=False):
        self.model.eval()
        total_samples = 0
        total_loss = 0
        total_rmse_loss = 0
        num_batches = 0
        all_list = []
        
        with torch.no_grad():
            for idx_batch, batch in enumerate(eval_loader):
                X_seq = batch[0].float().to(self.device)
                Y_seq = batch[1].float().to(self.device)
                
                logits_regression_seq = self.model(X_seq, Y_seq, teacher_forcing_ratio=0.0)
                
                # compute MAE, RMSE
                loss, loss_rmse = self.compute_loss_eval(Y_seq, logits_regression_seq)
                
                total_samples += logits_regression_seq.shape[0]
                
                total_loss += loss.item()
                total_rmse_loss += loss_rmse.item()
                num_batches = num_batches + 1
                
                Y_true = Y_seq.squeeze(dim=-1).reshape(-1, 1).data.cpu().numpy()
                Y_pred = logits_regression_seq.squeeze(dim=-1).reshape(-1, 1).data.cpu().numpy()
                
                out = np.concatenate([Y_true, Y_pred], axis=1)
                all_list.append(out)
                
        # compute average MAE, RMSE
        loss_eval_per_sample = total_loss / num_batches
        loss_rmse_eval_per_sample = total_rmse_loss / num_batches

        col_names = ['true_bw', 'pred_bw']
        df = pd.DataFrame(np.concatenate(all_list), columns=col_names)

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
        print('MAE: %.6f, RMSE: %.6f, MAR: %.6f, ARE95: %.6f, PARE10: %.6f' % (mae, rmse, mar, are95, pare10))
        # print('MAE=%.6f\nRMSE=%.6f\nMAR=%.6f\nARE95=%.6f\nPARE10=%.6f' % (mae, rmse, mar, are95, pare10))
        if(self.logger != None):
            self.logger.info('MAE: %.6f' % (mae))
            self.logger.info('RMSE: %.6f' % (rmse))
            self.logger.info('MAR: %.6f' % (mar))
            self.logger.info('ARE95: %.6f' % (are95))
            self.logger.info('PARE10: %.6f' % (pare10))
        
        self.eval_loss_list.append(loss_eval_per_sample)
        self.eval_mae_list.append(mae)
        self.eval_rmse_list.append(rmse)
        self.eval_mar_list.append(mar)
        self.eval_are95_list.append(are95)
        self.eval_pare10_list.append(pare10)
        
        return loss_eval_per_sample, Y_test, Y_test_pred

    def load_model(self, path):
        model_filename = path
        model_state = torch.load(model_filename)
        self.model.load_state_dict(model_state['model_state_dict'])
        return model_state

    def save_model(self, path):
        self.model.cpu()
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cpu()

        model_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if(torch.__version__ == '1.3.0'):
            torch.save(model_state, path)
        else:
            torch.save(model_state, path, _use_new_zipfile_serialization=False)

        self.model.to(self.device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def save_model_in_torch_script(self, eval_loader, path):
        with torch.no_grad():
            for idx_batch, batch in enumerate(eval_loader):
                X_seq = batch[0].float().to(self.device)
                Y_seq = batch[1].float().to(self.device)
                traced_script_module = torch.jit.trace(self.model, (X_seq, Y_seq, torch.tensor(0.0)))
                traced_script_module.save(path)
                break

    def save_model_in_torch_script_mobile(self, eval_loader, path):
        with torch.no_grad():
            for idx_batch, batch in enumerate(eval_loader):
                X_seq = batch[0].float().to(self.device)
                Y_seq = batch[1].float().to(self.device)
                traced_script_module = torch.jit.trace(self.model, (X_seq, Y_seq, torch.tensor(0.0)))
                traced_script_module_optimized = optimize_for_mobile(traced_script_module)
                traced_script_module_optimized._save_for_lite_interpreter(path)
                break

def get_data_local(FLAGS, scenarios_included=[1,2,3,4,5,6,7,8,9,10]):
    features_list = []
    labels_list = []

    features_included={"6DoF":True,"Throughput":True,"Signal":True,"Pose":FLAGS.use_pose,"Speed":False}

    data_list = {}

    for i in scenarios_included:
        for j in range(1, 4):
            trace_filepath = os.path.join(FLAGS.trace_data_dir, "trace%02d%02d.txt" % (i, j))
            if (not os.path.exists(trace_filepath)):
                continue
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
            features, labels = prepare_data_pairs_seq2seq(data_list=data_list, 
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

def train_seq2seq_10fold(FLAGS, LOG_DIR, features_list, labels_list):
    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=50)
    kf.get_n_splits(X)

    count = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=FLAGS.valid_size, random_state=50)
        
        print(X_train.shape)
        print(y_train.shape)
        print(X_eval.shape)
        print(y_eval.shape)
        
        train_dataset = Dataset(X_train, y_train)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=FLAGS.batch_size, shuffle=True)

        eval_dataset = Dataset(X_eval, y_eval)
        eval_loader = data.DataLoader(dataset=eval_dataset, batch_size=FLAGS.batch_size)

        num_input_features = train_dataset.num_input_features
        num_output_features = train_dataset.num_output_features

        # print(str(num_input_features) + ', ' + str(num_output_features))

        # create logger
        logger_name = 'Logger%02d' % (count)
        log_file = os.path.join(LOG_DIR, 'model%02d.log' % (count))
        logger = setup_logger(logger_name, log_file)

        myModel = Seq2SeqModel(FLAGS, num_input_features, num_output_features, logger)
        myModel.build_model()

        print('Training Model%02d...' % (count))

        # restore model
        existed_models = sorted(glob.glob(os.path.join(LOG_DIR, 'model%02d_*.pth') % (count)))
        # print(existed_models)
        # print(int(existed_models[-1][-8:-4]))
        start_epoch = 0
        if(len(existed_models) > 0):
            start_epoch = int(existed_models[-1][-8:-4])
            myModel.load_model(existed_models[-1])

        for epoch in range(start_epoch, FLAGS.num_epochs):
            print('------------------- Epoch %d -------------------' % (epoch+1))
            myModel.logger.info('Epoch: %d' % (epoch+1))
            start_time = time.time()
            myModel.train_per_epoch(train_loader, eval_loader)
            end_time = time.time()
            print('Time: %.6f' % (end_time-start_time))
            myModel.logger.info('Time: %.6f' % (end_time-start_time))

            if((epoch + 1) % FLAGS.save_steps == 0):
                model_save_path = os.path.join(LOG_DIR, 'model%02d_%04d.pth' % (count, epoch+1))
                myModel.save_model(model_save_path)
            
            if(FLAGS.enable_plot and (epoch + 1) % FLAGS.plot_result_interval == 0):
                plt.plot(myModel.train_loss_list, 'r', label='train_loss')
                plt.plot(myModel.eval_loss_list, 'b', label='eval_loss')
                plt.legend()
                plt.savefig('loss%02d_%04d.png' % (count, epoch+1))
                plt.plot(myModel.eval_mae_list, 'r', label='eval_mae')
                plt.plot(myModel.eval_rmse_list, 'b', label='eval_rmse')
                plt.legend()
                plt.savefig('mae%02d_%04d.png' % (count, epoch+1))

        del myModel

        count += 1

def train_seq2seq_cross_scenario(FLAGS, LOG_DIR, features_list, labels_list, scenarios_included=[1,2,3,4,5,6,7,8,9,10]):
    for i in range(0, int(len(features_list)/3)):
        X_train = np.concatenate(features_list[:(i*3)]+features_list[((i+1)*3):], axis=0)
        y_train = np.concatenate(labels_list[:(i*3)]+labels_list[((i+1)*3):], axis=0)

        X_test = np.concatenate(features_list[(i*3):((i+1)*3)], axis=0)
        y_test = np.concatenate(labels_list[(i*3):((i+1)*3)], axis=0)

        # X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=FLAGS.valid_size, random_state=50)

        print(X_train.shape)
        print(y_train.shape)
        # print(X_eval.shape)
        # print(y_eval.shape)
        
        train_dataset = Dataset(X_train, y_train)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=FLAGS.batch_size, shuffle=True)

        # eval_dataset = Dataset(X_eval, y_eval)
        eval_dataset = Dataset(X_test, y_test)
        eval_loader = data.DataLoader(dataset=eval_dataset, batch_size=FLAGS.batch_size, shuffle=True)

        num_input_features = train_dataset.num_input_features
        num_output_features = train_dataset.num_output_features

        # print(str(num_input_features) + ', ' + str(num_output_features))

        # create logger
        logger_name = 'Logger%02d' % (scenarios_included[i])
        log_file = os.path.join(LOG_DIR, 'model%02d.log' % (scenarios_included[i]))
        logger = setup_logger(logger_name, log_file)

        myModel = Seq2SeqModel(FLAGS, num_input_features, num_output_features, logger)
        myModel.build_model()

        print('Training Model%02d...' % (scenarios_included[i]))

        # restore model
        existed_models = sorted(glob.glob(os.path.join(LOG_DIR, 'model%02d_*.pth') % (scenarios_included[i])))
        # print(existed_models)
        # print(int(existed_models[-1][-8:-4]))
        start_epoch = 0
        if(len(existed_models) > 0):
            start_epoch = int(existed_models[-1][-8:-4])
            myModel.load_model(existed_models[-1])

        for epoch in range(start_epoch, FLAGS.num_epochs):
            print('------------------- Epoch %d -------------------' % (epoch+1))
            myModel.logger.info('Epoch: %d' % (epoch+1))
            start_time = time.time()
            myModel.train_per_epoch(train_loader, eval_loader)
            end_time = time.time()
            print('Time: %.6f' % (end_time-start_time))
            myModel.logger.info('Time: %.6f' % (end_time-start_time))

            if((epoch + 1) % FLAGS.save_steps == 0):
                model_save_path = os.path.join(LOG_DIR, 'model%02d_%04d.pth' % (scenarios_included[i], epoch+1))
                myModel.save_model(model_save_path)
            
            if(FLAGS.enable_plot and (epoch + 1) % FLAGS.plot_result_interval == 0):
                plt.plot(myModel.train_loss_list, 'r', label='train_loss')
                plt.plot(myModel.eval_loss_list, 'b', label='eval_loss')
                plt.legend()
                plt.savefig('loss%02d_%04d.png' % (scenarios_included[i], epoch+1))
                plt.plot(myModel.eval_mae_list, 'r', label='eval_mae')
                plt.plot(myModel.eval_rmse_list, 'b', label='eval_rmse')
                plt.legend()
                plt.savefig('mae%02d_%04d.png' % (scenarios_included[i], epoch+1))

        del myModel

def train_seq2seq_no_test(FLAGS, LOG_DIR, features_list, labels_list):
    features_train = []
    labels_train = []
    features_eval = []
    labels_eval = []

    # for i in range(0, len(features_list)):
    #     if(i % 3 == 2):
    #         features_eval.append(features_list[i])
    #         labels_eval.append(labels_list[i])
    #     else:
    #         features_train.append(features_list[i])
    #         labels_train.append(labels_list[i])
    
    # X_train = np.concatenate(features_train, axis=0)
    # y_train = np.concatenate(labels_train, axis=0)
    # X_eval = np.concatenate(features_eval, axis=0)
    # y_eval = np.concatenate(labels_eval, axis=0)
    
    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)

    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=FLAGS.valid_size, random_state=50)

    if(FLAGS.training_size < 1.0):
        X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=(1.0 - FLAGS.training_size), random_state=50)

    # X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=FLAGS.valid_size, random_state=50)

    print(X_train.shape)
    print(y_train.shape)
    print(X_eval.shape)
    print(y_eval.shape)
        
    train_dataset = Dataset(X_train, y_train)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=FLAGS.batch_size, shuffle=True)

    eval_dataset = Dataset(X_eval, y_eval)
    eval_loader = data.DataLoader(dataset=eval_dataset, batch_size=FLAGS.batch_size)

    num_input_features = train_dataset.num_input_features
    num_output_features = train_dataset.num_output_features

    print(str(num_input_features) + ', ' + str(num_output_features))

    # create logger
    logger_name = 'Logger'
    log_file = os.path.join(LOG_DIR, 'model.log')
    logger = setup_logger(logger_name, log_file)

    myModel = Seq2SeqModel(FLAGS, num_input_features, num_output_features, logger)
    myModel.build_model()

    print('Training Model...')

    # restore model
    existed_models = sorted(glob.glob(os.path.join(LOG_DIR, 'model_*.pth')))
    # print(existed_models)
    # print(int(existed_models[-1][-8:-4]))
    start_epoch = 0
    if(len(existed_models) > 0):
        start_epoch = int(existed_models[-1][-8:-4])
        myModel.load_model(existed_models[-1])

    for epoch in range(start_epoch, FLAGS.num_epochs):
        print('------------------- Epoch %d -------------------' % (epoch+1))
        myModel.logger.info('Epoch: %d' % (epoch+1))
        start_time = time.time()
        myModel.train_per_epoch(train_loader, eval_loader)
        end_time = time.time()
        print('Time: %.6f' % (end_time-start_time))
        myModel.logger.info('Time: %.6f' % (end_time-start_time))

        if((epoch + 1) % FLAGS.save_steps == 0):
            model_save_path = os.path.join(LOG_DIR, 'model_%04d.pth' % (epoch+1))
            myModel.save_model(model_save_path)
            
        if(FLAGS.enable_plot and (epoch + 1) % FLAGS.plot_result_interval == 0):
            plt.plot(myModel.train_loss_list, 'r', label='train_loss')
            plt.plot(myModel.eval_loss_list, 'b', label='eval_loss')
            plt.legend()
            plt.savefig('loss%02d_%04d.png' % (epoch+1))
            plt.plot(myModel.eval_mae_list, 'r', label='eval_mae')
            plt.plot(myModel.eval_rmse_list, 'b', label='eval_rmse')
            plt.legend()
            plt.savefig('mae%02d_%04d.png' % (epoch+1))

    del myModel

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # parse args first
    parser = get_parser()
    FLAGS, unknown = parser.parse_known_args()

    LOG_DIR = os.path.join(FLAGS.model_save_dir, FLAGS.model_id)
    if(not os.path.exists(LOG_DIR)):
        # create LOG_DIR
        try:
            os.makedirs(LOG_DIR)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # This was not a "directory exist" error..
    FLAGS_path = os.path.join(LOG_DIR, 'args.txt')
    if (not os.path.exists(FLAGS_path)):
        # save args
        with open(FLAGS_path, 'w') as log:
            for arg in sorted(vars(FLAGS)):
                log.write(arg + ': ' + str(getattr(FLAGS, arg)) + '\n')  # log of arguments
    else:
        # load FLAGS from file
        FLAGS = load_args_from_file(FLAGS_path, FLAGS)
    print(FLAGS)

    # load data and prepare data pairs
    scenarios_included=[1,2,3,4,5,6,7,8,9,10]
    features_list, labels_list = get_data_local(FLAGS, scenarios_included)
   
    # train
    if(FLAGS.no_test):
        train_seq2seq_no_test(FLAGS, LOG_DIR, features_list, labels_list)
    else:
        if(FLAGS.cross_scenario_validation):
            train_seq2seq_cross_scenario(FLAGS, LOG_DIR, features_list, labels_list, scenarios_included)
        else:
            train_seq2seq_10fold(FLAGS, LOG_DIR, features_list, labels_list)