'''
Main file for creating simulated data or loading real data
and running RegGNN and sample selection methods.
'''

import argparse
import pickle

import torch
import numpy as np

# import proposed_method.data_utils as data_utils
import evaluators
from config import Config

import pandas as pd
import seaborn as sns
import os
import gc
import h5py
import hdf5plugin


import UTR as utr



def return_h5_path(file):
    path = f'/Users/tele/open-problems-multimodal/{file}'
    return path

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['infer'],
                    help="Creates data and topological features OR make inferences on data")

parser.add_argument('--model', default='RegGNN', choices=['RegGNN'],
                    help="Chooses the inference model that will be used")

parser.add_argument('--data-source', default='simulated', choices=['simulated', 'saved'],
                    help="Simulates random data or loads from path in config")

parser.add_argument('--measure', default='eigen',
                    choices=['abs', 'geo', 'tan', 'node', 'eigen', 'close', 'concat_orig', 'concat_scale'],
                    help="Chooses the topological measure to be used")

opts = parser.parse_args()

opts.mode = 'infer'
opts.model = 'RegGNN'

if opts.mode == 'infer':
    '''
    Cross validation will be used to train and generate inferences
    on the data saved in the folder specified in config.py.

    Overall MAE and RMSE will be printed and predictions will be saved
    in same data folder.
    '''
    print(f"{opts.model} will be run on the data.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mae_evaluator = lambda p, s: np.mean(np.abs(p - s))
    rmse_evaluator = lambda p, s: np.sqrt(np.mean((p - s) ** 2))

    if opts.model == 'RegGNN':
        preds, scores, _ = evaluators.evaluate_RegGNN(shuffle=Config.SHUFFLE, random_state=Config.MODEL_SEED,
                                                      dropout=Config.RegGNN.DROPOUT, k_list=Config.SampleSelection.K_LIST,
                                                      lr=Config.RegGNN.LR, wd=Config.RegGNN.WD, device=device,
                                                      sample_selection=Config.SampleSelection.SAMPLE_SELECTION,
                                                      num_epoch=Config.RegGNN.NUM_EPOCH,
                                                      n_select_splits=Config.SampleSelection.N_SELECT_SPLITS)
        if Config.SampleSelection.SAMPLE_SELECTION:
            mae_arr = [mae_evaluator(p, s) for p, s in zip(preds.values(), scores.values())]
            rmse_arr = [rmse_evaluator(p, s) for p, s in zip(preds.values(), scores.values())]
            print(f"For k in {Config.SampleSelection.K_LIST}:")
            print(f"Mean MAE +- std over k: {np.mean(mae_arr):.3f} +- {np.std(mae_arr):.3f}")
            print(f"Min, Max MAE over k: {np.min(mae_arr):.3f}, {np.max(mae_arr):.3f}")
            print(f"Mean RMSE +- std over k: {np.mean(rmse_arr):.3f} +- {np.std(rmse_arr):.3f}")
            print(f"Min, Max RMSE over k: {np.min(rmse_arr):.3f}, {np.max(rmse_arr):.3f}")
        else:
            print(f"MAE: {mae_evaluator(preds, scores):.3f}")
            print(f"RMSE: {rmse_evaluator(preds, scores):.3f}")

    else:
        raise Exception("Unknown argument.")

    with open(f"{Config.RESULT_FOLDER}preds.pkl", 'wb') as f:
        pickle.dump(preds, f)
    with open(f"{Config.RESULT_FOLDER}scores.pkl", 'wb') as f:
        pickle.dump(scores, f)

    print(f"Predictions are successfully saved at {Config.RESULT_FOLDER}.")

else:
    raise Exception("Unknown argument.")
