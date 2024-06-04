import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import sys
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import pickle
import _pickle as cPickle
import bz2
import csv
import torch
import torchvision
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import random
from torchvision import transforms
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
# from sklearn.tree import export_graphviz
from IPython.display import Image
# import graphviz
from sklearn import tree

from pickle_aux import pet_load, decompress_pickle, pet_save
from dementia_network_class import Dementia, train_nn, train_nn_auto, getOutput
from dementia_tree_class import train_tree,  Customtree
from hyperparameter_optimization import get_optimal_params, find_hyperparams_optuna, visualizeresults

import time
import plotly
"""
Version 1.9.1 of Project
28/05/2024
"""
torch.multiprocessing.set_start_method(
    'spawn', force=True)  # kinda important mostly for CPU

device = torch.device(
    f'cuda:{torch.cuda.current_device()}'
    if torch.cuda.is_available()
    else 'cpu')


# pickle_directory = f"/content/drive/MyDrive/grupo1-saturdaysAI/data/save_dict3.p"
compressed_pickle_directory = "save_dict3"
if not os.path.exists('{0}_decompressed.p'.format(compressed_pickle_directory)):
    def force_dementia(dictionary):
        for key in dictionary:
            for key2 in dictionary[key]:
                if dictionary[key][key2]['CDR'] == '':
                    dictionary[key][key2]['Dementia'] = 0
                elif float(dictionary[key][key2]['CDR']) > 0:
                    dictionary[key][key2]['Dementia'] = 1
                else:
                    dictionary[key][key2]['Dementia'] = 0
        return dictionary

    def removeyoung(dictionary, age):
        dic_pacientes_viejos = {}
        for key in dictionary:
            for key2 in dictionary[key]:
                if int(dictionary[key][key2]['Age']) >= age:
                    dic_pacientes_viejos[key] = dictionary[key]
        return dic_pacientes_viejos

    tmp_dict = decompress_pickle(
        '{0}.pbz2'.format(compressed_pickle_directory))
    tmp_dict = force_dementia(tmp_dict)  # esto es la funcion del init
    tmp_dict = removeyoung(tmp_dict, 59)
    tmp_dict = {int(key): value for key, value in tmp_dict.items()}
    new_dict = {}
    current_index = 0
    for key in sorted(tmp_dict.keys()):
        new_dict[current_index] = tmp_dict[key]
        current_index += 1
    tmp_dict = new_dict

    pet_save(tmp_dict, '{0}_decompressed.p'.format(
        compressed_pickle_directory))
else:
    tmp_dict = pet_load('{0}_decompressed.p'.format(
        compressed_pickle_directory))

def createnecessaryfolders():
    if not os.path.exists(f"plots/"):
        os.makedirs(f"plots/")
    if not os.path.exists(f"studies/"):
        os.makedirs(f"studies/")
    if not os.path.exists(f"params_nn/"):
        os.makedirs(f"params_nn/")
    if not os.path.exists(f"nn/"):
        os.makedirs(f"nn/")
    if not os.path.exists(f"logs/"):
        os.makedirs(f"logs/")
    if not os.path.exists(f"hyperparams/"):
        os.makedirs(f"hyperparams/")
    if not os.path.exists(f"dataset/"):
        os.makedirs(f"dataset/")
    if not os.path.exists(f"params_tree/"):
        os.makedirs(f"params_tree/")
# device = "cpu"

# we create a random seed and torch random generator:


torch.set_default_device(device)

print(f" Using {device} in this run")

logpath = 'logs/'
createnecessaryfolders()
obj = Dementia(dictionary=tmp_dict, device=device)



# Podemos setear los parámetros para el entrenamiento aquí:
obj.setParam('image_type', 'T88_111')
obj.setParam('image_number', 1)

# print(tabulate(source_df, headers="keys", tablefmt="grid"))



time1 = time.time()

parameter_ranges = {
    'patience_validation': [3],
    'patience_plateau': [3],
    'delta_min': [0, 0.001, 0.01, 0.1],
    'batch_size': [10, 20, 30, 40, 50],
    'split_size': [0.7, 0.8, 0.9],
    'max_loss_reset': [1, 3, 5, 7, 10],
    'learning_rate': [0.0001, 0.0001, 0.00001],
    'weight_decay': [0.1, 0.01, 0.1, 0.2],
    'first_conv_outchann': [6, 8, 12],
    'second_conv_outchann': [16, 20, 24],
    'fclayer1': [120, 150, 200],
    'fclayer2': ['None', 100, 150],
    'optimizer': ['Adam', 'SGD']
}

# train_nn(obj, 'params_nn/', ['T88', 'FSL', 'RAW_1', 'RAW_2', 'RAW_3'],logpath = logpath)
# train_nn(obj, 'params_nn/', ['T88'],logpath = logpath)
# train_nn_auto(obj, 'params_nn/', ['T88', 'FSL', 'RAW_1', 'RAW_2', 'RAW_3'],logpath = logpath,experiments=parameter_ranges)

if os.path.exists('tmp_nn.p'):
    obj = pet_load('tmp_nn.p')
    print(f"USing old unfinished dataset split with seed {obj.seed}")
find_hyperparams_optuna(obj, 'params_nn/','hyperparams/','studies/', ['RAW_1'],logpath = logpath, n_trials= 2)
visualizeresults('studies/RAW_1.pbz2')

#train_nn_auto(obj, 'params_nn/', ['FSL'],logpath = logpath,experiments=parameter_ranges,max_iterations_ximage=10)


# para llamar al optimizador sin entrenar
# df = pd.read_csv('_T88.csv')
# get_optimal_params(df)


print(f"Normal network running time",time.time()- time1, 's')
# train {'plot': 'True', 'image_type': 'T88_111', 'image_number': 1, 'patience_validation': 3, 'patience_plateau': 3, 'validation_patience': 3, 'delta_min': 0, 'batch_size': 10, 'split_size': 0.8, 'max_loss_reset': 5, 'learning_rate': 0.0001, 'weight_decay': 0.1, 'nepochs': 1000, 'first_conv_outchann': 6, 'second_conv_outchann': 16, 'fclayer1': 120, 'fclayer2': 'None', 'criterion_type': 'BCElogitsloss', 'optimizer': 'Adam', 'verbosity': 0}
#       {'image_number': 1, 'patience_validation': 3, 'patience_plateau': 3, 'delta_min': 0, 'batch_size': 10, 'split_size': 0.7, 'max_loss_reset': 1, 'learning_rate': 0.0001, 'weight_decay': 0.1, 'first_conv_outchann': 6, 'second_conv_outchann': 16, 'fclayer1': 120, 'fclayer2': 'None', 'optimizer': 'Adam'}
#       {'plot': 'True', 'image_type': 'T88_111', 'image_number': 1, 'patience_validation': 3, 'patience_plateau': 3, 'validation_patience': 3, 'delta_min': 0, 'batch_size': 10, 'split_size': 0.8, 'max_loss_reset': 5, 'learning_rate': 0.0001, 'weight_decay': 0.1, 'nepochs': 1000, 'first_conv_outchann': 6, 'second_conv_outchann': 16, 'fclayer1': 120, 'fclayer2': 'None', 'criterion_type': 'CrossEntropyLoss', 'optimizer': 'Adam', 'verbosity': 0}
#       {'image_number': 1, 'patience_validation': 3, 'patience_plateau': 3, 'delta_min': 0, 'batch_size': 10, 'split_size': 0.7, 'max_loss_reset': 1, 'learning_rate': 0.0001, 'weight_decay': 0.1, 'first_conv_outchann': 6, 'second_conv_outchann': 16, 'fclayer1': 120, 'fclayer2': 'None', 'optimizer': 'Adam'}

#optimized neural network class
# getOutput(tmp_dict,'nn/', ['T88', 'FSL', 'RAW_1', 'RAW_2', 'RAW_3'],device = device)


# source_df = pd.read_csv('results.csv')
# treeclass = Customtree(source_df)

# #treeclass.write_dict_to_file('params_tree/treeparam.txt')

# train_tree(treeclass,'params_tree/treeparam.txt')
