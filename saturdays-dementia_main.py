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
from dementia_network_class import Dementia, train_nn, getOutput
from dementia_tree_class import train_tree,  Customtree


import time

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


# device = "cpu"

# we create a random seed and torch random generator:


torch.set_default_device(device)

print(f" Using {device} in this run")

logpath = 'logs/'

obj = Dementia(dictionary=tmp_dict, device=device)



# Podemos setear los parámetros para el entrenamiento aquí:
obj.setParam('image_type', 'T88_111')
obj.setParam('image_number', 1)

# print(tabulate(source_df, headers="keys", tablefmt="grid"))



time1 = time.time()
# train(obj,'params/',['T88'])
# train(obj,'params/',['T88','FSL'])
#train_nn(obj, 'params_nn/', ['T88', 'FSL', 'RAW_1', 'RAW_2', 'RAW_3'],logpath = logpath)
print(f"Normal network running time",time.time()- time1, 's')
#optimized neural network class

# getOutput('nn/',['T88'], device = device)
# getOutput('nn/',['T88','FSL'], device = device)
getOutput('nn/', ['T88', 'FSL', 'RAW_1', 'RAW_2', 'RAW_3'],device = device)

source_df = pd.read_csv('results.csv')
treeclass = Customtree(source_df)

#treeclass.write_dict_to_file('params_tree/treeparam.txt')

train_tree(treeclass,'params_tree/treeparam.txt')
