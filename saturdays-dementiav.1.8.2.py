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
#from sklearn.tree import export_graphviz
from IPython.display import Image
#import graphviz
from sklearn import tree

from pickle_aux import pet_load,decompress_pickle,pet_save
from dementia_network_class import Dementia,train, getOutput

torch.multiprocessing.set_start_method('spawn', force=True) #kinda important mostly for CPU

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

    tmp_dict = decompress_pickle('{0}.pbz2'.format(compressed_pickle_directory))
    tmp_dict = force_dementia(tmp_dict) #esto es la funcion del init
    tmp_dict = removeyoung(tmp_dict, 59)
    tmp_dict = {int(key): value for key, value in tmp_dict.items()}
    new_dict = {}
    current_index = 0
    for key in sorted(tmp_dict.keys()):
                        new_dict[current_index] = tmp_dict[key]
                        current_index += 1
    tmp_dict = new_dict
                        
    pet_save(tmp_dict,'{0}_decompressed.p'.format(compressed_pickle_directory))
else:
     tmp_dict=pet_load('{0}_decompressed.p'.format(compressed_pickle_directory))




#device = "cpu"

#we create a random seed and torch random generator:



torch.set_default_device(device)

print(device)

print(f" Using {device} in this run")

obj = Dementia(dictionary=tmp_dict,device=device)


#Podemos setear los parámetros para el entrenamiento aquí:
obj.setParam('image_type','T88_111')
obj.setParam('image_number',1)

         
    #print(tabulate(source_df, headers="keys", tablefmt="grid"))
def dropparameters(dataframe, keys = []):
    d2 = dataframe
    for key in keys:
        d2 = d2.drop(key,axis = 1)
    return d2
def runrandomforest(file_path, Paramsfile = ['ID', 'SES','CDR','Delay','USE','Hand','MMSE']):
    Paramsfile.append('Dementia')
    
    source_df = pd.read_csv('results.csv')
    source_df['M/F'] = source_df['M/F'].map({'F':1, 'M':0})
    df_train = source_df[source_df['USE'] == 'T']
    df_test = source_df[source_df['USE'] != 'T']
    X_train = dropparameters(df_train,Paramsfile)
    X_test =  dropparameters(df_test,Paramsfile)
    y_train = df_train['Dementia']
    y_test = df_test['Dementia']
    #print(list(X_train.columns))
    param_dist = {'n_estimators': np.random.randint(50,500,10),
              'max_depth': np.random.randint(2,10,4)}
    print(param_dist)
    rf = RandomForestClassifier()
    rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)
    rand_search.fit(X_train, y_train)
    print('Best hyperparameters:',  rand_search.best_params_)
    clf = rand_search.best_estimator_
    #clf = tree.DecisionTreeClassifier(max_depth = maxdepth)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    fn=X_train.columns
    #♣cn=X_train.columns
    
    for i in range(3):
        fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 600)
        tree.plot_tree(clf.estimators_[i],
                       feature_names = fn, 
                       class_names = ['healthy','demented'],
                       filled = True,
                       impurity = True
                       );
        fig.savefig(f'plottreefncn{i}.png')
        plt.figure()
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.show()
    accuracy = accuracy_score(y_test, y_pred)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    # Create a series containing feature importances from the model and feature names from the training data
    feature_importances = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    plt.show()
    # Plot a simple bar chart
    feature_importances.plot.bar()
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    return df_train,df_test
    
# train(obj,'params/',['T88'])
# train(obj,'params/',['T88','FSL'])
train(obj,'params/',['T88','FSL','RAW_1','RAW_2','RAW_3'])

# getOutput('nn/',['T88'])
# getOutput('nn/',['T88','FSL'])
getOutput('nn/',['T88','FSL','RAW_1','RAW_2','RAW_3'])
TreeParamsDrop= ['ID', 'SES','CDR','Delay','USE','Hand','Age','M/F','MMSE','eTIV','ASF','nWBV','Educ'] #only images
#TreeParamsDrop= ['ID', 'SES','CDR','Delay','USE','Hand','MMSE','PRED_FSL', 'PRED_RAW_1', 'PRED_RAW_2', 'PRED_RAW_3', 'PRED_T88'] #for testing without image predictions
aa,ab = runrandomforest('results.csv', Paramsfile = TreeParamsDrop)