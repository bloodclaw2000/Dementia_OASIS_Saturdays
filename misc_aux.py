import pandas as pd
import pickle
import bz2
import _pickle as cPickle


def getDataFrameFromDict(dict):
    properties = ['ID', 'M/F', 'Hand', 'Age', 'Educ', 'SES',
                  'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF', 'Delay', 'Dementia']
    subset = [[value['1'][prop] for prop in properties]
              for key, value in dict.items()]
    return pd.DataFrame(subset, columns=properties)


def dropparameters(dataframe, keys=[]):
    d2 = dataframe
    for key in keys:
        d2 = d2.drop(key, axis=1)
    return d2

def pet_save(pet, filename):
    pickle.dump(pet, open(filename, "wb"))
# #guardamos en un binario
# def compressed_pickle(data):
#     with bz2.BZ2File(f"/content/drive/MyDrive/grupo1-saturdaysAI/data/save_dict3" + '.pbz2', 'w') as f:
#                cPickle.dump(data, f)

# Auxiliares para pickle

def pet_load(file):
    # pickle.load(open(f"/content/drive/MyDrive/grupo1-saturdaysAI/data/image_data.p", "rb"))
    return pickle.load(open(file, "rb"))


def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

class Logger:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.verbosity = 1
    def log(self, message, level):
        if self.verbosity  >= level:
            print(message)
            if self.file_path is not None:
                with open(self.file_path, 'a') as file:
                    file.write(f"{level}: {message}\n")