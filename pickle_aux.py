import pickle
import bz2
import _pickle as cPickle

# Auxiliares para pickle
def pet_save(pet,filename):
    pickle.dump(pet, open(filename, "wb"))
# #guardamos en un binario
# def compressed_pickle(data):
#     with bz2.BZ2File(f"/content/drive/MyDrive/grupo1-saturdaysAI/data/save_dict3" + '.pbz2', 'w') as f:
#                cPickle.dump(data, f)
def pet_load(file):
    #pickle.load(open(f"/content/drive/MyDrive/grupo1-saturdaysAI/data/image_data.p", "rb"))
    return pickle.load(open(file, "rb"))
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

