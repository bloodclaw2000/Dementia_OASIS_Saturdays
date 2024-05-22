import torch
import pandas as pd

from functions_transform import fft_transform,border_transform

def getDatasetIDS (dataset,subset):
              ids=[]
              for data in dataset:
                        inputs, labels, id = data
                        ids.append([id,subset])
              # ids=np.array(ids)
              return pd.DataFrame(ids,columns=['ID','SUBSET'])

class CustomTransform(object):
    def __init__(self, split_percent=0.5):
              self.split_percent = split_percent

    # Defining the transform method
    def __call__(self, image):
              # Splitting the image into two parts

              image1 = fft_transform(image)
              image2 = border_transform(image)

              # Returning the two parts of the image
              return torch.tensor(image,dtype= torch.float32), torch.tensor(image1,dtype= torch.float32), torch.tensor(image2,dtype= torch.float32)
    
class CustomDataset(torch.utils.data.Dataset): #aqui habra que utilizar de lista de tipos de imgagen
    def __init__(self, data,transform = None, image_type = 'FSL_SEG', image_number = 1, BCE = True):
              self.data = data
              self.transform = transform
              self.image_type = image_type
              self.image_number = image_number
              self.BCE = BCE
              #self.img_labels = image_dict_orig
    def __len__(self):
              return len(self.data)

    def __getitem__(self, idx):
              #print(self.data['0001'])
              #image_array = self.data[idx]['i'][1]['image']
              #print(self.data.keys())
              image_array = self.data[idx]['1'][f'{self.image_type}'][self.image_number]['original']
              # Convert the 2D array to a PyTorch tensor
              tensor_image = torch.tensor(image_array, dtype=torch.float32)
              if self.BCE:
                        label = torch.tensor([self.data[idx]['1']["Dementia"]])
              else:
                        label = torch.tensor(self.data[idx]['1']["Dementia"])

              id = self.data[idx]['1']["ID"]

              # Applying the transform
              if self.transform:
                        tensor_image1, tensor_image2, tensor_image3 = self.transform(image_array)
                        tensor_image = torch.stack((tensor_image1,tensor_image2,tensor_image3))
              return tensor_image, label, id    