import os
import torch
import pandas as pd
import numpy as np
import torchvision
import torch.optim as optim

from torch import nn

from misc_aux import getDataFrameFromDict
from custom_dataset import CustomTransform,CustomDataset,getDatasetIDS
from custom_network import Net,ValidationLossEarlyStopping,calculate_validation_loss,reset_model_weights

def train(obj,folder_path,process_files):
    files = os.listdir(folder_path)
    # Iterate through each file
    for file_name in files:
              filepath=os.path.join(folder_path, file_name)
              # Check if the current item is a file (not a folder)
              if os.path.isfile(filepath) and 'output' not in filepath and 'skip' not in filepath and (len(process_files)==0 or (os.path.splitext(os.path.basename(filepath))[0]  in process_files) ):
                        obj.read_params_from_file_and_set(filepath)
                        print(f"Now calculating predictions for file {filepath}")
                        file_basename = os.path.basename(filepath)  # Get the basename with extension
                        file_without_extension = os.path.splitext(file_basename)[0]
                        # Open the file in write mode ('w')
                        with open(os.path.join(folder_path, '{0}_output.txt'.format(file_without_extension)), 'w') as f:
                            # Redirect stdout to the file
                            #sys.stdout = f
                            print(obj.trainParam)

                            obj.setTransform([CustomTransform()])
                            
                            obj.train()

                            labels, predicted  = obj.test()
                            obj.matrix(labels, predicted)
                            
                            # df=obj.resultsToDF(imageName=file_without_extension)
                            obj.cacheDataSets(dataset=file_without_extension)

                            obj.save('nn/{0}.pth'.format(file_without_extension))
                            

                            #sys.stdout = sys.__stdout__


def predictNN(myNN,imageName,criterion):
    if not (os.path.exists('tmp_df.csv') and os.path.exists('dataset/'+imageName+'.pth')):
              print('Error, both tmp_df.csv and tmp_df_totaldataset.pth must be available ')
    else:
              # merged_df = pd.read_csv('tmp_df.csv')
              total_set = torch.load('dataset/'+imageName+'.pth')

    total_loader = torch.utils.data.DataLoader(total_set, batch_size=32, shuffle=False)

    predictions = []

    # Iterar sobre el conjunto de datos de prueba
    for data in total_loader:
            inputs, labels, _ = data

            # Realizar predicciones utilizando el modelo entrenado
            outputs = myNN(inputs)

            probabilities = torch.sigmoid(outputs)

            if criterion == 'BCElogitsloss':
                predicted = torch.sigmoid(outputs)
                # print('PROBABILIDADES:  ',probabilities)
            else:
                _, predicted = outputs
            # Agregar las predicciones y las etiquetas reales a las listas
            predictions.extend(predicted.tolist())

    total_ids=getDatasetIDS(total_set,'TOT')
    total_ids.reset_index(inplace=True)
    total_ids.rename(columns={'index': 'ID_IDX'}, inplace=True)     

    # print(predictions)

    predictions_df = pd.DataFrame(predictions,columns=['PRED_'+imageName])


    predictions_df.reset_index(inplace=True)
    predictions_df.rename(columns={'index': 'ID_IDX'}, inplace=True)

    merged_predictions_df = pd.merge(total_ids, predictions_df, on='ID_IDX', how='left')
    merged_predictions_df = merged_predictions_df[['ID','PRED_'+imageName]]

    # merged_df=pd.merge(merged_df, merged_predictions_df, on='ID', how='left')

    # No nos haría falta guardar el parcial.
    # merged_predictions_df.to_csv('{0}.csv'.format(imageName),index=False)

    return merged_predictions_df

def getOutput(folder_path,process_files):
    source_df = pd.read_csv('tmp_df.csv')
    source_df['ID'] = source_df['ID'].astype('int32')
    files = os.listdir(folder_path)
    # Iterate through each file
    for file_name in files:
              filepath=os.path.join(folder_path, file_name)

              # Check if the current item is a file (not a folder)
              if os.path.isfile(filepath) and (len(process_files)==0 or (os.path.splitext(os.path.basename(filepath))[0]  in process_files) ):
                        # obj.read_params_from_file_and_set(filepath)
                        print(f"Now running training for file {filepath}")
                        file_basename = os.path.basename(filepath)  # Get the basename with extension
                        file_without_extension = os.path.splitext(file_basename)[0]
                        # Open the file in write mode ('w')

                        myNN = torch.load('nn/{0}.pth'.format(file_without_extension))
                        myNN.eval()
                        
                        predictions = predictNN(myNN,file_without_extension,'BCElogitsloss')
                        predictions['ID'] = predictions['ID'].astype('int32')
                        source_df=pd.merge(source_df, predictions, on='ID', how='left')

                        source_df.to_csv('results.csv',index=False)

              else: 
                        print('Not found model for '+file_name)      

class Dementia:

    def __init__(self,dictionary,device,generator=None,seed=None):

              self.device = device
              self.dict = dictionary

              self.transform=None
              self.dataset=None
              self.train_set=None
              self.test_set=None
              self.validation_set=None
              self.nn=None
              self.criterion = None
              self.optimizer = None
              self.trainParam = {
                        'image_type' : 'FSL_SEG',
                        'image_number': 1,
                        'patience_validation':3,
                        'patience_plateau':3,
                        'validation_patience' : 3,
                        'delta_min':0,
                        'batch_size':10,
                        'split_size': 0.8,
                        'max_loss_reset': 5,
                        'learning_rate' : 0.0001,
                        'weight_decay' : 0.1,
                        'first_conv_outchann' : 6,
                        'second_conv_outchann' : 16,
                        'fclayer1' : 120,
                        'fclayer2' : 'None',
                        'criterion_type' : 'CrossEntropyLoss',
                        'optimizer' : 'Adam'
                        
                        }
              self.valid_params = self.trainParam.copy()

              #Utilizado como cache para entrenamientos dentro de la misma instancia
              #Aunque existe el self.data_set, este total_set se reconstruye a partir de los dataset de los split
              self.total_set=None
              self.total_loader=None

            #   self.random_generator = torch.Generator(device=self.device)
              if not generator:
                self.random_generator = torch.Generator(device=self.device)
              else:
                self.random_generator=generator  


              if not seed:
                self.seed = np.random.randint(1,3e8)
              else:
                self.seed=seed  


    def setParam(self,key,val):
              try:
                        if key not in self.valid_params:
                            print (f'Param {key} does not exist')
                            raise Exception()
                        self.trainParam[key]=val
              except Exception  as error:
                        print ('Param does not exist')
                        print('Resetting to default value')
    def read_params_from_file_and_set(self,file_path):
              print(f"Read parameters dictionary at relative location {file_path} and setting them ")

              with open(file_path, 'r') as file:
                        for line in file:
                            # Strip leading and trailing whitespace
                            line = line.strip()
                            # Skip lines that are empty or start with #
                            if not line or line.startswith('#'):
                                      continue
                            line = line.strip()
                            if '#' in line:
                                     line = line.split('#', 1)[0].strip()
                                     # Skip line if only comment remains after stripping
                                     if not line:
                                         continue
                           
                            #print(line)
                            key, value = line.strip().split(':')
                            # Convert value to the appropriate type
                            if value.isdigit():
                                      value = int(value)
                            elif value.replace('.', '', 1).isdigit() and '.' in value:
                                      value = float(value)
                            self.setParam(key, value)
                            
    def write_dict_to_file(self, file_path):
              
              with open(file_path, 'w') as file:
                        for key, value in self.trainParam.items():
                            file.write(f'{key}:{value}\n')
              print(f"wrote parameters dictionary at relative location {file_path}")

    def setTransform(self,transform):
              self.transform = torchvision.transforms.Compose(transform)
              
    def separate_datasets(self, train_size, test_size,validation_size):
                random_generator = self.random_generator
                seed = self.seed
                random_generator.manual_seed(seed)
                try:
                         #print(train_size, test_size, validation_size)
                         self.train_set, self.validation_set, self.test_set = torch.utils.data.random_split(
                         self.dataset, [train_size, validation_size, test_size],generator=random_generator)
                except Exception:
                         train_size = train_size + 1
                         self.train_set, self.validation_set, self.test_set = torch.utils.data.random_split(
                            self.dataset, [train_size, validation_size, test_size],generator=random_generator)
                #print(self.train_set, self.validation_set, self.test_set)
                ids = []
                for i, data in enumerate(self.validation_set, 0):
                          inputs, labels, id = data
                          ids.append(id)
                #print(ids)
                return self.train_set, self.validation_set, self.test_set
    
    def apply_train(self, plateaupatience = 2, valpatience = 3, min_delta = 0,batch_size = 10):
              #separate into train_set val_set test_set

              trainloader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size,
              shuffle=True,generator=torch.Generator(device=self.device))
              validationloader = torch.utils.data.DataLoader(self.validation_set, batch_size=batch_size,
              shuffle=True,generator=torch.Generator(device=self.device))

              early_stopper = ValidationLossEarlyStopping(patience=3, min_delta=0)
              plateaupatience = 2 #tries to get out of a plateau and decrease training loss
              counterplateau = 0 #counter for above

                #-----------------------------------------------------
                
                #let's run the NET
              for epoch in range(1000):  # loop over the dataset multiple times

                running_loss = 0.0
                counterbad = 0
                countergood = 0
                for i, data in enumerate(trainloader, 0):
                          # get the inputs; data is a list of [inputs, labels]
                          inputs, labels, id = data
                          """labels = labels \
                        .type(torch.FloatTensor) \
                        .reshape((labels.shape[0], 1))"""
                          for i in labels:
                                    if i == 0:
                                        counterbad += 1
                                    else:
                                        countergood += 1
                          self.optimizer.zero_grad()
                          output = self.nn.forward(inputs)
                          # print(output, labels)
                          if self.trainParam['criterion_type'] == 'BCElogitsloss':
                                    loss = self.criterion(output, labels.float())
                                    bce = True
                          else:
                                    loss = self.criterion(output, labels)
                                    bce = False

                          #scheduler.step()
                          loss.backward()

            #               print(loss.grad)
                          # torch.nn.utils.clip_grad_norm_(my_nn.parameters(), 5)
                          self.optimizer.step()

                          running_loss += loss.item()
                print(f'[{epoch + 1}] tr loss: {running_loss:.3f}')
                validation_loss = calculate_validation_loss(
                          self.nn, validationloader, self.criterion, bce)
                #scheduler.step(validation_loss)
                print(f'[{epoch + 1}] va loss: {validation_loss:.3f}')
                print(counterbad,countergood)
                if early_stopper.early_stop_check(validation_loss):
                          if running_loss < self.trainParam['max_loss_reset']:
                                    if counterplateau < plateaupatience:
                                        counterplateau +=1
                                        early_stopper.reset_counter(val = False)
                                        print("Trying to force it out of a plateau")
                                    else:
                                        print("couldnt force out of Plateau")
                                        print("STOPPING EARLY BECAUSE NO IMPROVEMENT")
                                        break
                          else:
                                    print("SHOULD STOP EARLY BUT ERROR TOO HIGH")
                                    for layer in self.nn.children(): #we reset the NN so it can find another minima
                                        reset_model_weights(layer)
                                    early_stopper.reset_counter()
                                    print("Resetting Lineal Coefficients")
                running_loss = 0.0

              print('Finished Training')
              #print(output)

    def save(self,filename):
              torch.save(self.nn, filename)
    
    
    def train(self,split_size=0.8,batch_size=10,num_workers=2, nepochs=10):
            if self.trainParam['criterion_type'] == 'BCElogitsloss':
                bce = True
            else:
                bce = False

            if (self.transform):
                self.dataset = CustomDataset(self.dict, transform=self.transform, image_type = self.trainParam['image_type'],image_number =  self.trainParam['image_number'],BCE=bce)
            else:
                self.dataset = CustomDataset(self.dict, image_type = self.trainParam['image_type'],image_number =  self.trainParam['image_number'],BCE = bce)

            train_size = int(self.trainParam['split_size'] * len(self.dataset))
            test_size = 3 * (len(self.dataset) - train_size)//4
            validation_size = 1 * (len(self.dataset) - train_size)//4

            self.train_set, self.validation_set, self.test_set = self.separate_datasets( train_size, test_size, validation_size)

            #vamos a conseguir el tamaño de la imagen
            data, label, id = self.train_set[0]
            # print(data.shape)
            channel = data.shape[0]
            height = data.shape[1]
            width = data.shape[2]
            self.nn = Net(width,height,channel,first_conv_out = self.trainParam['first_conv_outchann'],
                                      second_conv_out= self.trainParam['second_conv_outchann'] ,
                                      fclayer1 = self.trainParam['fclayer1'], fclayer2 = self.trainParam['fclayer2'],BCE = bce)
            
    #   import torchsummary
    #   torchsummary.summary(self.nn, (3,208,176))

            # #Let’s use a Classification Cross-Entropy loss and SGD with momentum.
            if self.trainParam['criterion_type'] == 'CrossEntropyLoss':
                self.criterion = nn.CrossEntropyLoss()
            elif self.trainParam['criterion_type'] == 'BCElogitsloss':
                self.criterion = nn.BCEWithLogitsLoss() #on development this afternoon
            self.optimizer = optim.Adam(self.nn.parameters(), lr=self.trainParam['learning_rate'],weight_decay = self.trainParam['weight_decay'])
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            self.apply_train(plateaupatience=self.trainParam['patience_plateau'],valpatience = self.trainParam['patience_validation'],
                                         min_delta = self.trainParam['delta_min'],batch_size=self.trainParam['batch_size'])
            
    def test(self):
             
              from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

              # Lista para almacenar las predicciones y las etiquetas reales
              predictions = []
              true_labels = []
              total_loss = 0
              total_samples = 0

              # Configuración del modelo en modo de evaluación
              self.nn.eval()

              # Definir el tamaño del lote para cargar los datos
              batch_size = 32

              # Crear el dataloader para el conjunto de datos de prueba
              testloader = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, shuffle=False)

              # Iterar sobre el conjunto de datos de prueba
              for data in testloader:
                        inputs, labels, _ = data

                        # Realizar predicciones utilizando el modelo entrenado
                        outputs = self.nn(inputs)
                        #print(outputs)
                        # Calculate loss if needed
                        if self.trainParam['criterion_type'] == 'BCElogitsloss':
                            loss = self.criterion(outputs, labels.float())
                        else:
                            loss = self.criterion(outputs, labels)
                        total_loss += loss.item() * inputs.size(0)
                        total_samples += labels.size(0)

                        probabilities = torch.sigmoid(outputs)
                        if self.trainParam['criterion_type'] == 'BCElogitsloss':
                            predicted = (torch.sigmoid(outputs) >= 0.5).int()
                            print('PROBABILIDADES TEST:  ',probabilities)
                        else:
                            _, predicted = torch.max(outputs, 1)
                        # Agregar las predicciones y las etiquetas reales a las listas
                        predictions.extend(predicted.tolist())
                        true_labels.extend(labels.tolist())
              
              print(predictions)
              print(true_labels)
              # Calcular métricas de rendimiento
              accuracy = accuracy_score(true_labels, predictions)
              
              precision = precision_score(true_labels, predictions)
              
              recall = recall_score(true_labels, predictions)
              
              f1 = f1_score(true_labels, predictions)
              
              average_loss = total_loss / total_samples

              # Imprimir las métricas de rendimiento
              print("Accuracy: {0}".format(accuracy))
              print("Precision: {0}".format(precision))
              print("Recall: {0}".format(recall))
              print("1-score: {0}".format(f1))
              print("Average Loss: {0}".format(average_loss))

              return true_labels,predictions
    
    def cacheDataSets(self,dataset='T88',force=False):
            if (not (os.path.exists('tmp_df.csv'))) or force:
                        dict=getDataFrameFromDict(self.dict)
                        train=getDatasetIDS(self.train_set,'T')
                        test=getDatasetIDS(self.test_set,'P')
                        val=getDatasetIDS(self.validation_set,'V')
                        
                        # print('validation',train,test,val)
                        #   print(prueba['DATAFRAMEIDX'])
                        merged_df = pd.merge(dict, train, on='ID', how='left')
                        merged_df = pd.merge(merged_df, test, on='ID', how='left')
                        merged_df = pd.merge(merged_df, val, on='ID', how='left')

                        def find_non_nan(row):
                                    for val in row:
                                                if pd.notnull(val):
                                                            return val
                                    return np.nan

                        # Apply the function to each row to create the new column
                        merged_df['USE'] = merged_df[['SUBSET_x', 'SUBSET_y', 'SUBSET']].apply(find_non_nan, axis=1)
                        merged_df.drop(columns=['SUBSET_x', 'SUBSET_y', 'SUBSET'], inplace=True)
                        # merged_df = ['ID', 'M/F', 'Hand', 'Age', 'Educ', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF', 'Delay', 'Dementia']
                        
                        merged_df.to_csv('tmp_df.csv',index=False)
            else: 
               merged_df = pd.read_csv('tmp_df.csv')
            
            total_set = torch.utils.data.ConcatDataset([self.train_set, self.test_set, self.validation_set])

            torch.save(total_set, 'dataset/'+dataset+'.pth')

            merged_df['ID'] = merged_df['ID'].astype('int32')               

            return merged_df,total_set


    def matrix(self, labels, predicted, plot=False):
        """
        Compute the confusion matrix and optionally plot it.

        Parameters:
        - labels (array-like): The true labels.
        - predicted (array-like): The predicted labels.
        - plot (bool, optional): Whether to plot the confusion matrix. Default is False.

        Returns:
        None

        Example usage:
        matrix([0, 1, 0, 1], [1, 1, 0, 0], plot=True)
        """

        from sklearn.metrics import confusion_matrix

        # Compute confusion matrix
        conf_matrix = confusion_matrix(labels, predicted)

        print("Confusion Matrix:")
        print(conf_matrix)

        import seaborn as sns
        import matplotlib.pyplot as plt

        if plot:
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.show()
