import os
import torch
import pandas as pd
import numpy as np
import torchvision
import torch.optim as optim

from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, f1_score

from misc_aux import getDataFrameFromDict, Logger
from custom_dataset import CustomTransform, CustomDataset, getDatasetIDS
from custom_network import Net, ValidationLossEarlyStopping, calculate_validation_loss, reset_model_weights
from matplotlib import pyplot as plt

import time

def train_nn(obj, folder_path, process_files, logpath = None):
    """
     Train the neural network on the files in the folder_path and write the predictions to the output

     Args:
         obj: The object to use for training
         folder_path: The folder where the files are to be read
         process_files: A list of files to process or
    """
    files = os.listdir(folder_path)
    # Iterate through each file
    for file_name in files:
        filepath = os.path.join(folder_path, file_name)
        if logpath is not None:
            timeofcreation = time.strftime('%Y_%m_%d_%H_%M_%S')
            loggingpath = logpath + os.path.splitext(os.path.basename(filepath))[0]+ "_"+ timeofcreation +"_log.txt"

        logger = Logger(file_path = loggingpath)
        obj.logger = logger
        # Check if the current item is a file (not a folder)
        if os.path.isfile(filepath) and 'output' not in filepath and 'skip' not in filepath and (len(process_files) == 0 or (os.path.splitext(os.path.basename(filepath))[0] in process_files)):
            obj.read_params_from_file_and_set(filepath)
            obj.logger.verbosity = obj.trainParam['verbosity']
            obj.logger.log(f"Max verbosity level on this run: {obj.trainParam['verbosity']}", obj.trainParam['verbosity'])

            print(f"Now training network from params {filepath}")
            # Get the basename with extension
            file_basename = os.path.basename(filepath)
            file_without_extension = os.path.splitext(file_basename)[0]
            # Open the file in write mode ('w')
            with open(os.path.join(folder_path, '{0}_output.txt'.format(file_without_extension)), 'w') as f:
                # Redirect stdout to the file
                # sys.stdout = f

                obj.setTransform([CustomTransform()])

                obj.train()

                labels, predicted = obj.test()
                obj.matrix(labels, predicted)

                # df=obj.resultsToDF(imageName=file_without_extension)
                obj.cacheDataSets(dataset=file_without_extension)

                obj.save('nn/{0}.pth'.format(file_without_extension))

                # sys.stdout = sys.__stdout__


def predictNN(myNN, imageName, criterion, device):
    """
     Funcion que realiza el prediccion a todos los datos del dataframe

     Args:
         myNN: objeto de neural network.
         imageName: nombre del archivo que contiene los registros
         criterion: Tipo de criterio de clasificación a utilizar

     Returns: 
         Devuelve un Dataframe con el resultado de la prediccion
    """
    if not (os.path.exists('tmp_df.csv') and os.path.exists('dataset/'+imageName+'.pth')):
        print('Error, both tmp_df.csv and tmp_df_totaldataset.pth must be available ')
    else:
        # merged_df = pd.read_csv('tmp_df.csv')
        total_set = torch.load('dataset/'+imageName+'.pth')

    total_loader = torch.utils.data.DataLoader(
        total_set, batch_size=32, shuffle=False, generator=torch.Generator(device=device))

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

    total_ids = getDatasetIDS(total_set, 'TOT')
    total_ids.reset_index(inplace=True)
    total_ids.rename(columns={'index': 'ID_IDX'}, inplace=True)

    # print(predictions)

    predictions_df = pd.DataFrame(predictions, columns=['PRED_'+imageName])

    predictions_df.reset_index(inplace=True)
    predictions_df.rename(columns={'index': 'ID_IDX'}, inplace=True)

    merged_predictions_df = pd.merge(
        total_ids, predictions_df, on='ID_IDX', how='left')
    merged_predictions_df = merged_predictions_df[['ID', 'PRED_'+imageName]]

    # merged_df=pd.merge(merged_df, merged_predictions_df, on='ID', how='left')

    # No nos haría falta guardar el parcial.
    # merged_predictions_df.to_csv('{0}.csv'.format(imageName),index=False)

    return merged_predictions_df


def getOutput(folder_path, process_files, device):
    """
     Creates a dataframe that contains the output of the neural network. This is used to calculate predictions from NN

     Args:
         folder_path: Path to the folder where the files are
         process_files: List of files to process if empty all files
    """
    source_df = pd.read_csv('tmp_df.csv')
    source_df['ID'] = source_df['ID'].astype('int32')
    files = os.listdir(folder_path)
    # Iterate through each file
    for file_name in files:
        filepath = os.path.join(folder_path, file_name)

        # Check if the current item is a file (not a folder)
        if os.path.isfile(filepath) and (len(process_files) == 0 or (os.path.splitext(os.path.basename(filepath))[0] in process_files)):
            # obj.read_params_from_file_and_set(filepath)
            print(f"Now running predictions for neural network {filepath}")
            # Get the basename with extension
            file_basename = os.path.basename(filepath)
            file_without_extension = os.path.splitext(file_basename)[0]
            # Open the file in write mode ('w')

            myNN = torch.load('nn/{0}.pth'.format(file_without_extension))
            myNN.eval()

            predictions = predictNN(
                myNN, file_without_extension, 'BCElogitsloss', device = device)
            predictions['ID'] = predictions['ID'].astype('int32')
            source_df = pd.merge(source_df, predictions, on='ID', how='left')

            source_df.to_csv('results.csv', index=False)

        else:
            print('Not found model for '+file_name)


class Dementia:

    def __init__(self, dictionary, device, generator=None, seed=None):
        """
          Initialize the class. This is the method that must be called by the user to initialize the class

          Args:
                   dictionary: Dictionary containing the parameters needed to train the model
                   device: Device on which to run the model.
                   generator: Random number generator to use
                   seed: Random seed
        """

        self.device = device
        self.dict = dictionary

        self.transform = None
        self.dataset = None
        self.train_set = None
        self.test_set = None
        self.validation_set = None
        self.nn = None
        self.criterion = None
        self.optimizer = None
        self.trainParam = {
            'plot': 'True',
            'image_type': 'FSL_SEG',
            'image_number': 1,
            'patience_validation': 3,
            'patience_plateau': 3,
            'validation_patience': 3,
            'delta_min': 0,
            'batch_size': 10,
            'split_size': 0.8,
            'max_loss_reset': 5,
            'learning_rate': 0.0001,
            'weight_decay': 0.1,
            'nepochs': 1000,
            'first_conv_outchann': 6,
            'second_conv_outchann': 16,
            'fclayer1': 120,
            'fclayer2': 'None',
            'criterion_type': 'CrossEntropyLoss',
            'optimizer': 'Adam',
            'verbosity': 0
        }
        self.valid_params = self.trainParam.copy()

        # Utilizado como cache para entrenamientos dentro de la misma instancia
        # Aunque existe el self.data_set, este total_set se reconstruye a partir de los dataset de los split
        self.total_set = None
        self.total_loader = None
        
        # sets random seed for dataset splitting
        if not seed:
            self.seed = np.random.randint(1, 3e8)
        else:
            self.seed = seed
            
        self.logger = Logger("somethingwentwrong.txt")
    def setParam(self, key, val):
        try:
            if key not in self.valid_params:
                self.logger.log(f'Param {key} does not exist, resetting to default value',0)
            else:
                self.trainParam[key] = val
        except Exception as error:
            pass
    def read_params_from_file_and_set(self, file_path):
        self.logger.log(
            f"Read parameters dictionary at relative location {file_path} and setting them ",0)

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

                # print(line)
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
        self.logger.log(f"wrote parameters dictionary at relative location {file_path}",0)

    def setTransform(self, transform):
        self.transform = torchvision.transforms.Compose(transform)

    def separate_datasets(self, train_size, test_size, validation_size):
        """
         Separate datasets into training validation and test sets. This is a helper function

         Args:
                 train_size: Number of examples in training set
                 test_size: Number of examples in test set ( inclusive )
                 validation_size: Number of examples in validation set ( inclusive )

         Returns: 
                 Sets of training validation and test datasets
        """
        random_generator = torch.Generator(device=self.device)
        seed = self.seed
        random_generator.manual_seed(seed)
        try:
            # print(train_size, test_size, validation_size)
            self.train_set, self.validation_set, self.test_set = torch.utils.data.random_split(
                self.dataset, [train_size, validation_size, test_size], generator=random_generator)
        except Exception:
            train_size = train_size + 1
            self.train_set, self.validation_set, self.test_set = torch.utils.data.random_split(
                self.dataset, [train_size, validation_size, test_size], generator=random_generator)
        self.logger.log("Train, Validation and Test Sets", 2)
        self.logger.log(f"{self.train_set}, {self.validation_set}, {self.test_set}",2)
        ids = []
        for i, data in enumerate(self.validation_set, 0):
            inputs, labels, id = data
            ids.append(id)
        self.logger.log(f"Ids {ids}",2)
        return self.train_set, self.validation_set, self.test_set

    def apply_train(self, plateaupatience=2, valpatience=3, min_delta=0, batch_size=10, nepochs=1000):
        """
         Train the network with plateaupatience and validation patience. This is a blocking function that returns when the network is ready to be used.

         Args:
           plateaupatience: The plateau patience to use for training
           valpatience: The validation patience to use for validation
           min_delta: The minimum amount of validation reduction to count
           batch_size: The batch size to use for training
           epochs: Maximum numnber of epochs to train the network in
        """
        # separate into train_set val_set test_set
        trainingEpoch_loss = []
        validationEpoch_loss = []
        testingEpoch_loss = []
        trainloader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size,
                                                  shuffle=True, generator=torch.Generator(device=self.device))
        validationloader = torch.utils.data.DataLoader(self.validation_set, batch_size=batch_size,
                                                       shuffle=True, generator=torch.Generator(device=self.device))

        early_stopper = ValidationLossEarlyStopping(patience=3, min_delta=0)
        plateaupatience = 2  # tries to get out of a plateau and decrease training loss
        counterplateau = 0  # counter for above
        # -----------------------------------------------------

        # let's run the NET
        for epoch in range(nepochs):  # loop over the dataset multiple times

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
                self.logger.log(f"{output}, {labels}",2)
                if self.trainParam['criterion_type'] == 'BCElogitsloss':
                    loss = self.criterion(output, labels.float())
                    bce = True
                else:
                    loss = self.criterion(output, labels)
                    bce = False

                # scheduler.step()
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(my_nn.parameters(), 5)
                self.optimizer.step()

                running_loss += loss.item()
            self.logger.log(f'[{epoch + 1}] tr loss: {running_loss:.3f}',1)
            validation_loss = calculate_validation_loss(
                self.nn, validationloader, self.criterion, bce)
            # scheduler.step(validation_loss)
            self.logger.log(f'[{epoch + 1}] va loss: {validation_loss:.3f}',1)
            self.logger.log(f"nondemented {counterbad},demented {countergood}",2)
            if early_stopper.early_stop_check(validation_loss):
                if running_loss < self.trainParam['max_loss_reset']:
                    if counterplateau < plateaupatience:
                        counterplateau += 1
                        early_stopper.reset_counter(val=False)
                        self.logger.log("Trying to force it out of a plateau",1)
                    else:
                        self.logger.log("couldnt force out of Plateau",1)
                        self.logger.log("STOPPING EARLY BECAUSE NO IMPROVEMENT",1)
                        break
                else:
                    self.logger.log("SHOULD STOP EARLY BUT ERROR TOO HIGH",1)
                    for layer in self.nn.children():  # we reset the NN so it can find another minima
                        reset_model_weights(layer)
                    early_stopper.reset_counter()
                    self.logger.log("Resetting Lineal Coefficients",1)
            trainingEpoch_loss.append(running_loss/len(self.train_set))
            validationEpoch_loss.append(validation_loss/len(self.validation_set))
            running_loss = 0.0

            #testingEpoch_loss.append(self.calculatetestloss())
        if self.trainParam['plot'] == 'True':
            plt.plot(trainingEpoch_loss, label='train_loss')
            plt.plot(validationEpoch_loss,label='val_loss')
            #plt.plot(testingEpoch_loss,label='testing loss')
            plt.legend()
            image = self.trainParam['image_type']
            number = self.trainParam['image_number']
            plt.title(f'{image},{number} ')
            plt.savefig(f'plots/{image}_{number}_loss_evolution')
            plt.show()
        self.logger.log(f'Finished Training in {epoch} epochs',0)
        # print(output)

    def save(self, filename):
        torch.save(self.nn, filename)

    def train(self, split_size=0.8, batch_size=10, num_workers=2, nepochs=1000):
        """
         Train the neural network. This is the main method for training the neural network

         Args:
             split_size: size of the split used to split the data
             batch_size: size of the batch used to train the network
             num_workers: number of worker processes to use for training
             nepochs: number of epochs to run the network ( default 10
        """
        
        self.logger.log(self.trainParam,1)

        if self.trainParam['criterion_type'] == 'BCElogitsloss':
            bce = True
        else:
            bce = False

        if (self.transform):
            self.dataset = CustomDataset(self.dict, transform=self.transform,
                                         image_type=self.trainParam['image_type'], image_number=self.trainParam['image_number'], BCE=bce)
        else:
            self.dataset = CustomDataset(
                self.dict, image_type=self.trainParam['image_type'], image_number=self.trainParam['image_number'], BCE=bce)

        train_size = int(self.trainParam['split_size'] * len(self.dataset))
        test_size = 3 * (len(self.dataset) - train_size)//4
        validation_size = 1 * (len(self.dataset) - train_size)//4

        self.train_set, self.validation_set, self.test_set = self.separate_datasets(
            train_size, test_size, validation_size)

        # vamos a conseguir el tamaño de la imagen
        data, label, id = self.train_set[0]
        self.logger.log(data.shape, 2)
        channel = data.shape[0]
        height = data.shape[1]
        width = data.shape[2]
        self.nn = Net(width, height, channel, first_conv_out=self.trainParam['first_conv_outchann'],
                      second_conv_out=self.trainParam['second_conv_outchann'],
                      fclayer1=self.trainParam['fclayer1'], fclayer2=self.trainParam['fclayer2'], BCE=bce)
    #   import torchsummary
    #   torchsummary.summary(self.nn, (3,208,176))

        # #Let’s use a Classification Cross-Entropy loss and SGD with momentum.
        if self.trainParam['criterion_type'] == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
        elif self.trainParam['criterion_type'] == 'BCElogitsloss':
            self.criterion = nn.BCEWithLogitsLoss()  # on development this afternoon
            
        self.optimizer = optim.Adam(self.nn.parameters(
        ), lr=self.trainParam['learning_rate'], weight_decay=self.trainParam['weight_decay'])
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        self.apply_train(plateaupatience=self.trainParam['patience_plateau'], valpatience=self.trainParam['patience_validation'],
                         min_delta=self.trainParam['delta_min'], batch_size=self.trainParam['batch_size'], nepochs=self.trainParam['nepochs'])

    def test(self):


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
        testloader = torch.utils.data.DataLoader(
            self.test_set, batch_size=batch_size, shuffle=False)

        # Iterar sobre el conjunto de datos de prueba
        for data in testloader:
            inputs, labels, _ = data

            # Realizar predicciones utilizando el modelo entrenado
            outputs = self.nn(inputs)
            self.logger.log(outputs,2)
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
                self.logger.log(f'PROBABILIDADES TEST: {probabilities}',2)
            else:
                _, predicted = torch.max(outputs, 1)
            # Agregar las predicciones y las etiquetas reales a las listas
            predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())
        
        self.logger.log(predictions,2)
        self.logger.log(true_labels,2)
        # Calcular métricas de rendimiento
        accuracy = accuracy_score(true_labels, predictions)

        precision = precision_score(true_labels, predictions)

        recall = recall_score(true_labels, predictions)

        f1 = f1_score(true_labels, predictions)

        average_loss = total_loss / total_samples

        # Imprimir las métricas de rendimiento
        self.logger.log("Accuracy: {0}".format(accuracy),0)
        self.logger.log("Precision: {0}".format(precision),0)
        self.logger.log("Recall: {0}".format(recall),0)
        self.logger.log("1-score: {0}".format(f1),0)
        self.logger.log("Average Loss: {0}".format(average_loss),0)

        return true_labels, predictions
    def calculatetestloss(self):
        total_loss = 0
        total_samples = 0

        # Configuración del modelo en modo de evaluación
        self.nn.eval()

        # Definir el tamaño del lote para cargar los datos
        batch_size = 32

        # Crear el dataloader para el conjunto de datos de prueba
        testloader = torch.utils.data.DataLoader(
            self.test_set, batch_size=batch_size, shuffle=False)

        # Iterar sobre el conjunto de datos de prueba
        for data in testloader:
            inputs, labels, _ = data

            # Realizar predicciones utilizando el modelo entrenado
            outputs = self.nn(inputs)
            self.logger.log("outputs of test",2)
            self.logger.log(outputs,2)
            # Calculate loss if needed
            if self.trainParam['criterion_type'] == 'BCElogitsloss':
                loss = self.criterion(outputs, labels.float())
            else:
                loss = self.criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_samples += labels.size(0)
        self.nn.train()
        return total_loss
    
    def cacheDataSets(self, dataset='T88', force=False):
        if (not (os.path.exists('tmp_df.csv'))) or force:
            dict = getDataFrameFromDict(self.dict)
            train = getDatasetIDS(self.train_set, 'T')
            test = getDatasetIDS(self.test_set, 'P')
            val = getDatasetIDS(self.validation_set, 'V')

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
            merged_df['USE'] = merged_df[['SUBSET_x', 'SUBSET_y', 'SUBSET']].apply(
                find_non_nan, axis=1)
            merged_df.drop(
                columns=['SUBSET_x', 'SUBSET_y', 'SUBSET'], inplace=True)
            # merged_df = ['ID', 'M/F', 'Hand', 'Age', 'Educ', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF', 'Delay', 'Dementia']

            merged_df.to_csv('tmp_df.csv', index=False)
        else:
            merged_df = pd.read_csv('tmp_df.csv')

        total_set = torch.utils.data.ConcatDataset(
            [self.train_set, self.test_set, self.validation_set])

        torch.save(total_set, 'dataset/'+dataset+'.pth')

        merged_df['ID'] = merged_df['ID'].astype('int32')

        return merged_df, total_set

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

        # Compute confusion matrix
        conf_matrix = confusion_matrix(labels, predicted)

        self.logger.log("Confusion Matrix:",1)
        self.logger.log(conf_matrix,1)

        import seaborn as sns
        import matplotlib.pyplot as plt

        if plot:
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            image = self.trainParam['image_type']
            number = self.trainParam['image_number']
            plt.savefig(f'plots/{image}_{number}_confusion_matrix')

            plt.savefig("plots/")
            plt.show()
