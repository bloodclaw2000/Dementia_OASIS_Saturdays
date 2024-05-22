import torch
from torch import nn


def calculate_first_linear_coef(inchannel,orwidth, orheight, nconv, npool, kernconv, channelconv):
  outchannel = inchannel
  width = orwidth
  height = orheight
  for i in range(0,nconv):
    # print(i)
    outchannel = channelconv[i]
    width = width - (kernconv - 1)
    height = height - (kernconv -1)
    if i in range(npool):
            width = width //2
            height = height //2
  return outchannel * width *height

#same as before
def calculate_validation_loss(model, dataloader, loss_function, BCE = True):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # No need to compute gradients during validation
              for inputs, targets, id in dataloader:
                        outputs = model(inputs)
                        if BCE:
                            loss = loss_function(outputs, targets.float())
                        else:
                            loss = loss_function(outputs, targets)
                        total_loss += loss.item()
                        num_batches += 1
                        
    return total_loss

#added a new function to reset the counter  
class ValidationLossEarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
              # number of times to allow for no improvement before stopping the execution
              self.patience = patience
              self.min_delta = min_delta  # the minimum change to be counted as improvement
              self.counter = 0  # count the number of times the validation accuracy not improving
              self.min_validation_loss = 10

    # return True when validation loss is not decreased by the `min_delta` for `patience` times
    def early_stop_check(self, validation_loss):
              if ((validation_loss+self.min_delta) < self.min_validation_loss):
                        self.min_validation_loss = validation_loss
                        self.counter = 0  # reset the counter if validation loss decreased at least by min_delta
              elif ((validation_loss+self.min_delta) > self.min_validation_loss):
                        # increase the counter if validation loss is not decreased by the min_delta
                        self.counter += 1
                        if self.counter > self.patience:
                            return True
              return False
    def reset_counter(self, val = True):
              self.counter = 0
              if val:
                        self.min_validation_loss = 10

def reset_model_weights(layer): #resets the coefficients of the NN if not working correctly
    if hasattr(layer, 'reset_parameters'):
              layer.reset_parameters()
    else:
              if hasattr(layer, 'children'):
                        for child in layer.children():
                            reset_model_weights(child)


    def conv2d_size_out(size, kernel_size=5, stride=1, padding=0):
              return (size - kernel_size + 2 * padding) // stride + 1
    
    size = width
    size = conv2d_size_out(size, kernel_size)
    size //= pool1
    size = conv2d_size_out(size, kernel_size)
    size //= pool2
    return size * size * conv_layers_out[-1]

#OBviously copu this
class Net(nn.Module):
    # from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    # llok this later https://readmedium.com/en/https:/towardsdatascience.com/convolutional-neural-network-for-image-classification-with-implementation-on-python-using-pytorch-7b88342c9ca9

    def __init__(self,width,height,channel,first_conv_out = 6, second_conv_out = 16, fclayer1 = 120, fclayer2 = 'None',BCE = True):
              super(Net, self).__init__()
              
              #here we create the convolution layers
              self.layer1 = torch.nn.Sequential(#first convolutional layer
                        nn.BatchNorm2d(channel), #normalize data 
                        nn.Conv2d(channel, first_conv_out, 5, 1), #convolutional (remember the 2d transform with the matrix)
                        nn.LeakyReLU(inplace = True), # an activation function NonLineal and works with minus
                        nn.MaxPool2d(2, 2), #half the amount of 2D data
              )
              
              self.layer2 = torch.nn.Sequential( #second convolutional layer
                        nn.Conv2d(first_conv_out, second_conv_out, 5), #convolutional (remember the 2d transform with the matrix)
                        torch.nn.ReLU(inplace = True), # an activation function NonLineal 
                        torch.nn.MaxPool2d(kernel_size=2, stride=2),
                        )
              
              """self.layer3 = torch.nn.Sequential(
                        NO TOUCHY THIS; IDK YET WHY no worky
                        self.fc1,
                 torch.nn.LeakyReLU(inplace = True)
                        )"""

              # para calcular http://layer-calc.com/
              #with this we calculate in_features to the fully connected layers (the NN)
              n = calculate_first_linear_coef(channel, width, height, 2, 2, 5, [first_conv_out, second_conv_out])
              
              self.fc1 = nn.Linear(n, fclayer1) #first fully connected layer
              torch.nn.init.xavier_uniform_(self.fc1.weight) #initialize random weigths
              if fclayer2 != 'None':
                        self.fc2 = nn.ReLU()
                        if BCE:
                            self.fc3 = nn.Linear(fclayer1, 1) #second fully connected layer
                        else:
                            self.fc3 = nn.Linear(fclayer1, 2) #second fully connected layer
                        torch.nn.init.xavier_uniform_(self.fc3.weight)

                        self.extrarelu = True
              else:
                        if BCE:
                            self.fc3 = nn.Linear(fclayer1, 1) #second fully connected layer
                        else:
                            self.fc3 = nn.Linear(fclayer1, 2) #second fully connected layer
                        self.extrarelu = False

                        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):   
              #print('Lay1 Shape: {}'.format(x.shape))
              x = self.layer1(x)  #apply the first conv layer
              #print('Conv1 Shape: {}'.format(x.shape))
              x = self.layer2(x) #apply the second conv layer
              #print('Lay2 Shape: {}'.format(x.shape))

              #x = self.layer3(x)
              #print('Lay3 Shape: {}'.format(x.shape))

              # Flatten operation: purely functional, outputs a (N, 120) Tensor
              x = torch.flatten(x, 1) 
              #x = x.view(x.size(0), -1)   # Flatten them for FC
              x = self.fc1(x) #aply first fully conv layer
              if self.extrarelu:
                        x = self.fc2(x) #apply second fully conv layer
              x = self.fc3(x)
              #x = self.fc3(x)

              return x
