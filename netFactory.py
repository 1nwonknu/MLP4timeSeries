import torch.nn as nn
import torch
import numpy as np
from prettytable import PrettyTable


class Logger():

    def __init__(self):
        pass

    def log(self, epoch, it, numBatches, loss, acc):
        return('Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss:.4f}\t'
              'Acc {acc:.3f}'.format(
            epoch, it, numBatches, loss=loss, acc=acc))


class NetFactory():
    def __init__(self, performance_func, hidden_layers, n_input=1, n_output=1):
        self.hidden_layers = hidden_layers
        self.n_input = n_input
        self.n_output = n_output
        self.performance_func = performance_func


    def createNeuralNetwork(self):

        hidden_layers = self.hidden_layers
        loock_back = self.n_input
        numClasses = self.n_output

        class Net(nn.Module):
            def __init__(self, performance_func, logger):
                super(Net, self).__init__()
                self.performance_func = performance_func
                self.name = "Net"
                self.logger = logger

                assert hidden_layers[0] == loock_back

                self.params = nn.ModuleList()
                self.params.append(nn.Linear(loock_back, hidden_layers[1]))
                if len(hidden_layers) == 2:
                    self.params.append(nn.Linear(hidden_layers[1], numClasses))
                else:
                    for i in range(1, len(hidden_layers)-1):
                        self.params.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                    self.params.append(nn.Linear(hidden_layers[i], numClasses))

                self.activation = nn.Sigmoid()
                self.criterion = nn.MSELoss()

            def trainingLog(self, epoch, it, numBatches, loss, acc):
                print (self.logger.log(epoch, it, numBatches, loss, acc))


            def forward(self, x):
                x = x.view(-1, x.shape[1])

                for i, layer in enumerate(self.params):
                    if i != len(self.params):
                        x = layer(x)
                    else:
                        x = layer(x)
                return x

            def __getPrettyTensorShape(self, tensor):
                return [elem for elem in np.array(tensor.size())]

            def __getArchAsTable(self):

                variablesList = []
                totalSize = 0
                table = PrettyTable(['VarName', 'Shape', 'Size(kB)', 'Size(%)'])
                for name, param in self.named_parameters():
                    size = (np.product(list(map(int, param.shape)))) / (1024.0)  # get variable size in kiloBytes
                    variablesList.append([name, self.__getPrettyTensorShape(param), np.round(size, decimals=2)])
                    totalSize += size

                for elem in variablesList:
                    table.add_row([elem[0], elem[1], elem[2], np.round(100 * elem[2] / totalSize, decimals=1)])

                return table, totalSize

            def __str__(self):
                table, totalSize = self.__getArchAsTable()

                return "TRAINABLE VARIABLES INFORMATION - arch: %s \n" \
                       "%s \n Total (trainable) size: %f kB" % (self.name, table, totalSize)

            def train_(self, device, train_loader, writer, optimiser, epoch, n_iter, vizStep=100, logStep=25):

                model = self

                model.train()  # this enables parameter updates during backpropagation as well
                # as other updates such as those in batch-normalisation layers

                # for every mini-batch containing batch_size images...
                for i, data in enumerate(train_loader, 0):

                    # zero gradients from previous step
                    optimiser.zero_grad()

                    # send the data (images, labels) to the device (either CPU or GPU)
                    inputs, labels = data[0].to(device), data[1].to(device)

                    # forward pass
                    # this executes the forward() method in the model
                    outputs = model(inputs)

                    # compute loss
                    loss = model.criterion(outputs, labels)

                    # backward pass
                    loss.backward()

                    # evaluate trainable parameters
                    optimiser.step()

                    # the code below is just for monitoring purposes using print() statements
                    # as well as writing certain values to TensorBoard
                    if i % vizStep == 0:
                        #r2 = self.getRsquared(outputs, labels, inputs.shape[0])
                        r2 = self.performance_func(outputs, labels, inputs.shape[0])

                        # print training status
                        self.trainingLog(epoch, i, len(train_loader), loss.item(), r2)

                    if i % logStep == 0:
                        # Compute accuracy and write values to Tensorboard
                        #acc = self.getRsquared(outputs, labels, inputs.shape[0])
                        acc = self.performance_func(outputs, labels, inputs.shape[0])

                        writer.add_scalar('train/loss', loss.item(), n_iter)
                        writer.add_scalar('train/acc', r2, n_iter)

                    n_iter += inputs.shape[0]

                return n_iter





            def test(self, device, test_loader, writer):
                model = self

                model.eval()  # no update of trainable parameters (e.g. batch norm)
                predictions = []
                with torch.no_grad():
                    total = 0

                    # now we evaluate every test image and compute the predicted labels
                    for data in test_loader:
                        # send data to device
                        images, labels = data[0].to(device), data[1].to(device)

                        # pass the images through the network
                        outputs = model(images)

                        # obtain predicted labels
                        predicted = outputs.data
                        predictions.append(predicted)

                        total += labels.size(0)

                # compute accuracy

                return predictions

        class Linear(torch.nn.Linear):
            def __init__(self, in_features, out_features, bias):
                super(Linear, self).__init__(in_features, out_features, bias=bias)

                self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
                self.bias = torch.nn.Parameter(torch.Tensor(out_features))

            def forward(self, input):
                return torch.nn.functional.linear(input, self.weight, self.bias)

        return Net(self.performance_func, logger=Logger())
