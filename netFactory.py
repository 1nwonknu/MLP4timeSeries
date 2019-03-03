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


class Net(nn.Module):
    def __init__(self, performance_func, logger, hidden_layers, loock_back, numClasses):
        super(Net, self).__init__()
        self.performance_func = performance_func
        self.name = "Net"
        self.logger = logger

        if len(hidden_layers) > 1:
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

        for i, layer in enumerate(self.params):
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

            inputs, labels = data[0].to(device), data[1].to(device)

            # forward pass
            # this executes the forward() method in the model
            outputs = model(inputs)

            outputs = outputs.view(-1, outputs.shape[1])

            # compute loss
            loss = model.criterion(outputs, labels)

            # backward pass
            loss.backward()

            # evaluate trainable parameters
            optimiser.step()

            if i % vizStep == 0:
                r2 = self.performance_func(outputs, labels, inputs.shape[0])
                self.trainingLog(epoch, i, len(train_loader), loss.item(), r2)

            if i % logStep == 0:
                # Compute accuracy and write values to Tensorboard
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

            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)

                # pass the images through the network
                outputs = model(images)

                # obtain predicted labels
                predicted = outputs.data
                predictions.append(predicted)

                total += labels.size(0)

        # compute accuracy

        return predictions


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

        class MLP(Net):
            def __init__(self, performance_func, logger, hidden_layers, loock_back, numClasses):
                super(MLP, self).__init__(performance_func=performance_func,
                                          logger=logger,
                                          hidden_layers=hidden_layers,
                                          loock_back=loock_back,
                                          numClasses=numClasses)

            def forward(self, x):
                x = x.view(-1, x.shape[1])
                return super(MLP, self).forward(x)

        class Linear(torch.nn.Linear):
            def __init__(self, in_features, out_features, bias):
                super(Linear, self).__init__(in_features, out_features, bias=bias)

                self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
                self.bias = torch.nn.Parameter(torch.Tensor(out_features))

            def forward(self, input):
                return torch.nn.functional.linear(input, self.weight, self.bias)

        return MLP(performance_func=self.performance_func,
                   logger=Logger(),
                   hidden_layers=hidden_layers,
                   loock_back= loock_back ,
                   numClasses=numClasses)

    def createConvNet(self):
        hidden_layers = self.hidden_layers
        loock_back = self.n_input
        numClasses = self.n_output

        class ConvNet(Net):
            def __init__(self,  performance_func, logger, hidden_layers, loock_back, numClasses):
                super(ConvNet, self).__init__(performance_func=performance_func,
                                              logger=logger,
                                              hidden_layers=hidden_layers,
                                              loock_back=loock_back,
                                              numClasses=numClasses)

                self.layer1 = nn.Sequential(
                nn.Conv1d(in_channels=loock_back, out_channels=numClasses, kernel_size=1, stride=1),#, 1, kernel_size=1, stride=1, padding=2),
                #nn.BatchNorm1d(loock_back),
                #nn.Sigmoid(),
                nn.MaxPool1d(kernel_size=1, stride=2))

                self.layer2 = nn.Sequential(
                nn.Conv1d(in_channels=numClasses, out_channels=numClasses, kernel_size=1),# stride=1, padding=2),
                #nn.BatchNorm1d(loock_back),
                #nn.Sigmoid(),
                nn.MaxPool1d(kernel_size=1))#, stride=2))

                #self.layer3 = nn.Dropout(0.005)

                self.layer4 = nn.Sequential(
                    nn.Conv1d(in_channels=numClasses, out_channels=numClasses, kernel_size=1),  # stride=1, padding=2),
                    # nn.BatchNorm1d(loock_back),
                    # nn.Sigmoid(),
                    nn.MaxPool1d(kernel_size=1))  # , stride=2))


                #self.fc = nn.Linear(loock_back, numClasses)

                self.params = nn.ModuleList()
                self.params.append(self.layer1)
                self.params.append(self.layer2)
                #self.params.append(self.layer3)
                #self.params.append(self.layer4)

                #self.params.append(self.fc)

            def forward(self, x):
                x = x.view(-1, x.shape[1], 1)
                return super(ConvNet, self).forward(x)

        return ConvNet(performance_func=self.performance_func,
                   logger=Logger(),
                   hidden_layers=hidden_layers,
                   loock_back=loock_back,
                   numClasses=numClasses)