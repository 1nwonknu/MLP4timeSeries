import torch.nn as nn
import torch
import numpy as np
from prettytable import PrettyTable


class NetFactory():
    def __init__(self, hidden_layers, n_input=1, n_output=1):
        self.hidden_layers = hidden_layers
        self.n_input = n_input
        self.n_output = n_output

    def createNeuralNetwork(self):

        hidden_layers = self.hidden_layers
        loock_back = self.n_input
        numClasses = self.n_output

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.name = "Net"

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

            def forward(self, x):
                x = x.view(-1, x.shape[1])

                for i, layer in enumerate(self.params):
                    if i != len(self.params):
                        x = torch.sigmoid(layer(x))
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

        class Linear(torch.nn.Linear):
            def __init__(self, in_features, out_features, bias):
                super(Linear, self).__init__(in_features, out_features, bias=bias)

                self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
                self.bias = torch.nn.Parameter(torch.Tensor(out_features))

            def forward(self, input):
                return torch.nn.functional.linear(input, self.weight, self.bias)

        return Net()