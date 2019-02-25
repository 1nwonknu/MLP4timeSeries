import torch
import time
from tensorboardX import SummaryWriter
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import fxdata
import netFactory


gVerbose = False

def trainingLog(epoch, it, numBatches, loss, acc):
    print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'
                  'Acc {acc:.3f}'.format(
                   epoch, it, numBatches, loss=loss, acc=acc))


def train(model, device, train_loader, writer, optimiser, epoch, n_iter, vizStep=100, logStep=25):
    model.train() # this enables parameter updates during backpropagation as well
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
            r2 = getRsquared(outputs, labels, inputs.shape[0])
            # print training status
            trainingLog(epoch,i, len(train_loader), loss.item(), r2)

        if i % logStep == 0:
            # Compute accuracy and write values to Tensorboard
            acc = getRsquared(outputs, labels, inputs.shape[0])
            writer.add_scalar('train/loss', loss.item(), n_iter)
            writer.add_scalar('train/acc', r2, n_iter)

        n_iter += inputs.shape[0]

    return n_iter


def getRsquared(outputs, labels, num):
    """ Can only be used for multi step ahead foredasts"""

    y_hat = outputs.data.numpy() if torch.is_tensor(outputs) else outputs

    y_np = labels.data.numpy() if torch.is_tensor(labels) else labels

    y_bar = np.sum(y_np) / len(y_np)

    ssreg = np.sum((y_hat - y_bar) ** 2)

    sstot = np.sum((y_np - y_bar) ** 2)

    r2 = ssreg / sstot

    return r2


def test(model, device, test_loader, writer):

    model.eval() # no update of trainable parameters (e.g. batch norm)
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

def appendDateAndTimeToString(string):
    now = datetime.utcnow().strftime("%m_%d_%H_%M_%S")
    return string + "_" + now


def createTensorBoardWriter(resultsDirectory):
    dir = appendDateAndTimeToString(resultsDirectory + '/')
    writer = SummaryWriter(dir)
    print("TensorBoardX writer created directory in %s" % dir)
    return writer, dir


def main(numEpoch, use_cuda=False, lr=0.1, n_step_ahead_predictions = 1, loock_bak = 2):

    # Create Dataset loaders

    d = fxdata.DB()
    d.startDB()
    df = d.readFxData(start='2016-01-01', end = '2016-08-01', version=2, name='EURGBP').data

    aggregator = fxdata.Aggregator(df=df, interval='1H')
    aggregator.aggregate()

    df = aggregator.ag_df

    start = '2016-06-23 00:00:00'
    end = '2016-06-24 20:00:00'

    slicer = fxdata.Slicer(df, start, end,
                           n_step_ahead_prediction=n_step_ahead_predictions,
                           loockback= loock_bak)
    trainLoader = slicer.getTrainingSet()

    start = '2016-06-27 01:00:00'
    end = '2016-06-28 23:00:00'

    slicer = fxdata.Slicer(df, start, end,
                           n_step_ahead_prediction=n_step_ahead_predictions,
                           loockback=loock_bak)

    testLoader = slicer.getTrainingSet()

    # Create instance of Network
    net = netFactory.NetFactory(hidden_layers=[loock_bak, loock_bak],
        n_input=loock_bak, n_output=n_step_ahead_predictions).createNeuralNetwork()

    writer, dir = createTensorBoardWriter('./results/' + net.name + '/lr' + str(lr))

    # Display Network Architecture
    print(net)

    # Define optimiser
    optim = torch.optim.SGD(net.parameters(), lr=lr)

    # Define define where training/testing will take place
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Launching training on: %s" % device)

    # Send model to device
    net = net.to(device)

    # Main training loop
    n_iter = 0 # used as batch id for Tensorboard visualisations
    print("-------------------------------------------------")
    for epoch in range(1, numEpoch + 1):
        t0 = time.time()
        n_iter = train(net, device, trainLoader, writer, optim, epoch, n_iter)
        print("Epoch took: %.3f s" % (time.time() - t0))
        print("-------------------------------------------------")

    # Evaluate on test set
    predictions = test(net, device, testLoader, writer)

    if gVerbose:
        print ("predictions")
        print(predictions)

        print("start")
        print(start)

        print("end")
        print(end)

    start = df.index.get_loc(start)
    end = df.index.get_loc(end)

    if gVerbose:
        print("start")
        print(start)
        print("end")
        print(end)
    actual = df.iloc[start:end].values

    if gVerbose:
        print("actual")
        print(actual)

        print(len(predictions))
        print(len(actual))

    plt.plot(predictions, label = "predicted")
    plt.plot(actual, label="actual")
    plt.show()


if __name__ == '__main__':
    # execute the main loop
    main(numEpoch=5, use_cuda=False, lr=0.1)

    def launchTensorBoard():
        import os
        os.system('tensorboard --port 7001 --logdir=' + r"E:\deep_learning_time_series\results\Net\lr0.1")
        return

    import threading
    t = threading.Thread(target=launchTensorBoard, args=([]))
    t.start()