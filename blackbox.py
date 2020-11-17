import torch
import gc
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import os
import sys
from evaluator import *
from neural_net import *
from resnet import *
from datahandler import DataHandler
from thop import clever_format, profile

gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data set
dataset = 'CIFAR10'
indx = 1
depth = float(sys.argv[indx])
depth = int(round(depth))
width = float(sys.argv[indx+1])
resolution = float(sys.argv[indx+2])

batch_size = 256
# Dataloaders
dataloader = DataHandler(dataset, batch_size)
image_size, number_classes = dataloader.get_info_data
trainloader, testloader = dataloader.get_loaders(resolution=resolution)
#
initial_image_size = 32
total_classes = 10
number_input_channels = 3

model = NeuralNet(depth, width)
# model = resnet18()
# model = NeuralNet(1.5, 1.5)
model.to(device)
print(model)

# optimizer = optim.RMSprop(model.parameters())
optimizer = optim.Adam(model.parameters())
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# The evaluator trains and tests the network
evaluator = Evaluator(device, model, trainloader, testloader, optimizer, batch_size, dataset)
print('> Training')
# try:
best_val_acc, best_epoch, nb_epochs = evaluator.train_and_test()
print(best_val_acc)
cnt = 1
dsize = (1, 3, 32, 32)
inputs = torch.randn(dsize).to(device)
macs, params = profile(model, (inputs,), verbose=False)
#
# # Output of the blackbox
print('> Final accuracy %.3f' % best_val_acc)
print('Count eval', cnt)
print('Number of epochs ', nb_epochs)
print('MACS and NB_PARAMS', macs, params)
