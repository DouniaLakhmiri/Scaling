import torch
import gc
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import os
import sys
from evaluator import *
from neural_net2 import *
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

# for batch in testloader:
#     print(batch[0].size())
#
# for batch in trainloader:
#     print(batch[0].size())
#
initial_image_size = int(32*resolution)
total_classes = 10
number_input_channels = 3

model = NeuralNet(depth, width, initial_image_size)
# model = NeuralNet(1, 1)
model.to(device)
print(model)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# The evaluator trains and tests the network
evaluator = Evaluator(device, model, trainloader, testloader, optimizer, batch_size, dataset)
print('> Training')
# try:
# best_val_acc, best_epoch, nb_epochs = evaluator.train_and_test()
# print(best_val_acc)
cnt = 1
dsize = (1, 3, initial_image_size, initial_image_size)
inputs = torch.randn(dsize).to(device)
macs, params = profile(model, (inputs,), verbose=False)
#
# # Output of the blackbox
# print('> Final accuracy %.3f' % best_val_acc)

# For ResNet18 with an image size of 32
macs_baseline = 556651520
flops_baseline = 11173962

ratio_macs = macs / macs_baseline - 2.5
ratio_flops = params / flops_baseline - 2.5
print('MACS and FLOPS', ratio_macs, ratio_flops)
