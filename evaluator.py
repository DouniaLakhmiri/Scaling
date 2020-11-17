# ------------------------------------------------------------------------------
#  HyperNOMAD - Hyper-parameter optimization of deep neural networks with
#		        NOMAD.
#
#
#
#  This program is free software: you can redistribute it and/or modify it
#  under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or (at your
#  option) any later version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
#  for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#  You can find information on the NOMAD software at www.gerad.ca/nomad
# ------------------------------------------------------------------------------

import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn as nn
# from datahandler import *
import numpy as np


# sys.path.append(os.environ.get('HYPERNOMAD_HOME')+"/src/blackbox/blackbox")


class Evaluator(object):
    def __init__(self, device, cnn, trainloader, testloader, optimizer, batch_size, dataset):
        self.__device = device
        self.__cnn = cnn
        self.__trainloader = trainloader
        self.__testloader = testloader
        self.__batch_size = batch_size
        self.__optimizer = optimizer
        self.__dataset = dataset
        self.__train_acc = None
        self.__val_acc = None
        self.__test_acc = None
        self.__best_epoch = None

    @property
    def device(self):
        return self.__device

    @property
    def cnn(self):
        return self.__cnn

    @cnn.setter
    def cnn(self, new_cnn):
        self.__cnn = new_cnn

    @property
    def trainloader(self):
        return self.__trainloader

    @property
    def testloader(self):
        return self.__testloader

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def dataset(self):
        return self.__dataset

    def train_and_test(self):

        if torch.cuda.is_available():
            self.cnn = torch.nn.DataParallel(self.cnn)
            cudnn.benchmark = True

        epoch = 0
        stop = False
        best_test_acc = 0
        best_epoch = 0
        max_epochs = 200

        if self.dataset == 'MINIMNIST':
            max_epochs = 50

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=150, verbose=True)

        while (not stop) and (epoch < max_epochs):

            train_loss, train_acc = self.train()

            test_loss, test_acc = self.test()

            # save weights of best test score
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                torch.save(self.cnn.state_dict(), 'best_model.pth')

            print("Epoch {},  Train loss: {:.3f}, Train accuracy: {:.3f}, Val loss: {:.3f}, Val accuracy: {:.3f}, "
                  "Best val acc: {:.3f}".format(epoch + 1, train_loss, train_acc, test_loss, test_acc, best_test_acc))
            epoch += 1

            if self.optimizer.__class__.__name__ == 'SGD':
                scheduler.step(test_loss)

        print('> Finished Training')
        print('Best validation accuracy and corresponding epoch number : {:.3f}/{}'.format(
            best_test_acc, best_epoch + 1))
        return best_test_acc, best_epoch, epoch

    def train(self):
        self.cnn.train()
        criterion = nn.CrossEntropyLoss().cuda()
        train_loss = 0
        train_acc = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.cnn(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            train_acc = 100. * correct / total

        return train_loss, train_acc

    def test(self):
        criterion = nn.CrossEntropyLoss().cuda()
        total_test = 0
        correct_test = 0
        self.cnn.eval()
        test_loss = 0
        test_acc = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.cnn(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total_test += targets.size(0)
                correct_test += predicted.eq(targets).sum().item()
                test_acc = 100. * correct_test / total_test

        return test_loss, test_acc
