import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.simplefilter(action='ignore',category=UserWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchnet
import os
from LoadMNIST import LoadMNIST
from LoadCIFAR import LoadCIFAR10
from Network import BinarizedMNISTNetwork, BinarizedCIFARNetwork
from BinarizedModules import *
import argparse

class Model():

    def __init__(self,
                dataset=None,
                pt_file=None,
                batch_size=256):

        assert pt_file is not None, "A trained binarized model is required!"
        self.batch_size = batch_size
        if dataset == 'MNIST':
            self.network = BinarizedMNISTNetwork()
            self.train_loader, self.validation_loader, self.test_loader = LoadMNIST(self.batch_size)
        else:
            self.network = BinarizedCIFARNetwork()
            self.train_loader, self.validation_loader, self.test_loader = LoadCIFAR10(self.batch_size)

        self.network = torch.nn.DataParallel(self.network)
        self.network.load_state_dict(torch.load(pt_file))
        self.network.eval()
        self.network = self.network.cuda()
        device = torch.device("cuda:0")
        self.network.to(device)
        # Ensure that the weights of the model are binarized
        for m in self.network.modules():
            if isinstance(m, BinarizeConv2d) or isinstance(m, BinarizeLinear):
                m.weight.data = torch.sign(m.weight.data)

    def test(self):
        correct = 0
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target, volatile=True)
            output = self.network(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        print('Accuracy: {}/{} ({:.0f}%)'.format(
            correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
        validation_accuracy = 100. * float(correct) / float(len(self.test_loader.dataset))
        print(validation_accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Binarized weight inference routine')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--trained_model', type=str)
    args = parser.parse_args()
    assert args.dataset == 'MNIST' or args.dataset == 'CIFAR-10', 'dataset must be either MNIST or CIFAR-10'
    batch_size = args.batch_size
    test_model = Model(dataset=args.dataset, pt_file=args.trained_model, batch_size=batch_size)
    test_model.test()
